"""
Build precinct-level progressive lean scores for Chicago.

For each precinct, computes how much more (or less) progressive it voted
than the city as a whole, averaged across several reference elections.
Multiple score variants cover different candidate scenarios and 2019
sensitivity checks.

Sign convention
───────────────
Positive  → precinct is more progressive than the city
Negative  → precinct is less progressive than the city
Units      pp (percentage points)

Bloc margin
───────────
For each race we compute bloc_margin = (prog%) − (mod%), where shares are
normalized within labeled candidates only. Minor / unclassified candidates
are excluded from the denominator. Mayoral ideologies come from the
candidate_labels table; the "moderate bloc" pools Moderate + Conservative.

The precinct's score is its bloc_margin minus the city's bloc_margin, so
a precinct that matches the city scores ~0.

Citywide baselines are vote-weighted across precincts (votes / total),
not unweighted precinct averages.

Score variants written
──────────────────────
Generic            — all reference races except the 2019 mayoral runoff,
                     which is dropped because Lightfoot (Moderate) vs.
                     Preckwinkle (Progressive) is contaminated by both
                     being Black women with contested ideological labels.
                     The 2019 first round (14-candidate field with clean
                     bloc structure) is kept at full weight.
Black-progressive  — 2023 mayoral (1R + RO) + 2026 Senate (Stratton+Kelly v. Raja)
Latino-progressive — 2015 mayoral (1R + RO) + 2026 Comptroller (Karina v. Croke)

(White-progressive intentionally skipped — no viable historical example.)

Per-precinct standard error
───────────────────────────
SE_race ≈ 100 / √n_labeled, treating the bloc margin as roughly Bernoulli
within labeled votes. Across races we report SE of the weighted mean
assuming independence:  SE = √(Σ wᵣ² · seᵣ²) / Σ wᵣ.

Bloc compositions for 2026 statewides
─────────────────────────────────────
Senate DEM       progressive: Stratton, Kelly       moderate: Raja
Comptroller DEM  progressive: Karina Villa          moderate: Margaret Croke
(Other candidates in those races are unclassified and excluded.)

JoinField normalization
───────────────────────
The DB uses "CITY OF CHICAGO:Ward 01 Precinct 01" (mixed case); the 2026
statewide CSVs use "CITY OF CHICAGO:WARD 01 PRECINCT 01" (uppercase).
We uppercase both so they align.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DB_PATH = Path("/home/cole/databases/illinois_elections.db")
_STATEWIDE_DIR = Path(
    "/home/cole/elections/ilforecast_redesign/data/election_csvs/clean/2026/Primary/Statewide"
)
_OUTPUT_DIR = Path(__file__).parent / "outputs"
_OUTPUT_PATH = _OUTPUT_DIR / "precinct_progressive_scores.csv"

CHICAGO_PREFIX = "CITY OF CHICAGO:"


def _norm_joinfield(jf):
    return jf.upper() if isinstance(jf, str) else jf


@dataclass
class RaceData:
    name: str
    precincts: pd.DataFrame  # joinfield, prog_votes, mod_votes


def load_mayoral_race(year: int, election_type: str) -> RaceData:
    with sqlite3.connect(str(_DB_PATH)) as conn:
        df = pd.read_sql_query(
            """
            SELECT er.JoinField, er.votes, cl.ideology
            FROM election_results er
            JOIN candidates c
                 ON  c.candidate_name = er.candidate_name
                 AND c.election_type  = er.election_type
                 AND c.year           = er.year
                 AND c.race_type      = er.race_type
            LEFT JOIN candidate_labels cl
                 ON cl.normalized_name = c.normalized_name
            WHERE er.race_type      = 'chicago_mayor'
              AND er.year           = ?
              AND er.election_type  = ?
            """,
            conn,
            params=[year, election_type],
        )

    df["JoinField"] = df["JoinField"].map(_norm_joinfield)
    df["votes"] = df["votes"].fillna(0.0)

    prog = df[df["ideology"] == "Progressive"].groupby("JoinField")["votes"].sum()
    mod = (
        df[df["ideology"].isin(["Moderate", "Conservative"])]
        .groupby("JoinField")["votes"]
        .sum()
    )

    precincts = (
        pd.DataFrame({"prog_votes": prog, "mod_votes": mod})
        .fillna(0.0)
        .reset_index()
        .rename(columns={"JoinField": "joinfield"})
    )

    pretty = {"municipal": "first round", "municipal_runoff": "runoff"}.get(
        election_type, election_type
    )
    return RaceData(name=f"{year} mayoral {pretty}", precincts=precincts)


def load_statewide_csv(
    csv_path: Path,
    prog_candidates: list[str],
    mod_candidates: list[str],
    race_label: str,
) -> RaceData:
    df = pd.read_csv(str(csv_path))
    df = df[df["JoinField"].str.startswith(CHICAGO_PREFIX, na=False)].copy()
    df["joinfield"] = df["JoinField"].map(_norm_joinfield)

    missing = [c for c in prog_candidates + mod_candidates if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing expected columns {missing}")

    precincts = pd.DataFrame(
        {
            "joinfield": df["joinfield"].values,
            "prog_votes": df[prog_candidates].sum(axis=1).values,
            "mod_votes": df[mod_candidates].sum(axis=1).values,
        }
    )
    return RaceData(name=race_label, precincts=precincts)


def compute_precinct_lean(race: RaceData) -> pd.DataFrame:
    """Returns joinfield, lean_pp, se_pp, n_labeled for one race."""
    p = race.precincts.copy()
    p["n_labeled"] = p["prog_votes"] + p["mod_votes"]
    p = p[p["n_labeled"] > 0].copy()

    total_prog = p["prog_votes"].sum()
    total_mod = p["mod_votes"].sum()
    total_n = p["n_labeled"].sum()
    city_margin_pp = 100.0 * (total_prog - total_mod) / total_n

    p["precinct_margin_pp"] = 100.0 * (p["prog_votes"] - p["mod_votes"]) / p["n_labeled"]
    p["lean_pp"] = p["precinct_margin_pp"] - city_margin_pp
    p["se_pp"] = 100.0 / p["n_labeled"].clip(lower=1).pow(0.5)

    print(
        f"  {race.name:32s}  precincts={len(p):4d}  "
        f"city_margin={city_margin_pp:+6.2f}pp  "
        f"labeled_votes={int(total_n):,}"
    )
    return p[["joinfield", "lean_pp", "se_pp", "n_labeled"]].copy()


def combine_leans(
    race_leans: list[pd.DataFrame],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Weighted-mean lean across races, with SE assuming independence."""
    if weights is None:
        weights = [1.0] * len(race_leans)
    if len(weights) != len(race_leans):
        raise ValueError("weights length must match race_leans length")

    tagged = []
    for w, df in zip(weights, race_leans):
        if w <= 0 or df.empty:
            continue
        t = df.copy()
        t["weight"] = w
        tagged.append(t)

    if not tagged:
        return pd.DataFrame(
            columns=["joinfield", "score", "se", "n_races_used", "total_labeled"]
        )

    big = pd.concat(tagged, ignore_index=True)
    big["w_lean"] = big["weight"] * big["lean_pp"]
    big["w2_se2"] = (big["weight"] ** 2) * (big["se_pp"] ** 2)

    agg = big.groupby("joinfield").agg(
        sum_w=("weight", "sum"),
        sum_w_lean=("w_lean", "sum"),
        sum_w2_se2=("w2_se2", "sum"),
        n_races_used=("weight", "size"),
        total_labeled=("n_labeled", "sum"),
    )
    agg["score"] = agg["sum_w_lean"] / agg["sum_w"]
    agg["se"] = agg["sum_w2_se2"].pow(0.5) / agg["sum_w"]
    return agg[["score", "se", "n_races_used", "total_labeled"]].reset_index()


def main() -> None:
    print("Loading mayoral races...")
    mayoral = {
        ("2015", "1R"): compute_precinct_lean(load_mayoral_race(2015, "municipal")),
        ("2015", "RO"): compute_precinct_lean(load_mayoral_race(2015, "municipal_runoff")),
        ("2019", "1R"): compute_precinct_lean(load_mayoral_race(2019, "municipal")),
        ("2019", "RO"): compute_precinct_lean(load_mayoral_race(2019, "municipal_runoff")),
        ("2023", "1R"): compute_precinct_lean(load_mayoral_race(2023, "municipal")),
        ("2023", "RO"): compute_precinct_lean(load_mayoral_race(2023, "municipal_runoff")),
    }

    print("\nLoading 2026 statewide primaries...")
    senate = compute_precinct_lean(
        load_statewide_csv(
            _STATEWIDE_DIR / "Senate_DEM.csv",
            prog_candidates=["Juliana Stratton", "Robin Kelly"],
            mod_candidates=["Raja Krishnamoorthi"],
            race_label="2026 Senate (D)",
        )
    )
    comptroller = compute_precinct_lean(
        load_statewide_csv(
            _STATEWIDE_DIR / "Comptroller_DEM.csv",
            prog_candidates=["Karina Villa"],
            mod_candidates=["Margaret Croke"],
            race_label="2026 Comptroller (D)",
        )
    )

    # ── Generic score: 2019 runoff dropped (Lightfoot/Preckwinkle contamination),
    # 2019 first round retained at full weight (clean 14-candidate bloc structure).
    all_races = [
        mayoral[("2015", "1R")], mayoral[("2015", "RO")],
        mayoral[("2019", "1R")], mayoral[("2019", "RO")],
        mayoral[("2023", "1R")], mayoral[("2023", "RO")],
        senate, comptroller,
    ]
    w_generic = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    print("\nCombining variants...")
    g = combine_leans(all_races, w_generic).rename(
        columns={
            "score": "score_generic",
            "se": "se_generic",
            "n_races_used": "n_races_generic",
            "total_labeled": "n_labeled_generic",
        }
    )

    # ── Black-progressive: 2023 mayoral + 2026 Senate ──────────────────────
    black_races = [mayoral[("2023", "1R")], mayoral[("2023", "RO")], senate]
    blk = combine_leans(black_races, [1.0, 1.0, 1.0]).rename(
        columns={
            "score": "score_black",
            "se": "se_black",
            "n_races_used": "n_races_black",
            "total_labeled": "n_labeled_black",
        }
    )

    # ── Latino-progressive: 2015 mayoral + 2026 Comptroller ────────────────
    latino_races = [mayoral[("2015", "1R")], mayoral[("2015", "RO")], comptroller]
    lat = combine_leans(latino_races, [1.0, 1.0, 1.0]).rename(
        columns={
            "score": "score_latino",
            "se": "se_latino",
            "n_races_used": "n_races_latino",
            "total_labeled": "n_labeled_latino",
        }
    )

    # ── Merge and tidy ─────────────────────────────────────────────────────
    out = (
        g[["joinfield", "score_generic", "se_generic",
           "n_races_generic", "n_labeled_generic"]]
        .merge(blk, on="joinfield", how="outer")
        .merge(lat, on="joinfield", how="outer")
    )

    cols = [
        "joinfield",
        "score_generic", "score_black", "score_latino",
        "se_generic", "se_black", "se_latino",
        "n_races_generic", "n_races_black", "n_races_latino",
        "n_labeled_generic", "n_labeled_black", "n_labeled_latino",
    ]
    out = out[cols].sort_values("joinfield").reset_index(drop=True)

    for c in out.columns:
        if c.startswith("score_") or c.startswith("se_"):
            out[c] = out[c].round(2)
    for c in [c for c in out.columns if c.startswith("n_labeled_") or c.startswith("n_races_")]:
        out[c] = out[c].round(0).astype("Int64")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} precincts to {_OUTPUT_PATH}")

    print("\nScore summaries (pp):")
    for c in ["score_generic", "score_black", "score_latino"]:
        s = out[c].dropna()
        print(
            f"  {c:25s}  mean={s.mean():+5.2f}  std={s.std():5.2f}  "
            f"min={s.min():+6.1f}  max={s.max():+6.1f}  n={len(s)}"
        )


if __name__ == "__main__":
    main()
