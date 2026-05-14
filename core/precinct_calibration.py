"""
Race-agnostic logit-space demographic calibration for precinct vote share estimation.

Signals from multiple demographic crosstab dimensions are combined in logit space
so they compete multiplicatively in probability space rather than stacking
additively. This prevents conflicting signals (e.g. a heavily-Black precinct
with strongly conservative ideology) from producing nonsense estimates.

Supported dimension types (auto-detected from demographic_crosstabs keys):

  Race      "black", "hispanic", "white", "asian"
              → requires pct_{group} columns in the precinct DataFrame
              → loaded from precinct_demographics via enrich_precinct_df()

  Ideology  "very_conservative", "somewhat_conservative", "moderate",
            "somewhat_liberal", "very_liberal"
              → requires score_pp column in the precinct DataFrame
              → loaded from precinct_progressive_scores via enrich_precinct_df()
              → a Gaussian kernel on prog_score maps each precinct to a
                distribution over the five ideology bins

Other crosstab dimensions (age, gender, etc.) are ignored until precinct-level
data for those dimensions is available.

Typical usage (from precinct_pipeline.py step1):

    from .precinct_calibration import enrich_precinct_df, compute_precinct_shares

    df = enrich_precinct_df(df, config)
    shares_df = compute_precinct_shares(
        df,
        polling["demographic_crosstabs"],
        polling["baseline"],
        config.candidates,
    )
    for c in config.candidates:
        df[f"demo_est_{c}"] = shares_df[c]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .db import get_precinct_demographics, get_progressive_scores

# ── Dimension classification ───────────────────────────────────────────────

RACE_GROUPS = frozenset({"black", "hispanic", "white", "asian"})

IDEOLOGY_BINS = [
    "very_conservative",
    "somewhat_conservative",
    "moderate",
    "somewhat_liberal",
    "very_liberal",
]
IDEOLOGY_CENTERS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
IDEOLOGY_SIGMA   = 1.0    # spread in ideology units (±1 unit = one full bin)
IDEOLOGY_K       = 25.0   # prog_score pp per ideology unit (±50pp ≈ ±2 units)

# DB column name for each race group
_RACE_DB_COLS: dict[str, str] = {
    "black":    "total_not_hispanic_or_latino_black_or_african_american_alone",
    "hispanic": "total_hispanic_or_latino",
    "white":    "total_not_hispanic_or_latino_white_alone",
    "asian":    "total_not_hispanic_or_latino_asian_alone",
}
_RACE_LOAD_COLS = ["total"] + list(_RACE_DB_COLS.values())


# ── Math helpers ───────────────────────────────────────────────────────────

def _logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _ideology_weights(scores: np.ndarray) -> np.ndarray:
    """Convert prog_score array (pp) to ideology bin weights. Returns (n, 5)."""
    mu = scores[:, None] / IDEOLOGY_K
    w  = np.exp(-0.5 * ((IDEOLOGY_CENTERS - mu) / IDEOLOGY_SIGMA) ** 2)
    return w / w.sum(axis=1, keepdims=True)


# ── Data enrichment ────────────────────────────────────────────────────────

def enrich_precinct_df(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Add pct_black, pct_hispanic, pct_white, pct_asian, and score_pp columns
    to df by querying the DB. Columns already present are left untouched.

    config must have:
        race_context (str | None)      — key for precinct_progressive_scores
        prog_score_scenario (str)      — scenario to use ("generic" by default)
        demo_year (int)                — ACS year for demographics (default 2022)

    The JoinField column is found by looking for "joinfield", "JoinField", etc.
    """
    df = df.copy()

    jf_col = next(
        (c for c in ["joinfield", "JoinField", "JOINFIELD", "join_field"] if c in df.columns),
        None,
    )
    if jf_col is None:
        warnings.warn("No JoinField column found in precinct_df; skipping demographic enrichment")
        return df

    jf_values = df[jf_col].dropna().unique().tolist()
    demo_year  = getattr(config, "demo_year", 2022)

    # ── Race fractions ────────────────────────────────────────────────────
    needs_race = any(f"pct_{g}" not in df.columns for g in _RACE_DB_COLS)
    if needs_race and jf_values:
        try:
            demo = get_precinct_demographics(
                joinfields=jf_values,
                year=demo_year,
                columns=_RACE_LOAD_COLS,
            )
            t = demo["total"].replace(0, np.nan)
            pct_cols: list[str] = []
            for group, db_col in _RACE_DB_COLS.items():
                col = f"pct_{group}"
                if col not in df.columns:
                    demo[col] = demo[db_col] / t
                    pct_cols.append(col)

            demo["_jf_key"] = demo["JoinField"].str.upper()
            df["_jf_key"]   = df[jf_col].str.upper()
            df = df.merge(
                demo[["_jf_key", "total"] + pct_cols].drop_duplicates("_jf_key"),
                on="_jf_key", how="left",
            ).drop(columns=["_jf_key"])

        except Exception as exc:
            warnings.warn(f"Could not load precinct demographics: {exc}")

    # ── Progressive scores (ideology proxy) ──────────────────────────────
    race_context = getattr(config, "race_context", None)
    if "score_pp" not in df.columns and race_context is not None:
        scenario = getattr(config, "prog_score_scenario", "generic")
        try:
            scores = get_progressive_scores(race_context, scenario=scenario)[
                ["JoinField", "score_pp"]
            ].copy()
            scores["_jf_key"] = scores["JoinField"].str.upper()
            df["_jf_key"]     = df[jf_col].str.upper()
            df = df.merge(
                scores[["_jf_key", "score_pp"]].drop_duplicates("_jf_key"),
                on="_jf_key", how="left",
            ).drop(columns=["_jf_key"])
        except Exception as exc:
            warnings.warn(f"Could not load progressive scores for '{race_context}': {exc}")

    return df


# ── Core calibration ───────────────────────────────────────────────────────

def compute_precinct_shares(
    df: pd.DataFrame,
    demographic_crosstabs: dict[str, dict[str, float]],
    baseline: dict[str, float],
    candidates: list[str],
    weight_col: str = "turnout_weight",
) -> pd.DataFrame:
    """
    Estimate per-precinct vote shares by combining demographic crosstab signals
    in logit space.

    demographic_crosstabs: {group_name: {candidate: fraction}}  (0–1 scale)
        as returned by poll_weighting.aggregate_polls().
    baseline: {candidate: fraction}  (0–1 scale), the weighted poll topline.

    Returns a DataFrame indexed like df with:
      - one column per candidate (predicted pct, 0–100)
      - "predicted_winner" column

    Dimensions used:
      - Race: groups in RACE_GROUPS that are both in demographic_crosstabs
        and have a matching pct_{group} column in df.
      - Ideology: any IDEOLOGY_BINS keys present in demographic_crosstabs,
        if df has a score_pp column.

    Unrecognized groups are silently ignored. If no usable data is found,
    falls back to the district baseline for every precinct.
    """
    n = len(df)

    avail_race = [
        g for g in RACE_GROUPS
        if g in demographic_crosstabs and f"pct_{g}" in df.columns
    ]
    avail_ideo = [g for g in IDEOLOGY_BINS if g in demographic_crosstabs]
    has_ideology = bool(avail_ideo) and "score_pp" in df.columns

    if not avail_race and not has_ideology:
        out = pd.DataFrame(
            {c: baseline.get(c, 1.0 / len(candidates)) * 100.0 for c in candidates},
            index=df.index,
        )
        winner = max(baseline, key=baseline.get) if baseline else candidates[0]
        out["predicted_winner"] = winner
        return out

    # Ideology bin weights (n, 5) from prog_score
    ideo_w = _ideology_weights(df["score_pp"].fillna(0.0).values) if has_ideology else None

    # Fraction of precinct population covered by race groups with known crosstabs
    if avail_race:
        pct_covered = sum(df[f"pct_{g}"].fillna(0.0).values for g in avail_race)
        pct_other   = (1.0 - pct_covered).clip(0)

    # Turnout weights for calibration
    if weight_col in df.columns:
        w = df[weight_col].fillna(1.0).values
    elif "total" in df.columns:
        w = df["total"].fillna(1.0).values
    else:
        w = np.ones(n)

    out_data: dict[str, np.ndarray] = {}
    for cand in candidates:
        tl_f     = baseline.get(cand, 1.0 / len(candidates))
        tl_logit = float(_logit(np.array([tl_f]))[0])
        delta    = np.zeros(n)

        # Race: weighted avg of per-group crosstabs → logit delta
        if avail_race:
            race_f = pct_other * tl_f
            for g in avail_race:
                ct     = demographic_crosstabs[g].get(cand, tl_f)
                race_f = race_f + df[f"pct_{g}"].fillna(0.0).values * ct
            delta += _logit(race_f) - tl_logit

        # Ideology: Gaussian-weighted avg over ideology bins → logit delta
        if has_ideology and ideo_w is not None:
            ideo_cts = np.array([
                demographic_crosstabs.get(g, {}).get(cand, tl_f)
                for g in IDEOLOGY_BINS
            ])  # missing bins fall back to topline → contribute zero delta
            ideo_f = ideo_w @ ideo_cts
            delta += _logit(ideo_f) - tl_logit

        combined_pct = _expit(tl_logit + delta) * 100.0

        # Small additive calibration so turnout-weighted mean == topline
        mean_pct = (combined_pct * w).sum() / w.sum()
        out_data[cand] = np.clip(combined_pct + (tl_f * 100.0 - mean_pct), 0.0, None)

    out = pd.DataFrame(out_data, index=df.index)

    # Renormalize rows to sum to 100
    row_sum = out[candidates].sum(axis=1).replace(0, np.nan)
    for cand in candidates:
        out[cand] = out[cand] / row_sum * 100.0

    out["predicted_winner"] = out[candidates].idxmax(axis=1)
    return out


# ── Convenience: check whether calibration will do anything ───────────────

def has_usable_crosstabs(
    demographic_crosstabs: dict,
    df: pd.DataFrame,
) -> bool:
    """Return True if compute_precinct_shares will use at least one dimension."""
    race_ok = any(
        g in demographic_crosstabs and f"pct_{g}" in df.columns
        for g in RACE_GROUPS
    )
    ideo_ok = (
        any(g in demographic_crosstabs for g in IDEOLOGY_BINS)
        and "score_pp" in df.columns
    )
    return race_ok or ideo_ok
