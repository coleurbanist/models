"""
IL-09 2026 Democratic Primary — post-mortem analysis.

Compares the final pre-election predictions (March 14, 2026) against actual
certified results from the DB, across five dimensions:

  1. Overall error summary (by candidate, by pipeline stage)
  2. Undecided break pattern (did top-tier candidates absorb more than modeled?)
  3. Geographic loyalty (per-candidate error by jurisdiction / township)
  4. Community bloc identification (precincts with extreme overperformance)
  5. Ideological bloc correlation (did bloc-mates' errors actually correlate?)

Outputs
───────
  - Printed report to stdout
  - analysis/il09_2026/calibration_params.json   — suggested constraint values
  - analysis/il09_2026/precinct_errors.csv        — full precinct-level error table

Usage
─────
  python -m analysis.il09_2026.postmortem
  python -m analysis.il09_2026.postmortem --quiet   # suppress per-precinct tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
_ROOT       = _HERE.parent.parent
_PRED_FILE  = Path("/home/cole/elections/il9prediction_and_tracker/data/csv_data/expectations/IL_09_precinct_probabilities_old_2026_03_14_21_38.csv")
_DB_PATH    = Path("/home/cole/databases/illinois_elections.db")
_OUT_PARAMS = _HERE / "calibration_params.json"
_OUT_CSV    = _HERE / "precinct_errors.csv"

# ── Race constants ─────────────────────────────────────────────────────────
CANDIDATES = ["Fine", "Biss", "Abughazaleh", "Simmons", "Amiwala", "Andrew", "Huynh"]
TOP_TIER   = {"Fine", "Biss", "Abughazaleh"}
LOWER_TIER = {"Simmons", "Amiwala", "Andrew", "Huynh"}

DB_NAME_MAP = {
    "Laura Fine":      "Fine",
    "Daniel Biss":     "Biss",
    "Kat Abughazaleh": "Abughazaleh",
    "Mike Simmons":    "Simmons",
    "Bushra Amiwala":  "Amiwala",
    "Phil Andrew":     "Andrew",
    "Hoan Huynh":      "Huynh",
}

# Geographic boosts that were applied in the original model.
# Used to measure whether they were correctly sized.
GEO_BOOST_COLS = [
    "geo_boost_Biss_evanston",
    "geo_boost_Abughazaleh_chicago",
    "geo_boost_Simmons_chicago",
    "geo_boost_Huynh_chicago",
    "geo_boost_Fine_outside_chicago",
    "geo_boost_Amiwala_niles",
]

# Community bloc threshold: if a candidate's actual share exceeds this multiple
# of their district-wide actual share, flag the precinct as a community bloc.
COMMUNITY_BLOC_MULTIPLIER = 2.0


# ── Data loading ───────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(str(_PRED_FILE), low_memory=False)
    df["jf_key"] = df["JoinField"].str.upper()
    for c in CANDIDATES:
        df = df.rename(columns={f"final_{c}": f"pred_{c}"})
    return df


def load_actuals() -> pd.DataFrame:
    conn = sqlite3.connect(str(_DB_PATH))
    long = pd.read_sql_query(
        """
        SELECT JoinField, candidate_name, votes
        FROM election_results
        WHERE election_type='primary' AND year=2026
          AND race_type='us_house' AND district=9
        """,
        conn,
    )
    conn.close()

    long = long[long["candidate_name"].isin(DB_NAME_MAP)].copy()
    long["cand"] = long["candidate_name"].map(DB_NAME_MAP)
    wide = long.pivot_table(
        index="JoinField", columns="cand", values="votes",
        aggfunc="sum", fill_value=0,
    ).reset_index()
    wide.columns.name = None

    for c in CANDIDATES:
        if c not in wide.columns:
            wide[c] = 0.0

    tot = wide[CANDIDATES].sum(axis=1)
    wide["total_votes"] = tot
    for c in CANDIDATES:
        wide[f"actual_{c}"] = wide[c] / tot * 100

    wide["jf_key"] = wide["JoinField"].str.upper()
    return wide


def merge_data(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    pred_cols = (
        ["jf_key", "jurisdiction", "township", "ward_preci",
         "in_chicago", "in_evanston", "estimated_turnout"]
        + [f"pred_{c}" for c in CANDIDATES]
        + [f"exp_{c}"  for c in CANDIDATES]
        + [f"raw_{c}"  for c in CANDIDATES]
        + GEO_BOOST_COLS
    )
    # Only keep columns that actually exist in the file
    pred_cols = [c for c in pred_cols if c in pred.columns]

    df = pred[pred_cols].merge(
        actual[["jf_key", "JoinField", "total_votes"]
               + [f"actual_{c}" for c in CANDIDATES]
               + CANDIDATES],
        on="jf_key", how="inner",
    )

    for c in CANDIDATES:
        df[f"err_{c}"] = df[f"pred_{c}"] - df[f"actual_{c}"]

    return df


# ── Analysis functions ─────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")


def overall_errors(df: pd.DataFrame) -> None:
    section("1. OVERALL ERROR SUMMARY  (predicted − actual, in pp)")
    hdr = f"  {'Candidate':<16} {'Mean':>8} {'MAE':>8} {'RMSE':>8} {'Max+':>9} {'Max−':>9}"
    print(hdr)
    print("  " + "─" * 60)
    for c in CANDIDATES:
        e = df[f"err_{c}"]
        tier = "top" if c in TOP_TIER else "low"
        print(
            f"  {c:<16} {e.mean():>+7.2f}pp {e.abs().mean():>7.2f}pp "
            f"{(e**2).mean()**0.5:>7.2f}pp "
            f"{e.max():>+8.2f}pp {e.min():>+8.2f}pp  [{tier}]"
        )


def pipeline_stage_errors(df: pd.DataFrame) -> None:
    """Compare raw → exp → pred errors to see which stage added or removed bias."""
    section("2. ERROR BY PIPELINE STAGE  (district-weighted mean error, pp)")

    # Only show if exp_ columns are present
    if "exp_Fine" not in df.columns:
        print("  (exp_ columns not available — skipping)")
        return

    stages = []
    if all(f"raw_{c}" in df.columns for c in CANDIDATES):
        stages.append(("Demographic model (raw)", "raw"))
    stages.append(("Poll baseline (exp)", "exp"))
    stages.append(("Final prediction (pred)", "pred"))

    tot = df["total_votes"]

    print(f"  {'Stage':<28}", end="")
    for c in CANDIDATES:
        print(f" {c:>10}", end="")
    print()
    print("  " + "─" * (28 + len(CANDIDATES) * 11))

    for label, prefix in stages:
        print(f"  {label:<28}", end="")
        for c in CANDIDATES:
            col = f"{prefix}_{c}"
            if col not in df.columns:
                print(f" {'N/A':>10}", end="")
                continue
            err = df[col] - df[f"actual_{c}"]
            weighted = (err * tot).sum() / tot.sum()
            print(f" {weighted:>+9.2f}pp", end="")
        print()


def undecided_break(df: pd.DataFrame) -> dict:
    """
    Measures how undecideds actually broke vs how the model distributed them.

    The model distributed undecideds proportionally (with small weight tweaks).
    Actual results show top-tier candidates absorbed nearly all of them.

    Returns calibration params for undecided_skew.
    """
    section("3. UNDECIDED BREAK ANALYSIS")

    # District-wide actual shares
    tot_votes = df[CANDIDATES].sum().sum()
    actual_district = {c: df[c].sum() / tot_votes * 100 for c in CANDIDATES}

    # District-wide predicted shares (turnout-weighted)
    pred_district = {}
    for c in CANDIDATES:
        pred_district[c] = (df[f"pred_{c}"] * df["total_votes"]).sum() / df["total_votes"].sum()

    # Poll baseline = exp_ district average (what the polls said before undecided allocation)
    poll_baseline = {}
    if "exp_Fine" in df.columns:
        for c in CANDIDATES:
            poll_baseline[c] = (df[f"exp_{c}"] * df["total_votes"]).sum() / df["total_votes"].sum()
    else:
        # Fallback: infer poll baseline from final predictions (less precise)
        poll_baseline = pred_district.copy()

    # Undecided pool = gap between poll totals and 100%
    polled_total = sum(poll_baseline.values())
    undecided_pool = 100.0 - polled_total

    print(f"\n  Poll totals sum to {polled_total:.1f}%  →  undecided pool ≈ {undecided_pool:.1f}pp")
    print()
    print(f"  {'Candidate':<16} {'Poll base':>10} {'Predicted':>10} {'Actual':>10} "
          f"{'Δ pred−poll':>12} {'Δ actual−poll':>14} {'Tier':>6}")
    print("  " + "─" * 80)

    top_actual_gain = 0.0
    low_actual_gain = 0.0
    top_pred_gain   = 0.0
    low_pred_gain   = 0.0

    for c in CANDIDATES:
        pb  = poll_baseline[c]
        prd = pred_district[c]
        act = actual_district[c]
        tier = "top" if c in TOP_TIER else "low"
        print(
            f"  {c:<16} {pb:>9.2f}% {prd:>9.2f}% {act:>9.2f}%"
            f" {prd-pb:>+11.2f}pp {act-pb:>+13.2f}pp {tier:>6}"
        )
        if c in TOP_TIER:
            top_actual_gain += act - pb
            top_pred_gain   += prd - pb
        else:
            low_actual_gain += act - pb
            low_pred_gain   += prd - pb

    print()
    print(f"  Top-tier absorbed:  predicted {top_pred_gain:+.1f}pp of undecideds, "
          f"actual {top_actual_gain:+.1f}pp")
    print(f"  Lower-tier absorbed: predicted {low_pred_gain:+.1f}pp of undecideds, "
          f"actual {low_actual_gain:+.1f}pp")

    if undecided_pool > 0:
        top_actual_frac = top_actual_gain / undecided_pool
        low_actual_frac = low_actual_gain / undecided_pool
        top_pred_frac   = top_pred_gain   / undecided_pool
        print()
        print(f"  Top-tier fraction of undecided pool:  predicted {top_pred_frac:.1%}, "
              f"actual {top_actual_frac:.1%}")
        print(f"  Lower-tier fraction:                  predicted {1-top_pred_frac:.1%}, "
              f"actual {1-top_actual_frac:.1%}")
        print()
        print("  CALIBRATION FINDING: Model spread undecideds too evenly across tiers.")
        print(f"  Top-tier should receive ~{top_actual_frac:.0%} of undecideds, "
              f"not ~{top_pred_frac:.0%}.")

    return {
        "undecided_pool_pct": round(undecided_pool, 2),
        "top_tier_actual_fraction": round(top_actual_frac, 3) if undecided_pool > 0 else None,
        "top_tier_predicted_fraction": round(top_pred_frac, 3) if undecided_pool > 0 else None,
        "actual_gain_by_candidate": {
            c: round(actual_district[c] - poll_baseline[c], 2) for c in CANDIDATES
        },
    }


def geographic_errors(df: pd.DataFrame) -> dict:
    """Per-candidate mean error by jurisdiction and township."""
    section("4. GEOGRAPHIC ERROR BREAKDOWN  (mean error pred − actual, pp)")

    calibration = {}

    # By jurisdiction
    print("\n  BY JURISDICTION:")
    print(f"  {'Jurisdiction':<24}", end="")
    for c in CANDIDATES:
        print(f" {c:>10}", end="")
    print("   n")
    print("  " + "─" * (24 + len(CANDIDATES)*11 + 5))

    jur_params = {}
    for jur, grp in df.groupby("jurisdiction"):
        print(f"  {jur:<24}", end="")
        jur_params[jur] = {}
        for c in CANDIDATES:
            err = (grp[f"pred_{c}"] - grp[f"actual_{c}"]).mean()
            jur_params[jur][c] = round(err, 2)
            print(f" {err:>+9.1f}pp", end="")
        print(f"  ({len(grp)})")
    calibration["mean_error_by_jurisdiction"] = jur_params

    # By township (Cook County only, where township data exists)
    print("\n  BY TOWNSHIP (Cook County):")
    cook = df[df["jurisdiction"] == "COOK"].copy()
    if "township" in cook.columns and cook["township"].notna().any():
        print(f"  {'Township':<20}", end="")
        for c in CANDIDATES:
            print(f" {c:>10}", end="")
        print("   n")
        print("  " + "─" * (20 + len(CANDIDATES)*11 + 5))

        twp_params = {}
        for twp, grp in cook.groupby("township"):
            print(f"  {str(twp):<20}", end="")
            twp_params[str(twp)] = {}
            for c in CANDIDATES:
                err = (grp[f"pred_{c}"] - grp[f"actual_{c}"]).mean()
                twp_params[str(twp)][c] = round(err, 2)
                print(f" {err:>+9.1f}pp", end="")
            print(f"  ({len(grp)})")
        calibration["mean_error_by_township"] = twp_params

    # Geo boost effectiveness
    if any(c in df.columns for c in GEO_BOOST_COLS):
        print("\n  GEO BOOST EFFECTIVENESS (was the boost the right size?):")
        boost_map = {
            "geo_boost_Biss_evanston":         ("Biss",         "in_evanston"),
            "geo_boost_Fine_outside_chicago":  ("Fine",         None),
            "geo_boost_Abughazaleh_chicago":   ("Abughazaleh",  "in_chicago"),
            "geo_boost_Simmons_chicago":       ("Simmons",      "in_chicago"),
            "geo_boost_Amiwala_niles":         ("Amiwala",      None),
        }
        for boost_col, (cand, region_col) in boost_map.items():
            if boost_col not in df.columns:
                continue
            boosted = df[df[boost_col] > 1.001]
            if boosted.empty:
                continue
            err = (boosted[f"pred_{cand}"] - boosted[f"actual_{cand}"]).mean()
            print(f"  {boost_col:<40}  n={len(boosted):>3}  "
                  f"mean err in boosted precincts: {err:>+.2f}pp  ", end="")
            if err > 2:
                print("(boost TOO LARGE)")
            elif err < -2:
                print("(boost TOO SMALL — needed more)")
            else:
                print("(approximately correct)")

    return calibration


def community_blocs(df: pd.DataFrame) -> dict:
    """
    Identifies precincts where a candidate's actual share was >= COMMUNITY_BLOC_MULTIPLIER
    times their district-wide actual share — indicating a concentrated community bloc vote.
    """
    section("5. COMMUNITY BLOC IDENTIFICATION")
    print(f"  (flagging precincts where actual share ≥ {COMMUNITY_BLOC_MULTIPLIER}× district average)\n")

    tot_votes = df[CANDIDATES].sum().sum()
    district_avg = {c: df[c].sum() / tot_votes * 100 for c in CANDIDATES}

    bloc_precincts: dict[str, list] = {}

    for c in CANDIDATES:
        threshold = district_avg[c] * COMMUNITY_BLOC_MULTIPLIER
        blocs = df[df[f"actual_{c}"] >= threshold].copy()
        blocs = blocs.sort_values(f"actual_{c}", ascending=False)

        if blocs.empty:
            continue

        bloc_precincts[c] = []
        print(f"  {c} (district avg {district_avg[c]:.1f}%, threshold {threshold:.1f}%):")
        print(f"  {'Precinct':<46} {'Actual':>8} {'Predicted':>10} {'Error':>8} {'Votes':>7}")
        print("  " + "─" * 82)

        for _, row in blocs.head(20).iterrows():
            jf = row["JoinField"]
            act = row[f"actual_{c}"]
            prd = row[f"pred_{c}"]
            err = prd - act
            votes = int(row["total_votes"]) if "total_votes" in row else 0
            print(f"  {jf:<46} {act:>7.1f}% {prd:>9.1f}% {err:>+7.1f}pp {votes:>7}")
            bloc_precincts[c].append({
                "JoinField": jf,
                "actual_pct": round(act, 2),
                "predicted_pct": round(prd, 2),
                "error_pp": round(err, 2),
                "total_votes": votes,
            })

        if len(blocs) > 20:
            print(f"  ... and {len(blocs)-20} more")
        print()

    return {"district_avg_pct": {c: round(district_avg[c], 2) for c in CANDIDATES},
            "community_bloc_precincts": bloc_precincts}


def bloc_correlation(df: pd.DataFrame) -> dict:
    """
    Checks whether candidates in the same ideological bloc had correlated errors.
    If Fine/Biss errors correlate strongly, the bloc structure was right.
    If Abughazaleh/Simmons errors don't correlate, the bloc structure was wrong.
    """
    section("6. IDEOLOGICAL BLOC ERROR CORRELATION")
    print("  (positive = errors move together; if bloc-mates are correlated,")
    print("   the bloc shock model was capturing real co-movement)\n")

    pairs = [
        ("Fine",        "Biss",         "moderate/establishment bloc"),
        ("Abughazaleh", "Simmons",      "progressive bloc"),
        ("Fine",        "Abughazaleh",  "cross-bloc (should be lower)"),
        ("Biss",        "Abughazaleh",  "cross-bloc Biss vs progressive"),
        ("Amiwala",     "Andrew",       "lower-tier cross"),
    ]

    corr_results = {}
    for c1, c2, label in pairs:
        r = df[f"err_{c1}"].corr(df[f"err_{c2}"])
        corr_results[f"{c1}_vs_{c2}"] = round(r, 3)
        strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        direction = "positive" if r > 0 else "negative"
        print(f"  {c1:<16} ↔ {c2:<16} r={r:>+.3f}  {strength} {direction}  [{label}]")

    print()
    print("  NOTE: Biss straddles moderate/progressive — his errors may correlate")
    print("  with BOTH blocs, which the two-bloc structure doesn't capture.")

    return corr_results


def suggested_constraints(
    undecided_params: dict,
    geo_params: dict,
    bloc_params: dict,
) -> dict:
    """
    Synthesizes findings into suggested extra_constraints and undecided_allocation
    values for a re-run of IL-09 or a similar race.
    """
    section("7. SUGGESTED CALIBRATION PARAMETERS")

    # Undecided skew: top-tier should absorb more
    top_frac = undecided_params.get("top_tier_actual_fraction", 0.85)
    gains    = undecided_params.get("actual_gain_by_candidate", {})

    print("\n  UNDECIDED ALLOCATION WEIGHTS (relative, before normalization):")
    print("  Scale top-tier weights up and lower-tier weights down to match")
    print(f"  observed pattern where top-tier absorbed ~{top_frac:.0%} of undecideds.\n")

    # Compute suggested weights: proportional to actual gain above poll baseline
    # Floor lower-tier at 0.3 to avoid zeroing them out entirely
    total_gain = sum(max(v, 0) for v in gains.values())
    suggested_weights = {}
    for c in CANDIDATES:
        gain = gains.get(c, 0)
        if gain > 0:
            # Weight proportional to how much they actually absorbed
            suggested_weights[c] = round(max(gain / total_gain * len(CANDIDATES), 0.3), 3)
        else:
            suggested_weights[c] = 0.3
    print("  undecided_allocation = {")
    for c, w in suggested_weights.items():
        print(f"    \"{c}\": {w},")
    print("  }")

    # Geographic constraints
    print("\n  GEOGRAPHIC CONSTRAINTS (extra_constraints additions):")

    # Fine community bloc precincts — Ward 50 + specific Niles precincts
    fine_blocs = bloc_params.get("community_bloc_precincts", {}).get("Fine", [])
    fine_bloc_jfs = [b["JoinField"] for b in fine_blocs if b["actual_pct"] > 45]
    if fine_bloc_jfs:
        ward50 = [jf for jf in fine_bloc_jfs if "WARD 50" in jf.upper() or "Ward 50" in jf]
        niles  = [jf for jf in fine_bloc_jfs if "COOK:" in jf]
        print(f"\n  Fine community bloc precincts ({len(fine_bloc_jfs)} total):")
        print(f"    Ward 50 precincts: {len(ward50)}")
        print(f"    Niles Township precincts: {len(niles)}")
        print("    → Add as fine_orthodox_precincts list in extra_constraints")
        print("    → Apply ~3x baseline boost in these precincts")

    # Amiwala Niles community bloc
    amiwala_blocs = bloc_params.get("community_bloc_precincts", {}).get("Amiwala", [])
    amiwala_niles = [b for b in amiwala_blocs if "COOK:" in b["JoinField"]]
    if amiwala_niles:
        print(f"\n  Amiwala community bloc precincts ({len(amiwala_niles)} Niles precincts):")
        print("    → Add as amiwala_south_asian_precincts list in extra_constraints")
        print("    → Apply ~2x baseline boost in these precincts")

    # Biss Lake County
    lake_biss_err = (
        geo_params.get("mean_error_by_jurisdiction", {})
        .get("LAKE", {})
        .get("Biss", 0)
    )
    if lake_biss_err < -4:
        print(f"\n  Biss underestimated in Lake County by {lake_biss_err:.1f}pp on average.")
        print("    → geo_boost_Biss_lakecounty should be added (~1.15–1.20×)")

    constraints = {
        "undecided_allocation": suggested_weights,
        "fine_community_bloc_precincts": fine_bloc_jfs,
        "amiwala_community_bloc_precincts": [b["JoinField"] for b in amiwala_niles],
        "notes": {
            "undecided_skew": (
                f"Top-tier absorbed {top_frac:.0%} of undecideds vs "
                f"{undecided_params.get('top_tier_predicted_fraction', 0):.0%} predicted"
            ),
            "fine_community_bloc": (
                "Ward 50 + Niles Township Orthodox Jewish precincts. "
                "Fine got 55-80% actual vs ~18% predicted. "
                "Use actual 2026 results to identify; ACS has no religion field."
            ),
            "niles_complexity": (
                "Niles Township has overlapping community signals — Fine (Orthodox Jewish) "
                "and Amiwala (South Asian) both overperformed in DIFFERENT precincts. "
                "Township-level boost is wrong; must constrain at precinct level."
            ),
            "biss_bloc": (
                "Biss straddles moderate/progressive — his errors correlate with both blocs. "
                "Grouping him with Fine as 'moderate' understates his progressive support."
            ),
        },
    }

    return constraints


# ── Main ───────────────────────────────────────────────────────────────────

def main(quiet: bool = False) -> None:
    print("Loading data...")
    pred   = load_predictions()
    actual = load_actuals()
    df     = merge_data(pred, actual)
    print(f"Matched {len(df)} precincts.\n")

    overall_errors(df)
    pipeline_stage_errors(df)
    undecided_params = undecided_break(df)
    geo_params       = geographic_errors(df)
    bloc_params      = community_blocs(df)
    corr_params      = bloc_correlation(df)
    constraints      = suggested_constraints(undecided_params, geo_params, bloc_params)

    # ── Save outputs ────────────────────────────────────────────────────────
    calibration_output = {
        "race":          "il09_2026",
        "prediction_as_of": "2026-03-14",
        "undecided":     undecided_params,
        "geography":     geo_params,
        "community_blocs": bloc_params,
        "bloc_correlations": corr_params,
        "suggested_constraints": constraints,
    }

    with _OUT_PARAMS.open("w", encoding="utf-8") as f:
        json.dump(calibration_output, f, indent=2)
    print(f"\n  Calibration params saved → {_OUT_PARAMS}")

    # Precinct error CSV
    err_cols = (
        ["JoinField", "jurisdiction", "township", "total_votes"]
        + [f"pred_{c}"   for c in CANDIDATES]
        + [f"actual_{c}" for c in CANDIDATES]
        + [f"err_{c}"    for c in CANDIDATES]
    )
    err_cols = [c for c in err_cols if c in df.columns]
    df[err_cols].to_csv(str(_OUT_CSV), index=False)
    print(f"  Precinct error table saved → {_OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IL-09 2026 post-mortem analysis")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose tables")
    args = parser.parse_args()
    main(quiet=args.quiet)
