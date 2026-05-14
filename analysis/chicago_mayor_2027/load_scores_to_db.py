"""
Load Chicago mayoral progressive scores into illinois_elections.db.

Reads:  outputs/precinct_progressive_scores.csv (must be rebuilt first
        via build_progressive_scores.py if scoring methodology changes)
Writes: precinct_progressive_scores table

race_context = 'chicago_mayor'
scenarios    = ['generic', 'black', 'latino']

The CSV's joinfields are uppercased; we look up the canonical mixed-case
form from election_results so this table's JoinField column joins
naturally against the rest of the DB.

methodology_version
───────────────────
Defaults to today's date + '-v1'. Bump suffix (-v2, -v3, ...) when you
re-run with the same date but a changed methodology — that way history
is preserved instead of overwritten.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Make `core.*` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.precinct_score_db import (  # noqa: E402
    load_canonical_chicago_joinfields,
    upsert_scores,
)

SCRIPT_DIR = Path(__file__).parent
SCORES_CSV = SCRIPT_DIR / "outputs" / "precinct_progressive_scores.csv"
RACE_CONTEXT = "chicago_mayor"
METHODOLOGY_VERSION = f"{date.today().isoformat()}-v1"

SCENARIOS = [
    # (db scenario name, score column, se column, n_races column, n_labeled column)
    ("generic", "score_generic", "se_generic", "n_races_generic", "n_labeled_generic"),
    ("black",   "score_black",   "se_black",   "n_races_black",   "n_labeled_black"),
    ("latino",  "score_latino",  "se_latino",  "n_races_latino",  "n_labeled_latino"),
]


def main() -> None:
    print(f"Loading scores from {SCORES_CSV}")
    df = pd.read_csv(SCORES_CSV)
    print(f"  {len(df)} precincts")

    print("Resolving JoinField casing against election_results...")
    canonical = load_canonical_chicago_joinfields()
    df["JoinField"] = df["joinfield"].map(canonical)
    unmatched = df["JoinField"].isna().sum()
    if unmatched:
        print(f"  ⚠ {unmatched} joinfields not found in election_results; falling back to raw uppercase")
        df["JoinField"] = df["JoinField"].fillna(df["joinfield"])
    else:
        print(f"  All {len(df)} joinfields matched canonical DB form")

    print(f"\nWriting to DB with methodology_version='{METHODOLOGY_VERSION}'")
    total = 0
    for scenario, score_col, se_col, n_races_col, n_labeled_col in SCENARIOS:
        rows = pd.DataFrame({
            "JoinField":       df["JoinField"],
            "score_pp":        df[score_col],
            "se_pp":           df[se_col],
            "n_races_used":    df[n_races_col],
            "n_labeled_votes": df[n_labeled_col],
        })
        n = upsert_scores(
            race_context=RACE_CONTEXT,
            scenario=scenario,
            rows=rows,
            methodology_version=METHODOLOGY_VERSION,
        )
        print(f"  scenario={scenario!r:10s}  rows written: {n}")
        total += n

    print(f"\nDone — {total} rows total across {len(SCENARIOS)} scenarios.")


if __name__ == "__main__":
    main()
