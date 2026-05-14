"""
Build per-precinct turnout statistics for Chicago mayoral races.

Reference elections: 2015, 2019, 2023 (both first round and runoff).

For each precinct × round type (first_round / runoff) we compute:
  - mean_turnout    : average total votes cast across the three cycles
  - std_turnout     : sample standard deviation across cycles
  - cv_turnout      : coefficient of variation (std / mean), a normalized
                      fluctuation metric that is comparable across precincts
                      of different size

Output CSV: outputs/precinct_turnout_stats.csv
  JoinField, mean_turnout_1r, std_turnout_1r, cv_turnout_1r,
             mean_turnout_ro, std_turnout_ro, cv_turnout_ro,
             n_cycles_1r, n_cycles_ro
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = Path("/home/cole/databases/illinois_elections.db")
SCRIPT_DIR = Path(__file__).parent
OUT_CSV = SCRIPT_DIR / "outputs" / "precinct_turnout_stats.csv"

YEARS = [2015, 2019, 2023]


def load_precinct_turnout(election_type: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (JoinField, year): total votes cast.
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            """
            SELECT JoinField, year, SUM(votes) AS turnout
            FROM election_results
            WHERE race_type = 'chicago_mayor'
              AND election_type = ?
              AND year IN (2015, 2019, 2023)
            GROUP BY JoinField, year
            """,
            conn,
            params=[election_type],
        )
    return df


def summarize(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    From a long (JoinField, year, turnout) DataFrame, produce per-precinct
    mean / std / CV across years.
    """
    wide = df.pivot_table(index="JoinField", columns="year", values="turnout")
    wide.columns = [str(c) for c in wide.columns]

    mean_ = wide.mean(axis=1)
    std_ = wide.std(axis=1, ddof=1)      # sample std
    cv_ = std_ / mean_
    n_ = wide.notna().sum(axis=1)

    out = pd.DataFrame({
        "JoinField": mean_.index,
        f"mean_turnout_{suffix}": mean_.values,
        f"std_turnout_{suffix}": std_.values,
        f"cv_turnout_{suffix}": cv_.values,
        f"n_cycles_{suffix}": n_.values,
    })
    return out


def main() -> None:
    print("Loading first-round turnout (municipal)...")
    df_1r = load_precinct_turnout("municipal")
    print(f"  {len(df_1r)} precinct-year rows")

    print("Loading runoff turnout (municipal_runoff)...")
    df_ro = load_precinct_turnout("municipal_runoff")
    print(f"  {len(df_ro)} precinct-year rows")

    stats_1r = summarize(df_1r, "1r")
    stats_ro = summarize(df_ro, "ro")

    combined = stats_1r.merge(stats_ro, on="JoinField", how="outer")
    combined = combined.sort_values("JoinField").reset_index(drop=True)

    print(f"\nPrecincts in output: {len(combined)}")
    print("\nSample (sorted by mean first-round turnout, descending):")
    print(
        combined.sort_values("mean_turnout_1r", ascending=False)
        .head(10)
        .to_string(index=False, float_format=lambda x: f"{x:.1f}")
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
