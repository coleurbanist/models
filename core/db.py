"""
Read-only access to /home/cole/databases/illinois_elections.db.

Public API
──────────
get_candidate_totals(election_type, year, race_type, district=None)
    → {candidate_name: pct_of_total_vote}

get_precinct_results(election_type, year, race_type, district=None)
    → DataFrame with JoinField + one column per candidate (vote counts)

get_precinct_demographics(joinfields=None, year=2022)
    → DataFrame with JoinField + key demographic columns (race, age, education, income)

get_progressive_scores(race_context, scenario=None, methodology_version=None)
    → DataFrame with JoinField + score_pp + se_pp + diagnostics

All functions raise ValueError if no matching rows are found.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd

DB_PATH = Path("/home/cole/databases/illinois_elections.db")


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_candidate_totals(
    election_type: str,
    year: int,
    race_type: str,
    district: int | float | None = None,
) -> dict[str, float]:
    """
    Returns {candidate_name: pct_of_total_vote} for a race.

    Uses the `candidates` table (district-level totals), not precinct-level
    election_results, so it's fast and doesn't require aggregation.

    Example
    ───────
    get_candidate_totals("primary", 2026, "us_house", 9)
    → {"Daniel Biss": 30.12, "Kat Abughazaleh": 26.94, ...}
    """
    params: list = [election_type, year, race_type]
    district_clause = "AND district IS NULL" if district is None else "AND district = ?"
    if district is not None:
        params.append(district)

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT candidate_name, total_votes
            FROM candidates
            WHERE election_type = ? AND year = ? AND race_type = ?
            {district_clause}
            """,
            params,
        ).fetchall()

    if not rows:
        label = f"{election_type}/{year}/{race_type}" + (f"/{district}" if district is not None else "")
        raise ValueError(f"No candidates found in DB for {label}")

    total = sum(r["total_votes"] for r in rows if r["total_votes"] is not None)
    if total <= 0:
        raise ValueError("Total votes sum to zero — check DB integrity")

    return {
        r["candidate_name"]: round(100.0 * r["total_votes"] / total, 4)
        for r in rows
        if r["total_votes"] is not None
    }


def get_precinct_results(
    election_type: str,
    year: int,
    race_type: str,
    district: int | float | None = None,
) -> pd.DataFrame:
    """
    Returns a wide DataFrame: one row per precinct (JoinField), one column per candidate.
    Values are raw vote counts (float).  Missing = 0.

    Pivots the long `election_results` table.  Suitable for precinct-level calibration
    and for building precinct baselines.
    """
    params: list = [election_type, year, race_type]
    district_clause = "AND district IS NULL" if district is None else "AND district = ?"
    if district is not None:
        params.append(district)

    with _connect() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT JoinField, candidate_name, votes
            FROM election_results
            WHERE election_type = ? AND year = ? AND race_type = ?
            {district_clause}
            """,
            conn,
            params=params,
        )

    if df.empty:
        label = f"{election_type}/{year}/{race_type}" + (f"/{district}" if district is not None else "")
        raise ValueError(f"No precinct results found in DB for {label}")

    wide = df.pivot_table(
        index="JoinField",
        columns="candidate_name",
        values="votes",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    wide.columns.name = None
    return wide


# ── Demographic column groups ───────────────────────────────────────────────
# Curated subsets of the 435-column precinct_demographics table.
# These are the columns actually used by the precinct pipeline.

_DEMO_COLS_RACE = [
    "total",
    "total_hispanic_or_latino",
    "total_not_hispanic_or_latino_white_alone",
    "total_not_hispanic_or_latino_black_or_african_american_alone",
    "total_not_hispanic_or_latino_asian_alone",
]

# Age: sum of male + female buckets already computed in the table
_DEMO_COLS_AGE = [
    "total_male",
    "total_female",
    "total_male_18_and_19_years",
    "total_male_20_years",
    "total_male_21_years",
    "total_male_22_to_24_years",
    "total_male_25_to_29_years",
    "total_male_30_to_34_years",
    "total_male_35_to_39_years",
    "total_male_40_to_44_years",
    "total_male_45_to_49_years",
    "total_male_50_to_54_years",
    "total_male_55_to_59_years",
    "total_male_60_and_61_years",
    "total_male_62_to_64_years",
    "total_male_65_and_66_years",
    "total_male_67_to_69_years",
    "total_male_70_to_74_years",
    "total_male_75_to_79_years",
    "total_male_80_to_84_years",
    "total_male_85_years_and_over",
    "total_female_18_and_19_years",
    "total_female_20_years",
    "total_female_21_years",
    "total_female_22_to_24_years",
    "total_female_25_to_29_years",
    "total_female_30_to_34_years",
    "total_female_35_to_39_years",
    "total_female_40_to_44_years",
    "total_female_45_to_49_years",
    "total_female_50_to_54_years",
    "total_female_55_to_59_years",
    "total_female_60_and_61_years",
    "total_female_62_to_64_years",
    "total_female_65_and_66_years",
    "total_female_67_to_69_years",
    "total_female_70_to_74_years",
    "total_female_75_to_79_years",
    "total_female_80_to_84_years",
    "total_female_85_years_and_over",
]

_DEMO_COLS_EDUCATION = [
    "total.3",                                    # total pop 25+ (education universe)
    "total_no_schooling_completed",
    "total_12th_grade_no_diploma",
    "total_regular_high_school_diploma",
    "total_ged_or_alternative_credential",
    "total_some_college_less_than_1_year",
    "total_some_college_1_or_more_years_no_degree",
    "total_associates_degree",
    "total_bachelors_degree",
    "total_masters_degree",
    "total_professional_school_degree",
    "total_doctorate_degree",
]

# Median income column name varies by ACS year (year embedded in name).
# Include all four; callers can pick the one that matches their year.
_DEMO_COLS_INCOME = [
    "median_household_income_in_the_past_12_months_in_2016_inflat",
    "median_household_income_in_the_past_12_months_in_2018_inflat",
    "median_household_income_in_the_past_12_months_in_2020_inflat",
    "median_household_income_in_the_past_12_months_in_2022_inflat",
    "median_household_income_in_the_past_12_months_in_2024_inflat",
]

DEMO_COLS_DEFAULT: list[str] = (
    _DEMO_COLS_RACE + _DEMO_COLS_AGE + _DEMO_COLS_EDUCATION + _DEMO_COLS_INCOME
)


def get_progressive_scores(
    race_context: str,
    scenario: str | None = None,
    methodology_version: str | None = None,
) -> pd.DataFrame:
    """
    Returns precinct-level progressive lean scores for a race context.

    If `scenario` is None, returns rows for all scenarios (long format).
    If `methodology_version` is None, returns the most recent version per
    (race_context, scenario, JoinField) tuple — handy for grabbing the
    "current" scores without needing to know the version string.

    Columns
    ───────
    JoinField, race_context, scenario, score_pp, se_pp,
    n_races_used, n_labeled_votes, methodology_version, created_at
    """
    base_sql = "SELECT * FROM precinct_progressive_scores WHERE race_context = ?"
    params: list = [race_context]

    if scenario is not None:
        base_sql += " AND scenario = ?"
        params.append(scenario)

    if methodology_version is not None:
        base_sql += " AND methodology_version = ?"
        params.append(methodology_version)
    else:
        # Take the most recent created_at per (scenario, JoinField) tuple
        base_sql = f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY scenario, JoinField
                           ORDER BY created_at DESC
                       ) AS rn
                FROM ({base_sql})
            )
            SELECT * FROM ranked WHERE rn = 1
        """

    with _connect() as conn:
        df = pd.read_sql_query(base_sql, conn, params=params)

    if df.empty:
        label = f"{race_context}" + (f"/{scenario}" if scenario else "")
        raise ValueError(f"No progressive scores found for {label}")

    return df.drop(columns=["rn"], errors="ignore")


def get_precinct_demographics(
    joinfields: Sequence[str] | None = None,
    year: int = 2022,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with JoinField + demographic columns for the given ACS year.

    Parameters
    ──────────
    joinfields : optional list of JoinField values to filter to.  If None, returns all.
    year       : ACS year.  Available: 2016, 2018, 2020, 2022, 2024.
    columns    : list of demographic column names.  Defaults to DEMO_COLS_DEFAULT
                 (race, age, education, income).

    The returned DataFrame always includes JoinField as the first column.
    """
    cols = columns if columns is not None else DEMO_COLS_DEFAULT

    # Validate requested columns exist in the table (avoids silent empty results)
    with _connect() as conn:
        existing = {
            r[1]
            for r in conn.execute("PRAGMA table_info(precinct_demographics)").fetchall()
        }

    missing = [c for c in cols if c not in existing]
    if missing:
        raise ValueError(
            f"Columns not found in precinct_demographics: {missing}\n"
            f"Use DEMO_COLS_DEFAULT or inspect the table directly."
        )

    col_expr = ", ".join(f'"{c}"' for c in cols)
    base_sql = f'SELECT "JoinField", {col_expr} FROM precinct_demographics WHERE year = ?'

    with _connect() as conn:
        if joinfields is None:
            df = pd.read_sql_query(base_sql, conn, params=[year])
        else:
            placeholders = ",".join("?" * len(joinfields))
            df = pd.read_sql_query(
                f'{base_sql} AND "JoinField" IN ({placeholders})',
                conn,
                params=[year, *joinfields],
            )

    if df.empty:
        raise ValueError(f"No demographic data found for year={year}")

    return df
