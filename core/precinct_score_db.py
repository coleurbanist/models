"""
Write utility for precinct_progressive_scores in illinois_elections.db.

Designed to be reused across race contexts (Chicago mayor, alderperson,
US House, state legislature, etc.). Each (race_context, scenario,
JoinField, methodology_version) tuple is a unique row, so re-running a
loader with the same methodology_version overwrites in place, while
bumping methodology_version preserves history.

Schema
──────
precinct_progressive_scores (
    score_id            INTEGER PK
    race_context        TEXT   -- 'chicago_mayor', 'chicago_alderperson', ...
    scenario            TEXT   -- 'generic', 'black', 'latino', ...
    JoinField           TEXT   -- matches election_results.JoinField (mixed case)
    score_pp            REAL   -- progressive lean vs. citywide, in pp
    se_pp               REAL   -- standard error of the score, in pp
    n_races_used        INTEGER
    n_labeled_votes     INTEGER
    methodology_version TEXT
    created_at          TIMESTAMP
    UNIQUE(race_context, scenario, JoinField, methodology_version)
)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("/home/cole/databases/illinois_elections.db")

_DDL = """
CREATE TABLE IF NOT EXISTS precinct_progressive_scores (
    score_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    race_context        TEXT    NOT NULL,
    scenario            TEXT    NOT NULL,
    JoinField           TEXT    NOT NULL,
    score_pp            REAL    NOT NULL,
    se_pp               REAL    NOT NULL,
    n_races_used        INTEGER,
    n_labeled_votes     INTEGER,
    methodology_version TEXT,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (race_context, scenario, JoinField, methodology_version)
);
CREATE INDEX IF NOT EXISTS idx_pps_joinfield ON precinct_progressive_scores(JoinField);
CREATE INDEX IF NOT EXISTS idx_pps_context   ON precinct_progressive_scores(race_context, scenario);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)


def upsert_scores(
    race_context: str,
    scenario: str,
    rows: pd.DataFrame,
    methodology_version: str,
) -> int:
    """
    Insert or update precinct scores for one (race_context, scenario) bundle.

    rows must have columns: JoinField, score_pp, se_pp.
    Optional columns: n_races_used, n_labeled_votes.

    Returns the number of rows written.
    """
    required = {"JoinField", "score_pp", "se_pp"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"upsert_scores: rows missing required columns: {missing}")

    df = rows.copy()
    for c in ("n_races_used", "n_labeled_votes"):
        if c not in df.columns:
            df[c] = None

    df = df.dropna(subset=["JoinField", "score_pp", "se_pp"])
    if df.empty:
        return 0

    def _to_int_or_none(v):
        return int(v) if pd.notna(v) else None

    params = [
        (
            race_context,
            scenario,
            str(r.JoinField),
            float(r.score_pp),
            float(r.se_pp),
            _to_int_or_none(r.n_races_used),
            _to_int_or_none(r.n_labeled_votes),
            methodology_version,
        )
        for r in df.itertuples(index=False)
    ]

    with sqlite3.connect(str(DB_PATH)) as conn:
        ensure_schema(conn)
        conn.executemany(
            """
            INSERT INTO precinct_progressive_scores
                (race_context, scenario, JoinField, score_pp, se_pp,
                 n_races_used, n_labeled_votes, methodology_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (race_context, scenario, JoinField, methodology_version)
            DO UPDATE SET
                score_pp        = excluded.score_pp,
                se_pp           = excluded.se_pp,
                n_races_used    = excluded.n_races_used,
                n_labeled_votes = excluded.n_labeled_votes,
                created_at      = CURRENT_TIMESTAMP
            """,
            params,
        )
        conn.commit()
    return len(params)


def load_canonical_chicago_joinfields() -> dict[str, str]:
    """
    Build a map from uppercase JoinField → the canonical mixed-case form
    used in election_results. Useful for loaders whose source data has
    been uppercased.
    """
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT JoinField FROM election_results
            WHERE JoinField LIKE 'CITY OF CHICAGO:%'
            """
        ).fetchall()
    return {r[0].upper(): r[0] for r in rows}
