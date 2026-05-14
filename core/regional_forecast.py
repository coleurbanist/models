"""
Aggregate precinct-level probabilities into regional summaries.

Region assignment is derived from JoinField prefix (never from misaligned flag columns).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .race_config import RaceConfig
from .precinct_pipeline import assign_region_from_joinfield


def generate_regional_forecast(
    config: RaceConfig,
    precinct_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Returns {region_name: {...}} with:
        expected_turnout:    total expected votes in region
        turnout_share_pct:   fraction of district total
        num_precincts:       int
        vote_shares:         {candidate: turnout-weighted median_pct}
        expected_votes:      {candidate: expected vote count}
    """
    df = precinct_df.copy()

    # Ensure region column is present
    if "region" not in df.columns:
        df["region"] = df["joinfield"].apply(
            lambda jf: assign_region_from_joinfield(jf, config.joinfield_format)
        )

    if "turnout_weight" not in df.columns:
        df["turnout_weight"] = 100.0

    result: dict[str, dict[str, Any]] = {}
    total_turnout = df["turnout_weight"].sum()

    for region in config.regions:
        mask = df["region"] == region
        sub = df[mask]
        if sub.empty:
            result[region] = {
                "expected_turnout": 0,
                "turnout_share_pct": 0.0,
                "num_precincts": 0,
                "vote_shares": {c: 0.0 for c in config.candidates},
                "expected_votes": {c: 0.0 for c in config.candidates},
            }
            continue

        region_turnout = sub["turnout_weight"].sum()
        weights = sub["turnout_weight"].values

        vote_shares: dict[str, float] = {}
        expected_votes: dict[str, float] = {}
        for c in config.candidates:
            col = f"median_pct_{c}"
            if col in sub.columns:
                wt_avg = np.average(sub[col].values, weights=weights)
            else:
                wt_avg = 0.0
            vote_shares[c] = float(wt_avg)
            expected_votes[c] = float(wt_avg * region_turnout)

        result[region] = {
            "expected_turnout": float(region_turnout),
            "turnout_share_pct": float(region_turnout / total_turnout) if total_turnout > 0 else 0.0,
            "num_precincts": int(mask.sum()),
            "vote_shares": vote_shares,
            "expected_votes": expected_votes,
        }

    return result
