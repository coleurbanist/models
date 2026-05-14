"""
Chicago election-night ward inference engine.

Uses aldermanic race results as a proxy for mayoral turnout before
mayoral precincts fully report. See doc §12 for full design.

Required data files (in races/chicago_mayor_2027/data/):
    ward_group_map.json       {ward_number_str: group_name}
    ward_turnout_prior.json   {ward_number_str: expected_votes}
    ward_share_prior.json     {group_name: {candidate: pct_fraction}}

Runtime input:
    aldermanic_results_live.csv   rows=ward_precinct, col=total_votes_cast

Usage:
    engine = WardInferenceEngine(config)
    estimates = engine.estimate(aldermanic_csv_path, current_polling_snapshot)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..race_config import RaceConfig


# Contested aldermanic races generate full mayoral-equivalent turnout.
# Uncontested wards get a turnout uplift factor applied.
UNCONTESTED_UPLIFT = 1.10


class WardInferenceEngine:
    def __init__(self, config: RaceConfig) -> None:
        self.config = config
        self._ward_group_map: dict[str, str] = {}
        self._turnout_prior: dict[str, float] = {}
        self._share_prior: dict[str, dict[str, float]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        data_dir = self.config.data_dir

        wgm_path = data_dir / "ward_group_map.json"
        wtp_path = data_dir / "ward_turnout_prior.json"
        wsp_path = data_dir / "ward_share_prior.json"

        for path in (wgm_path, wtp_path, wsp_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Ward inference data file not found: {path}\n"
                    f"Build it from 2023 BOE results before using this engine."
                )

        with wgm_path.open(encoding="utf-8") as f:
            self._ward_group_map = json.load(f)
        with wtp_path.open(encoding="utf-8") as f:
            self._turnout_prior = {k: float(v) for k, v in json.load(f).items()}
        with wsp_path.open(encoding="utf-8") as f:
            self._share_prior = json.load(f)

        self._loaded = True

    def estimate(
        self,
        aldermanic_csv_path: Path,
        polling_snapshot: dict[str, Any],
        contested_wards: set[int] | None = None,
    ) -> dict[str, Any]:
        """
        Phase A: compute reported turnout per ward group from aldermanic results.
        Phase B: estimate mayoral vote shares using credibility-weighted blend
                 of ward-share prior and polling snapshot.

        Returns:
        {
            "by_ward_group": {
                group: {
                    "reported_votes": int,
                    "expected_votes": float,
                    "pct_reporting": float,
                    "candidate_estimates": {candidate: int},
                    "candidate_pcts": {candidate: float},
                }
            },
            "district_estimates": {candidate: int},
            "district_total_estimated": int,
        }
        """
        self._load()

        if not aldermanic_csv_path.exists():
            return {"by_ward_group": {}, "district_estimates": {}, "district_total_estimated": 0}

        ald_df = pd.read_csv(str(aldermanic_csv_path))
        # Expected columns: ward (int), precinct (int), total_votes_cast (int)
        required = {"ward", "total_votes_cast"}
        missing = required - set(ald_df.columns)
        if missing:
            raise ValueError(f"aldermanic_results_live.csv missing columns: {missing}")

        # Phase A — aggregate aldermanic votes per ward
        ward_reported: dict[int, float] = {}
        for ward_num, grp in ald_df.groupby("ward"):
            ward_num = int(ward_num)
            votes = float(grp["total_votes_cast"].sum())
            # Uplift for uncontested wards
            if contested_wards is not None and ward_num not in contested_wards:
                votes *= UNCONTESTED_UPLIFT
            ward_reported[ward_num] = votes

        # Aggregate to ward groups
        group_reported: dict[str, float] = {g: 0.0 for g in self.config.regions}
        group_expected: dict[str, float] = {g: 0.0 for g in self.config.regions}

        for ward_str, group in self._ward_group_map.items():
            ward_num = int(ward_str)
            if group not in group_reported:
                continue
            group_reported[group] += ward_reported.get(ward_num, 0.0)
            group_expected[group] += self._turnout_prior.get(ward_str, 0.0)

        # Phase B — credibility-weighted share estimation
        snap_shares = polling_snapshot.get("baseline", {})

        results: dict[str, Any] = {}
        for group in self.config.regions:
            expected = group_expected.get(group, 0.0)
            reported = group_reported.get(group, 0.0)
            pct_reporting = min(reported / expected, 1.0) if expected > 0 else 0.0

            # weight_actual ramps from 0 to 1 as the ward reports
            weight_actual = pct_reporting
            weight_prior = 1.0 - weight_actual

            prior = self._share_prior.get(group, {})
            # Scale prior to current polling snapshot via 60/40 blend
            blended_prior: dict[str, float] = {
                c: 0.60 * prior.get(c, snap_shares.get(c, 0.0))
                + 0.40 * snap_shares.get(c, 0.0)
                for c in self.config.candidates
            }
            prior_total = sum(blended_prior.values())
            if prior_total > 1e-9:
                blended_prior = {c: v / prior_total for c, v in blended_prior.items()}

            # Final estimate = credibility blend
            # (actual only available once mayoral results start arriving)
            final_shares = {c: weight_prior * blended_prior.get(c, 0.0) for c in self.config.candidates}
            final_total = sum(final_shares.values())
            if final_total > 1e-9:
                final_shares = {c: v / final_total for c, v in final_shares.items()}

            candidate_estimates = {
                c: round(final_shares[c] * reported) for c in self.config.candidates
            }

            results[group] = {
                "reported_votes": round(reported),
                "expected_votes": round(expected),
                "pct_reporting": round(pct_reporting, 4),
                "candidate_estimates": candidate_estimates,
                "candidate_pcts": final_shares,
            }

        # District totals
        district_estimates: dict[str, int] = {c: 0 for c in self.config.candidates}
        district_total = 0
        for group_data in results.values():
            for c in self.config.candidates:
                district_estimates[c] += group_data["candidate_estimates"][c]
            district_total += group_data["reported_votes"]

        return {
            "by_ward_group": results,
            "district_estimates": district_estimates,
            "district_total_estimated": district_total,
        }


# ── Utility: build ward_group_map.json from a manual mapping ──────────────

def build_ward_group_map(ward_assignments: dict[int, str], output_path: Path) -> None:
    """
    ward_assignments: {ward_number: group_name}
    Must cover all 50 Chicago wards.
    """
    missing = [w for w in range(1, 51) if w not in ward_assignments]
    if missing:
        raise ValueError(f"Missing ward assignments for wards: {missing}")
    data = {str(k): v for k, v in ward_assignments.items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ward group map to {output_path}")
