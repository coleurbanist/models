"""
Election night tick loop logger.

Writes one CSV row per tick to election_night_log.csv with:
  - timestamp
  - reported precincts / total precincts per region
  - reported vote totals per candidate
  - projected totals (blended live + model estimates)
  - current win probabilities

Usage:
    logger = ElectionLogger(config, output_path)
    logger.tick(reported_votes, reported_precincts, district_results)
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..race_config import RaceConfig


# Win probability below which a candidate is eliminated.
ELIMINATION_THRESHOLD = 0.02
# Guard: elimination only fires after this fraction of precincts have reported.
ELIMINATION_PRECINCT_GUARD = 0.20


class ElectionLogger:
    def __init__(self, config: RaceConfig, output_path: Path) -> None:
        self.config = config
        self.output_path = output_path
        self._eliminated: set[str] = set()
        self._fieldnames: list[str] | None = None
        self._writer: csv.DictWriter | None = None
        self._file = None

    def _ensure_open(self, row: dict) -> None:
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = self.output_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

    def tick(
        self,
        reported_votes: dict[str, int],      # {candidate: vote_count}
        reported_precincts: dict[str, int],   # {region: precincts_reported}
        total_precincts: dict[str, int],      # {region: total_precincts}
        district_results: dict[str, Any],     # latest win_probs + median shares
        timestamp: str | None = None,
    ) -> list[str]:
        """
        Log one tick. Returns list of newly eliminated candidate names (may be empty).
        """
        ts = timestamp or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        total_reported = sum(reported_precincts.values())
        total_all = sum(total_precincts.values())
        pct_reporting = total_reported / total_all if total_all > 0 else 0.0

        row: dict[str, Any] = {
            "timestamp": ts,
            "pct_precincts_reporting": round(pct_reporting, 4),
            "total_precincts_reporting": total_reported,
            "total_precincts": total_all,
        }

        total_reported_votes = sum(reported_votes.values())
        for cand in self.config.candidates:
            votes = reported_votes.get(cand, 0)
            row[f"reported_votes_{cand}"] = votes
            row[f"reported_pct_{cand}"] = (
                votes / total_reported_votes if total_reported_votes > 0 else 0.0
            )
            row[f"win_prob_{cand}"] = district_results.get("win_probs", {}).get(cand, 0.0)
            row[f"median_pct_{cand}"] = district_results.get("median_vote_shares", {}).get(cand, 0.0)

        for region, n_reported in reported_precincts.items():
            row[f"precincts_{region}"] = n_reported
            row[f"precincts_total_{region}"] = total_precincts.get(region, 0)

        self._ensure_open(row)
        self._writer.writerow(row)
        self._file.flush()

        # Elimination check
        newly_eliminated = []
        if pct_reporting >= ELIMINATION_PRECINCT_GUARD:
            for cand in self.config.candidates:
                if cand in self._eliminated:
                    continue
                wp = district_results.get("win_probs", {}).get(cand, 1.0)
                if wp < ELIMINATION_THRESHOLD:
                    self._eliminated.add(cand)
                    newly_eliminated.append(cand)

        return newly_eliminated

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
