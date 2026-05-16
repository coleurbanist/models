"""
Pollster rating objects and quality score derivation.

Supports two rating agencies:
  VoteHub       — reports the percentage of polls within the stated margin of error.
                  Grades: A+ >95%, A >75%, B >65%, C >61%, D >50%.
                  We store the raw percentage (0.0–1.0), not the letter grade.

  Silver Bulletin — reports letter grades with +/- modifiers at every level.
                  When a pollster has sparse data, Silver rounds and reports a
                  combined grade like "A/B" or "B/C" instead of a single letter.

Quality score
-------------
Each agency's rating is converted to a 0–1 quality score using the mappings
below. When both agencies have rated a pollster, the two scores are averaged.
When only one has, that score is used alone. When neither has, 0.7 is returned
as a neutral default (equivalent to a low-B pollster).

House effects
-------------
Stored as {candidate: pp} where positive means the pollster over-estimates
that candidate. Applied as a subtraction when aggregating toplines.
The house_effect here is a systematic bias in the pollster's methodology and
is separate from (and stacks with) the per-poll manual adjustments and the
internal candidate discount.

JSON format (pollster_db.json)
------------------------------
{
  "ppp": {
    "name": "Public Policy Polling",
    "votehub_pct_within_moe": 0.78,
    "silver_grade": "B+",
    "silver_house_effect_lean": -0.20,
    "votehub_house_effect_lean": -0.10,
    "house_effect": {"Candidate A": 1.5, "Candidate B": -0.5}
  },
  "emerson": {
    "name": "Emerson College Polling",
    "votehub_pct_within_moe": 0.82,
    "silver_house_effect_lean": null,
    "votehub_house_effect_lean": null,
    "house_effect": {}
  },
  "internal_xyz": {
    "name": "XYZ Campaign Internal",
    "silver_grade": "B/C"
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── Silver Bulletin grade → quality score ──────────────────────────────────
#
# Aligned approximately with VoteHub's thresholds:
#   VoteHub A+ > 95%,  A > 75%,  B > 65%,  C > 61%,  D > 50%
#
# Combined grades (A/B, B/C, C/D) are Silver's notation for sparse data —
# they represent uncertainty about where in the range the pollster falls,
# so they're mapped to the midpoint between the two adjacent grades.

SILVER_GRADE_QUALITY: dict[str, float] = {
    "A+":  0.97,
    "A":   0.93,
    "A-":  0.88,
    "A/B": 0.82,  # sparse — between A- and B+
    "B+":  0.78,
    "B":   0.73,
    "B-":  0.68,
    "B/C": 0.64,  # sparse — between B- and C+
    "C+":  0.62,
    "C":   0.59,
    "C-":  0.55,
    "C/D": 0.52,  # sparse — between C- and D+
    "D+":  0.52,
    "D":   0.50,
    "D-":  0.47,
    "F":   0.35,
}


@dataclass
class PollsterRating:
    pollster_id:   str
    pollster_name: str

    # VoteHub: fraction of polls within the pollster's stated margin of error.
    # Enter as a decimal (0.78, not 78). None if VoteHub hasn't rated them.
    votehub_pct_within_moe: float | None = None

    # Silver Bulletin letter grade. Accepts +/- modifiers and combined grades
    # (e.g. "A-", "B+", "A/B"). None if Silver Bulletin hasn't rated them.
    silver_grade: str | None = None

    # Per-agency directional lean estimates (pp). Negative = R/conservative, positive = D/progressive.
    # Enter whichever agencies have published a lean estimate; leave the other None.
    # The computed house_effect_lean property weights them 2/3 Silver, 1/3 VoteHub
    # when both are present; falls back to whichever single value is available.
    silver_house_effect_lean: float | None = None
    votehub_house_effect_lean: float | None = None

    # Per-candidate bias overrides in percentage points (optional fine-tuning).
    # Positive = pollster consistently over-estimates this candidate.
    # Use this when a pollster's bias toward a specific candidate doesn't follow
    # the general ideological lean (e.g. a pollster that over-estimates one
    # incumbent regardless of ideology).
    house_effect: dict[str, float] = field(default_factory=dict)

    # Manual quality grade — your own assessment using the Silver Bulletin scale.
    # Only applied when neither Silver Bulletin nor VoteHub has rated this pollster.
    # Use the same grade format as silver_grade (e.g. "B-", "C+", "B/C").
    manual_grade: str | None = None

    @property
    def house_effect_lean(self) -> float | None:
        """
        Weighted lean: 2/3 Silver Bulletin + 1/3 VoteHub.
        Uses whichever single value is available if only one agency has rated the pollster.
        Returns None if neither agency has provided a lean estimate.
        """
        silver = self.silver_house_effect_lean
        votehub = self.votehub_house_effect_lean
        if silver is not None and votehub is not None:
            return (2 / 3) * silver + (1 / 3) * votehub
        return silver if silver is not None else votehub

    @property
    def votehub_quality(self) -> float | None:
        """VoteHub score as 0–1, or None if not rated."""
        if self.votehub_pct_within_moe is None:
            return None
        return max(0.0, min(1.0, self.votehub_pct_within_moe))

    @property
    def silver_quality(self) -> float | None:
        """Silver Bulletin grade converted to 0–1, or None if not rated."""
        if self.silver_grade is None:
            return None
        score = SILVER_GRADE_QUALITY.get(self.silver_grade)
        if score is None:
            raise ValueError(
                f"Unknown Silver Bulletin grade '{self.silver_grade}' for "
                f"pollster '{self.pollster_id}'. "
                f"Valid grades: {sorted(SILVER_GRADE_QUALITY)}"
            )
        return score

    @property
    def manual_quality(self) -> float | None:
        """Manual grade converted to 0–1 using the Silver Bulletin scale, or None if not set."""
        if self.manual_grade is None:
            return None
        score = SILVER_GRADE_QUALITY.get(self.manual_grade)
        if score is None:
            raise ValueError(
                f"Unknown grade '{self.manual_grade}' for pollster '{self.pollster_id}'. "
                f"Valid grades: {sorted(SILVER_GRADE_QUALITY)}"
            )
        return score

    @property
    def quality(self) -> float:
        """
        Combined 0–1 pollster quality score.
        Averages whichever official agencies have rated this pollster.
        Falls back to manual_grade if set, then 0.7 if nothing is available.
        """
        scores = [s for s in (self.votehub_quality, self.silver_quality) if s is not None]
        if scores:
            return sum(scores) / len(scores)
        return self.manual_quality if self.manual_quality is not None else 0.7


# ── I/O ───────────────────────────────────────────────────────────────────


def load_pollster_db(path: Path) -> dict[str, PollsterRating]:
    """
    Load a pollster_db.json and return {pollster_id: PollsterRating}.
    See module docstring for the expected JSON format.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    db: dict[str, PollsterRating] = {}
    for pid, entry in raw.items():
        if not isinstance(entry, dict):
            continue  # skip _comment keys and other metadata
        db[pid] = PollsterRating(
            pollster_id=pid,
            pollster_name=entry.get("name", pid),
            votehub_pct_within_moe=entry.get("votehub_pct_within_moe"),
            silver_grade=entry.get("silver_grade"),
            silver_house_effect_lean=entry.get("silver_house_effect_lean"),
            votehub_house_effect_lean=entry.get("votehub_house_effect_lean"),
            house_effect=entry.get("house_effect", {}),
            manual_grade=entry.get("manual_grade"),
        )
    return db


def to_poll_weighting_format(db: dict[str, PollsterRating]) -> dict[str, dict]:
    """
    Convert a pollster DB to the dict format poll_weighting._apply_pollster_ratings
    expects: {pollster_id: {"quality": float, "house_effect_adjustment": {...}, "lean": float|None}}

    "lean" is kept in the output so poll_weighting can translate it to per-candidate
    house effects once it knows the race's ideological bloc structure.
    """
    return {
        pid: {
            "quality": rating.quality,
            "house_effect_adjustment": rating.house_effect,
            "lean": rating.house_effect_lean,
        }
        for pid, rating in db.items()
    }
