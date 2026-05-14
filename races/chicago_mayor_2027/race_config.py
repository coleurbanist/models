"""
Chicago Mayoral 2027 — race configuration scaffold.

Fill in candidates, polls, and shapefile paths as they become available.
See doc §9 for the full list of things that differ from IL-09.

Key differences from IL-09:
  - has_runoff=True (top-two if no majority)
  - regions = ward groups (not counties)
  - joinfield_format="CHICAGO" (WARD XX PRECINCT YY)
  - shapefile_crosstab = ward boundaries (if polls provide ward-level crosstabs)
  - Empirical pollster quality from pollster_calibration/pollster_ratings.json
"""

from pathlib import Path
from core.race_config import RaceConfig

_DATA_DIR = Path(__file__).parent / "data"
_OUTPUT_DIR = Path(__file__).parent / "outputs"

# Ward groups — 6 geographic clusters for regional breakdown.
# Refine after reviewing 2023 ward-level results.
# See doc §12.3 for the full ward-to-group mapping.
WARD_GROUPS = [
    "North Lakefront",
    "Northwest Side",
    "West Side",
    "Near West / Latino",
    "South Side",
    "Southwest Side",
]

CONFIG = RaceConfig(
    # ── Identity ────────────────────────────────────────────────────────────
    race_id="chicago_mayor_2027",
    race_label="Chicago Mayoral Race 2027",
    election_date="2027-02-23",  # placeholder — confirm when election is set

    # ── Candidates ──────────────────────────────────────────────────────────
    # TODO: fill in when field is known
    candidates=["Candidate A", "Candidate B", "Candidate C"],

    colors={
        "Candidate A": "#2196F3",
        "Candidate B": "#F44336",
        "Candidate C": "#4CAF50",
    },

    ideological_blocs=[
        # TODO: fill in based on ideological alignment
        ["Candidate A"],
        ["Candidate B", "Candidate C"],
    ],

    # ── Polls ────────────────────────────────────────────────────────────────
    # TODO: add polls as they are released.
    # Use pollster_ratings.json (from pollster_calibration.py) for quality scores.
    polls=[],

    # ── Undecided allocation & modeling constants ────────────────────────────
    # TODO: fill in based on candidate profiles
    undecided_allocation={
        "Candidate A": 1.0,
        "Candidate B": 1.0,
        "Candidate C": 1.0,
    },
    favorability_blend=0.25,
    second_choice_strength=0.60,

    # ── Simulation parameters ────────────────────────────────────────────────
    n_sim_district=1_000_000,
    n_sim_precinct=50_000,
    moe_district=4.4,
    moe_precinct=6.0,

    # ── Geography (ward-based) ───────────────────────────────────────────────
    # Chicago mayoral uses individual wards as regions (not election authorities —
    # there is only one election authority: the Chicago BOE).
    region_type="ward",
    regions=WARD_GROUPS,

    data_dir=_DATA_DIR,
    output_dir=_OUTPUT_DIR,

    # TODO: update shapefile paths once Chicago precinct shapefile is obtained.
    # Confirm the JoinField format (likely "WARD XX PRECINCT YY") before running.
    shapefile_precinct=Path("shapefile/chicago_precincts.shp"),
    shapefile_district=Path("shapefile/chicago_city_boundary.shp"),
    shapefile_crosstab=None,  # set to ward boundaries shapefile if polls provide ward crosstabs

    joinfield_format="CHICAGO",

    # Early vote modeling uses Chicago BOE's election-night demographic release.
    # Chicago BOE posts age + gender breakdowns ~30 min after 7pm; these are used
    # with demographic crosstabs from polls to distribute early votes by candidate.
    banked_vote_mode="chicago",

    # votes.csv not used for Chicago mayor — early votes handled by chicago_early_votes.py
    votes_csv_region_map={},

    # ── Runoff tracking ───────────────────────────────────────────────────────
    has_runoff=True,
    runoff_threshold=0.50,  # must exceed 50% to win outright; otherwise top-two runoff

    # ── Pollster ratings ──────────────────────────────────────────────────────
    # Path to empirical quality ratings derived from 2023 mayoral polling.
    # Run pollster_calibration/pollster_calibration.py to generate this file.
    pollster_ratings_path=Path(__file__).parent.parent.parent
        / "pollster_calibration" / "pollster_ratings.json",

    # ── Race-specific constraints ─────────────────────────────────────────────
    # Ward organization / machine effects can be encoded here once ward-level
    # endorsement data is available.
    extra_constraints={},
)
