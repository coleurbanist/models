"""
Chicago Mayoral 2027 — race configuration scaffold.

Fill in candidates, polls, and shapefile paths as they become available.

Key differences from IL-09:
  - has_runoff=True (top-two if no majority)
  - regions = all 50 individual wards
  - joinfield_format="CHICAGO" (WARD XX PRECINCT YY)
  - shapefile_crosstab = ward boundaries (if polls provide ward-level crosstabs)
"""

from pathlib import Path
from core.race_config import RaceConfig

_DATA_DIR = Path(__file__).parent / "data"
_OUTPUT_DIR = Path(__file__).parent / "outputs"
_SHAPEFILES_DIR = Path(__file__).parent.parent.parent / "Shapefiles"

WARDS = [f"Ward {i}" for i in range(1, 51)]


CONFIG = RaceConfig(
    # ── Identity ────────────────────────────────────────────────────────────
    race_id="chicago_mayor_2027",
    race_label="Chicago Mayoral Race 2027",
    election_date="2027-02-23",  # placeholder — confirm when election is set

    # ── Candidates ──────────────────────────────────────────────────────────
    
    candidates=["Brandon Johnson", "Susana Mendoza", "Alexi Giannoulias"],

    colors={
        "Brandon Johnson": "#00FFF2FF",
        "Alexi Giannoulias": "#FF8800",
        "Susana Mendoza": "#4CAF50",
    },

    ideological_blocs=[
        ["Susana Mendoza"],
        ["Alexi Giannoulias"],
        ["Brandon Johnson"],
    ],
    bloc_ideological_positions=[-0.8, 0.2, 0.8],

    # ── Polls ────────────────────────────────────────────────────────────────
    # Add polls to polls_round1.json (first round) or polls_runoff.json (runoff).
    # Run with --round round1 (default) or --round runoff to select which set.
    polls=[],  # loaded at runtime from polls_round1_path / polls_runoff_path
    polls_round1_path=Path(__file__).parent / "polls_round1.json",
    polls_runoff_path=Path(__file__).parent / "polls_runoff.json",

    # ── Undecided allocation & modeling constants ────────────────────────────
    # TODO: fill in based on candidate profiles
    undecided_allocation={
        "Alexi Giannoulias": 1.0,
        "Brandon Johnson": 1.0,
        "Susana Mendoza": 1.0,
    },
    favorability_blend=0.25,
    second_choice_strength=0.60,

    late_poll_multiplier=6.0,

    bloc_ideology_strength=1,

    ideology_race_weights={
        "pct_white":    1.0,
        "pct_asian":    0.9,
        "pct_hispanic": 0.55,
        "pct_black":    0.35,
    },

    # Johnson incumbency advantage among Black voters (mirrors Lightfoot 2023 boost).
    # Tune via compare_results.py once election results are available.
    candidate_precinct_boosts={
        "Brandon Johnson": {"pct_black": 1.1},
    },

    race_context="chicago_mayor",

    # ── Simulation parameters ────────────────────────────────────────────────
    n_sim_district=1_000_000,
    n_sim_precinct=50_000,
    moe_district=4.4,
    moe_precinct=6.0,

    # ── Geography (ward-based) ───────────────────────────────────────────────
    # Chicago mayoral uses individual wards as regions (not election authorities —
    # there is only one election authority: the Chicago BOE).
    region_type="ward",
    regions=WARDS,

    data_dir=_DATA_DIR,
    output_dir=_OUTPUT_DIR,

    shapefile_precinct=_SHAPEFILES_DIR / "chicago_precincts.geojson",
    shapefile_crosstab=_SHAPEFILES_DIR / "chicago_wards.geojson",

    joinfield_format="CHICAGO",

    # Early vote modeling uses Chicago BOE's election-night demographic release.
    # Chicago BOE posts age + gender breakdowns ~30 min after 7pm; these are used
    # with demographic crosstabs from polls to distribute early votes by candidate.
    banked_vote_mode="chicago",

    # Per-precinct turnout weights averaged from prior Chicago mayoral races.
    turnout_races=[
        {"race_type": "chicago_mayor", "election_type": "municipal", "year": 2023},
        {"race_type": "chicago_mayor", "election_type": "municipal", "year": 2019},
    ],

    # votes.csv not used for Chicago mayor — early votes handled by chicago_early_votes.py
    votes_csv_region_map={},

    # ── Runoff tracking ───────────────────────────────────────────────────────
    has_runoff=True,
    runoff_threshold=0.50,  # must exceed 50% to win outright; otherwise top-two runoff

    # ── Pollster ratings ──────────────────────────────────────────────────────
    # Shared pollster ratings database (VoteHub + Silver Bulletin).
    # Add pollsters to pollster_db.json at the project root as polls come in.
    pollster_ratings_path=Path(__file__).parent.parent.parent / "pollster_db.json",

    # ── Race-specific constraints ─────────────────────────────────────────────
    # Ward organization / machine effects can be encoded here once ward-level
    # endorsement data is available.
    extra_constraints={},
)
