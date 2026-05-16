"""
Chicago Mayoral 2023 — backtesting race configuration.

Used to validate the model against known outcomes.

Round 1 results (2023-02-28):
  Paul Vallas       33.8%
  Brandon Johnson   21.7%
  Lori Lightfoot    17.1%  (incumbent; eliminated — biggest upset)
  Chuy García       13.7%
  Willie Wilson      8.9%
  Others             4.8%  (Ja'Mal Green, Sophia King, Roderick Sawyer, Tom Tunney)

Runoff results (2023-04-04):
  Brandon Johnson   51.4%
  Paul Vallas       48.6%

Ideological blocs (round 1):
  Progressive:  Johnson, García
  Moderate:     Lightfoot  (ran as incumbent centrist; drew from both sides)
  Conservative: Vallas
  Independent:  Wilson     (largely independent of ideological spectrum)
"""

from pathlib import Path
from core.race_config import RaceConfig

_DATA_DIR = Path(__file__).parent / "data"
_OUTPUT_DIR = Path(__file__).parent / "outputs"
_SHAPEFILES_DIR = Path(__file__).parent.parent.parent / "Shapefiles"  # models/Shapefiles/

WARDS = [f"Ward {i}" for i in range(1, 51)]


CONFIG = RaceConfig(
    # ── Identity ────────────────────────────────────────────────────────────
    race_id="chicago_mayor_2023",
    race_label="Chicago Mayoral Race 2023",
    election_date="2023-02-28",

    # ── Candidates ──────────────────────────────────────────────────────────
    # Modeling the five candidates who cleared 5%. The remaining four
    # (Green, King, Sawyer, Tunney) totaled ~4.8% combined and are omitted;
    # their votes will be absorbed into the undecided pool.
    candidates=["Paul Vallas", "Brandon Johnson", "Lori Lightfoot", "Chuy García", "Willie Wilson", "Kam Buckner", "Sophia King", "Jamal Green"],

    colors={
        "Paul Vallas":      "#15C048",
        "Brandon Johnson":  "#F270AD",
        "Lori Lightfoot":   "#35F2F6",
        "Chuy García":      "#FF0019",
        "Willie Wilson":    "#FF9800",
        "Kam Buckner":      "#9B59B6",
        "Sophia King":      "#E3E622",
        "Jamal Green":      "#00224E4C",
    },

    # Bloc structure: Johnson + García share progressive precinct correlation;
    # Lightfoot is her own bloc (idiosyncratic incumbent dynamics);
    # Vallas is his own bloc (conservative/moderate); Wilson is independent.
    ideological_blocs=[
        ["Paul Vallas"],
        ["Lori Lightfoot"],
        ["Chuy García"],
        ["Brandon Johnson", "Kam Buckner"],
        ["Willie Wilson"],
    ],
    # Positions on an ideological axis (-1.0 = far left, +1.0 = far right).
    # Vallas ran as the most conservative candidate; Johnson/García as the
    # progressive bloc; Lightfoot occupied the center-left as the incumbent;
    # Wilson is roughly center (community/religious independent, not ideology-driven).
    bloc_ideological_positions=[-0.8, 0.1,0.4, 1, -0.6],

    # ── Polls ────────────────────────────────────────────────────────────────
    polls=[],
    polls_round1_path=Path(__file__).parent / "polls_round1.json",
    polls_runoff_path=Path(__file__).parent / "polls_runoff.json",

    # ── Undecided allocation ─────────────────────────────────────────────────
    # Equal weights initially; adjust once you've reviewed the polls.
    undecided_allocation={
        "Paul Vallas":     1.0,
        "Brandon Johnson": 1.0,
        "Lori Lightfoot":  1.0,
        "Chuy García":     1.0,
        "Willie Wilson":   1.0,
        "Sophia King": 0.3,
        "Jamal Green": 0.1
    },
    favorability_blend=0.25,
    second_choice_strength=0.60,

    # ── Simulation parameters ────────────────────────────────────────────────
    n_sim_district=1_000_000,
    n_sim_precinct=50_000,
    moe_district=4.4,
    moe_precinct=6.0,

    # Extra weight for polls fielded within the final 7 days before election day.
    late_poll_multiplier=6.0,

    # Logit boost from precinct progressive score × candidate bloc position.
    # 1.0 ≈ ±1 logit unit swing from most conservative to most progressive precinct
    # for a candidate at the ideological extreme. Tune via compare_results.py.
    bloc_ideology_strength=1,

    # How much the ideology signal matters per racial group (0=ignore, 1=full).
    # Black and Hispanic voters sort less on ideology than white voters because
    # racial solidarity and co-ethnic candidate pulls compete with ideology.
    ideology_race_weights={
        "pct_white":    1.0,
        "pct_asian":    0.9,
        "pct_hispanic": 0.55,
        "pct_black":    0.35,
    },

    race_context="chicago_mayor",

    # Lightfoot incumbency advantage among Black voters.
    # logit_delta = boost * pct_black per precinct; tune via compare_results.py.
    candidate_precinct_boosts={
        "Lori Lightfoot": {"pct_black": 1.1},
    },

    # ── Geography (ward-based) ───────────────────────────────────────────────
    region_type="ward",
    regions=WARDS,

    data_dir=_DATA_DIR,
    output_dir=_OUTPUT_DIR,

    shapefile_precinct=_SHAPEFILES_DIR / "chicago_precincts.geojson",
    shapefile_crosstab=_SHAPEFILES_DIR / "chicago_wards.geojson",

    joinfield_format="CHICAGO",

    banked_vote_mode="chicago",

    # Per-precinct turnout weights averaged from prior Chicago mayoral races.
    turnout_races=[
        {"race_type": "chicago_mayor", "election_type": "municipal", "year": 2019},
        {"race_type": "chicago_mayor", "election_type": "municipal", "year": 2015},
    ],

    votes_csv_region_map={},

    # ── Runoff ────────────────────────────────────────────────────────────────
    has_runoff=True,
    runoff_threshold=0.50,

    # ── Pollster ratings ──────────────────────────────────────────────────────
    pollster_ratings_path=Path(__file__).parent.parent.parent / "pollster_db.json",

    extra_constraints={},
)
