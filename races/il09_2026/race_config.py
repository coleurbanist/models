"""
IL-09 2026 Democratic Primary — race configuration.

Candidates: Fine, Biss, Abughazaleh, Simmons, Amiwala, Andrew, Huynh
Election date: 2026-03-17
"""

from pathlib import Path
from core.race_config import RaceConfig

_DATA_DIR = Path(__file__).parent / "data"
_OUTPUT_DIR = Path(__file__).parent / "outputs"

CONFIG = RaceConfig(
    # ── Identity ────────────────────────────────────────────────────────────
    race_id="il09_2026",
    race_label="IL-09 Democratic Primary",
    election_date="2026-03-17",

    # ── Candidates ──────────────────────────────────────────────────────────
    candidates=["Fine", "Biss", "Abughazaleh", "Simmons", "Amiwala", "Andrew", "Huynh"],

    colors={
        "Fine":        "#2196F3",  # blue
        "Biss":        "#4CAF50",  # green
        "Abughazaleh": "#F44336",  # red
        "Simmons":     "#FF9800",  # orange
        "Amiwala":     "#9C27B0",  # purple
        "Andrew":      "#00BCD4",  # cyan
        "Huynh":       "#FF5722",  # deep orange
    },

    # Confirmed from IL-09 2026 post-mortem error correlations:
    # Fine↔Abughazaleh r=−0.61 (opposing blocs); Abughazaleh↔Simmons r=+0.32 (co-moving).
    # Biss straddles ideological lines but correlates with progressive candidates.
    ideological_blocs=[
        ["Fine", "Andrew"],
        ["Biss", "Abughazaleh", "Simmons", "Amiwala", "Huynh"],
    ],

    # ── Polls ────────────────────────────────────────────────────────────────
    # Poll dict fields:
    #   pollster_id, pollster_name, pollster_quality (0–1), field_end,
    #   sample_size, moe, is_internal,
    #   house_effect: {candidate: pp} — positive = pollster over-estimates that candidate
    #   topline: {candidate: pct} — undecideds excluded, renormalized to decided voters
    #   crosstabs: {senate_district: {candidate: pct}} — same normalization
    #   demographic_crosstabs: {group: {candidate: pct}} — race/ideology, same normalization
    #   favorability: {candidate: {"favorable": pct, "unfavorable": pct}}
    #   second_choice: {from_cand: {to_cand: pct}} — normalized excl. no-second
    #
    # All topline/crosstab values are renormalized to exclude undecideds and minor
    # "others" bucket. Source: poll_config.py from il9prediction_and_tracker.
    # pollster_quality = old 1–5 rating ÷ 5.
    polls=[
        # ── PPP / RoundTable Wave 2  (2026-03-10, n=741) ──────────────────
        # Topline from results dict (Amiwala=6,Andrew=7,Huynh=1,Biss=24,Fine=14,
        # Abu=20,Sim=10) divided by decided sum=82; undecided=17%.
        {
            "pollster_id":      "ppp_wave2",
            "pollster_name":    "PPP / RoundTable Wave 2",
            "pollster_quality": 0.90,
            "field_end":        "2026-03-10",
            "sample_size":      741,
            "moe":              3.6,
            "is_internal":      False,
            "topline": {
                "Fine":        17.1,
                "Biss":        29.3,
                "Abughazaleh": 24.4,
                "Simmons":     12.2,
                "Amiwala":      7.3,
                "Andrew":       8.5,
                "Huynh":        1.2,
            },
            # SD crosstabs from senate_district_crosstabs field, renormalized excl. undecided.
            # SD-7 decided sum=82, SD-8=88, SD-9=81.
            "crosstabs": {
                "SD-7": {
                    "Fine": 11.0, "Biss": 23.2, "Abughazaleh": 29.3,
                    "Simmons": 23.2, "Amiwala": 8.5, "Andrew": 1.2, "Huynh": 3.7,
                },
                "SD-8": {
                    "Fine": 13.6, "Biss": 23.9, "Abughazaleh": 25.0,
                    "Simmons": 11.4, "Amiwala": 14.8, "Andrew": 11.4, "Huynh": 0.0,
                },
                "SD-9": {
                    "Fine": 21.0, "Biss": 38.3, "Abughazaleh": 16.0,
                    "Simmons": 7.4, "Amiwala": 3.7, "Andrew": 13.6, "Huynh": 0.0,
                },
            },
            # Race crosstabs renormalized excl. undecided within each group.
            # white sum=85, black sum=70, hispanic sum=63, asian sum=90.
            "demographic_crosstabs": {
                "white":    {"Fine": 17.6, "Biss": 31.8, "Abughazaleh": 24.7, "Simmons": 11.8, "Amiwala": 4.7,  "Andrew": 8.2,  "Huynh": 1.2},
                "black":    {"Fine": 20.0, "Biss": 31.4, "Abughazaleh": 15.7, "Simmons": 28.6, "Amiwala": 0.0,  "Andrew": 4.3,  "Huynh": 0.0},
                "hispanic": {"Fine": 14.3, "Biss": 30.2, "Abughazaleh": 28.6, "Simmons": 12.7, "Amiwala": 11.1, "Andrew": 0.0,  "Huynh": 3.2},
                "asian":    {"Fine":  5.6, "Biss": 10.0, "Abughazaleh": 23.3, "Simmons":  4.4, "Amiwala": 43.3, "Andrew": 13.3, "Huynh": 0.0},
            },
            "favorability": {
                "Fine":        {"favorable": 28.0, "unfavorable": 50.0},
                "Biss":        {"favorable": 50.0, "unfavorable": 31.0},
                "Abughazaleh": {"favorable": 39.0, "unfavorable": 34.0},
                "Simmons":     {"favorable": 35.0, "unfavorable":  6.0},
                "Amiwala":     {"favorable": 35.0, "unfavorable":  9.0},
                "Andrew":      {"favorable": 24.0, "unfavorable": 15.0},
                "Huynh":       {"favorable": 21.0, "unfavorable": 10.0},
            },
            # Second choice from matrix; each row normalized over candidate sums (excl. no_second).
            "second_choice": {
                "Fine":        {"Biss": 50.6, "Abughazaleh": 18.2, "Andrew": 14.3, "Simmons":  7.8, "Amiwala":  6.5, "Huynh":  2.6},
                "Biss":        {"Fine": 29.9, "Abughazaleh": 27.3, "Simmons": 16.9, "Amiwala": 13.0, "Andrew": 10.4, "Huynh":  2.6},
                "Abughazaleh": {"Biss": 29.6, "Amiwala": 27.2, "Simmons": 24.7, "Fine":  7.4, "Huynh":  6.2, "Andrew":  4.9},
                "Simmons":     {"Biss": 32.4, "Amiwala": 28.4, "Andrew": 13.5, "Huynh":  9.5, "Fine":  9.5, "Abughazaleh":  6.8},
                "Amiwala":     {"Abughazaleh": 40.7, "Simmons": 18.7, "Biss": 18.7, "Andrew": 16.5, "Huynh":  3.3, "Fine":  2.2},
                "Andrew":      {"Biss": 23.9, "Fine": 22.5, "Amiwala": 21.1, "Abughazaleh": 14.1, "Simmons": 14.1, "Huynh":  4.2},
                "Huynh":       {"Biss": 31.5, "Fine": 28.3, "Simmons": 25.0, "Abughazaleh": 15.2, "Amiwala":  0.0, "Andrew":  0.0},
            },
        },

        # ── PPP / RoundTable Wave 1  (2026-02-21, n=501) ──────────────────
        # Topline from results (Amiwala=4,Andrew=5,Huynh=2,Biss=24,Fine=16,Abu=17,Sim=6)
        # divided by decided sum=74; undecided=22%.
        {
            "pollster_id":      "ppp_wave1",
            "pollster_name":    "PPP / RoundTable Wave 1",
            "pollster_quality": 0.90,
            "field_end":        "2026-02-21",
            "sample_size":      501,
            "moe":              4.4,
            "is_internal":      False,
            "topline": {
                "Fine":        21.6,
                "Biss":        32.4,
                "Abughazaleh": 23.0,
                "Simmons":      8.1,
                "Amiwala":      5.4,
                "Andrew":       6.8,
                "Huynh":        2.7,
            },
            # SD-7 decided=78, SD-8=73, SD-9=79, sd_other=63.
            "crosstabs": {
                "SD-7":    {"Fine":  7.7, "Biss": 30.8, "Abughazaleh": 28.2, "Simmons": 20.5, "Amiwala": 5.1, "Andrew":  1.3, "Huynh": 6.4},
                "SD-8":    {"Fine": 19.2, "Biss": 37.0, "Abughazaleh": 20.5, "Simmons":  5.5, "Amiwala": 5.5, "Andrew": 11.0, "Huynh": 1.4},
                "SD-9":    {"Fine": 30.4, "Biss": 34.2, "Abughazaleh": 16.5, "Simmons":  2.5, "Amiwala": 8.9, "Andrew":  7.6, "Huynh": 0.0},
                "sd_other":{"Fine": 28.6, "Biss": 30.2, "Abughazaleh": 28.6, "Simmons":  3.2, "Amiwala": 1.6, "Andrew":  7.9, "Huynh": 0.0},
            },
            # Race: white=76, black=67, hispanic=67, asian=72.
            # Ideology: very_liberal=79, somewhat_liberal=79, moderate=61.
            "demographic_crosstabs": {
                "white":            {"Fine": 19.7, "Biss": 34.2, "Abughazaleh": 21.1, "Simmons":  9.2, "Amiwala":  5.3, "Andrew":  7.9, "Huynh": 2.6},
                "black":            {"Fine": 10.4, "Biss": 32.8, "Abughazaleh": 23.9, "Simmons": 11.9, "Amiwala": 20.9, "Andrew":  0.0, "Huynh": 0.0},
                "hispanic":         {"Fine": 35.8, "Biss": 44.8, "Abughazaleh": 10.4, "Simmons":  0.0, "Amiwala":  0.0, "Andrew":  3.0, "Huynh": 6.0},
                "asian":            {"Fine": 20.8, "Biss": 15.3, "Abughazaleh": 51.4, "Simmons":  0.0, "Amiwala":  5.6, "Andrew":  6.9, "Huynh": 0.0},
                "very_liberal":     {"Fine": 11.4, "Biss": 22.8, "Abughazaleh": 38.0, "Simmons": 13.9, "Amiwala":  8.9, "Andrew":  2.5, "Huynh": 2.5},
                "somewhat_liberal": {"Fine": 19.0, "Biss": 36.7, "Abughazaleh": 21.5, "Simmons":  8.9, "Amiwala":  5.1, "Andrew":  6.3, "Huynh": 2.5},
                "moderate":         {"Fine": 29.5, "Biss": 16.4, "Abughazaleh": 29.5, "Simmons":  6.6, "Amiwala":  6.6, "Andrew":  9.8, "Huynh": 1.6},
            },
            "favorability": {
                "Fine":        {"favorable": 36.0, "unfavorable": 35.0},
                "Biss":        {"favorable": 51.0, "unfavorable": 23.0},
                "Abughazaleh": {"favorable": 35.0, "unfavorable": 27.0},
                "Simmons":     {"favorable": 28.0, "unfavorable":  8.0},
                "Amiwala":     {"favorable": 28.0, "unfavorable":  8.0},
                "Andrew":      {"favorable": 20.0, "unfavorable": 12.0},
                "Huynh":       {"favorable": 23.0, "unfavorable":  9.0},
            },
            "second_choice": {
                "Fine":        {"Biss": 52.6, "Amiwala": 17.5, "Andrew": 12.4, "Abughazaleh":  9.3, "Huynh":  5.2, "Simmons":  3.1},
                "Biss":        {"Fine": 35.2, "Amiwala": 19.3, "Abughazaleh": 15.9, "Simmons": 12.5, "Huynh": 11.4, "Andrew":   5.7},
                "Abughazaleh": {"Amiwala": 37.6, "Simmons": 19.3, "Huynh": 16.5, "Biss": 12.8, "Andrew": 11.9, "Fine":  1.8},
                "Simmons":     {"Abughazaleh": 55.6, "Huynh": 17.8, "Biss":  8.9, "Fine":  6.7, "Amiwala":  5.6, "Andrew":  5.6},
                "Amiwala":     {"Abughazaleh": 50.0, "Biss": 17.7, "Fine": 17.7, "Simmons": 11.5, "Huynh":  3.1, "Andrew":  0.0},
                "Andrew":      {"Fine": 39.7, "Biss": 28.2, "Abughazaleh": 16.7, "Amiwala":  6.4, "Huynh":  6.4, "Simmons":  2.6},
                "Huynh":       {"Abughazaleh": 42.4, "Simmons": 21.2, "Biss": 19.7, "Fine": 10.6, "Amiwala":  4.5, "Andrew":  1.5},
            },
        },

        # ── Biss Campaign Internal  (2026-02-11, n=500) ───────────────────
        {
            "pollster_id":      "biss_internal_feb2026",
            "pollster_name":    "Biss Campaign Internal (Feb)",
            "pollster_quality": 0.80,
            "field_end":        "2026-02-11",
            "sample_size":      500,
            "moe":              4.4,
            "is_internal":      True,
            "house_effect":     {"Biss": 2.0},
            "topline": {
                "Fine":        20.5,
                "Biss":        35.2,
                "Abughazaleh": 20.5,
                "Simmons":      8.0,
                "Amiwala":      4.5,
                "Andrew":       8.0,
                "Huynh":        3.4,
            },
            "crosstabs": None, "demographic_crosstabs": None,
            "favorability": None, "second_choice": None,
        },

        # ── Fine Campaign Internal  (2026-02-01, n=500) ───────────────────
        {
            "pollster_id":      "fine_internal_feb2026",
            "pollster_name":    "Fine Campaign Internal (Feb)",
            "pollster_quality": 0.80,
            "field_end":        "2026-02-01",
            "sample_size":      500,
            "moe":              4.4,
            "is_internal":      True,
            "house_effect":     {"Fine": 2.0},
            "topline": {
                "Fine":        28.8,
                "Biss":        28.8,
                "Abughazaleh": 19.2,
                "Simmons":      9.6,
                "Amiwala":      5.5,
                "Andrew":       5.5,
                "Huynh":        2.7,
            },
            "crosstabs": None, "demographic_crosstabs": None,
            "favorability": None, "second_choice": None,
        },

        # ── Fine Campaign Internal  (2025-11-01, n=600) ───────────────────
        {
            "pollster_id":      "fine_internal_nov2025",
            "pollster_name":    "Fine Campaign Internal (Nov)",
            "pollster_quality": 0.80,
            "field_end":        "2025-11-01",
            "sample_size":      600,
            "moe":              3.4,
            "is_internal":      True,
            "house_effect":     {"Fine": 2.0},
            "topline": {
                "Fine":        19.4,
                "Biss":        29.9,
                "Abughazaleh": 20.9,
                "Simmons":     14.9,
                "Amiwala":      7.5,
                "Andrew":       1.5,
                "Huynh":        6.0,
            },
            "crosstabs": None, "demographic_crosstabs": None,
            "favorability": None, "second_choice": None,
        },

        # ── Data for Progress  (2025-10-26, n=569) ───────────────────────
        # Not an internal poll. Andrew not yet in race.
        # Topline from results (Fine=10,Biss=18,Abu=18,Sim=6,Ami=6,Huynh=5) ÷ 63; undecided=31%.
        {
            "pollster_id":      "dfp_oct2025",
            "pollster_name":    "Data for Progress",
            "pollster_quality": 0.60,
            "field_end":        "2025-10-26",
            "sample_size":      569,
            "moe":              4.4,
            "is_internal":      False,
            "topline": {
                "Fine":        15.9,
                "Biss":        28.6,
                "Abughazaleh": 28.6,
                "Simmons":      9.5,
                "Amiwala":      9.5,
                "Huynh":        7.9,
                # Andrew not included — not yet in race at time of poll
            },
            "crosstabs": None,
            # Ideology + race crosstabs from DFP cross-tab table.
            # white sum=63; very_liberal=78, somewhat_liberal=66, moderate=48.
            "demographic_crosstabs": {
                "white":            {"Fine": 15.9, "Biss": 31.7, "Abughazaleh": 30.2, "Simmons": 11.1, "Amiwala":  6.3, "Huynh":  4.8},
                "very_liberal":     {"Fine":  9.0, "Biss": 24.4, "Abughazaleh": 37.2, "Simmons": 11.5, "Amiwala": 10.3, "Huynh":  7.7},
                "somewhat_liberal": {"Fine": 13.6, "Biss": 37.9, "Abughazaleh": 24.2, "Simmons": 10.6, "Amiwala":  6.1, "Huynh":  7.6},
                "moderate":         {"Fine": 29.2, "Biss": 29.2, "Abughazaleh": 16.7, "Simmons":  6.3, "Amiwala": 12.5, "Huynh":  6.3},
            },
            "favorability": None, "second_choice": None,
        },

        # ── Biss Campaign Internal  (2025-10-25, n=500) ──────────────────
        {
            "pollster_id":      "biss_internal_oct2025",
            "pollster_name":    "Biss Campaign Internal (Oct)",
            "pollster_quality": 0.80,
            "field_end":        "2025-10-25",
            "sample_size":      500,
            "moe":              4.4,
            "is_internal":      True,
            "house_effect":     {"Biss": 2.0},
            "topline": {
                "Fine":        13.5,
                "Biss":        41.9,
                "Abughazaleh": 23.0,
                "Simmons":      8.1,
                "Amiwala":      4.1,
                "Andrew":       4.1,
                "Huynh":        5.4,
            },
            "crosstabs": None, "demographic_crosstabs": None,
            "favorability": None, "second_choice": None,
        },

        # ── MDW (Abughazaleh Internal)  (2025-10-20, n=917) ──────────────
        {
            "pollster_id":      "mdw_oct2025",
            "pollster_name":    "MDW / Abughazaleh Campaign Internal",
            "pollster_quality": 0.60,
            "field_end":        "2025-10-20",
            "sample_size":      917,
            "moe":              3.4,
            "is_internal":      True,
            "house_effect":     {"Abughazaleh": 2.0},
            "topline": {
                "Fine":        17.6,
                "Biss":        35.3,
                "Abughazaleh": 25.5,
                "Simmons":      7.8,
                "Amiwala":      3.9,
                "Andrew":       3.9,
                "Huynh":        5.9,
            },
            "crosstabs": None, "demographic_crosstabs": None,
            "favorability": None, "second_choice": None,
        },
    ],

    # ── Undecided allocation & modeling constants ────────────────────────────
    # Base weights for distributing undecided voters (before favorability blend).
    # Higher = gets more of the undecided pool.
    undecided_allocation={
        "Fine":        1.0,
        "Biss":        1.1,  # slight incumbency-familiarity boost
        "Abughazaleh": 0.95,
        "Simmons":     0.90,
        "Amiwala":     0.85,
        "Andrew":      0.80,
        "Huynh":       0.75,
    },
    favorability_blend=0.25,        # 25% of undecided weight from favorability aware_rate
    second_choice_strength=0.60,    # 60% of downward deviation routed via second-choice matrix

    # ── Simulation parameters ────────────────────────────────────────────────
    n_sim_district=1_000_000,
    n_sim_precinct=50_000,
    moe_district=4.4,
    moe_precinct=6.0,

    # ── Geography (election authorities) ────────────────────────────────────
    # IL-09 spans four election authorities, each of which reports results
    # independently: Chicago BOE, Cook County Clerk (suburban Cook),
    # Lake County Clerk, McHenry County Clerk.
    region_type="election_authority",
    regions=["Chicago", "Suburban Cook", "Lake County", "McHenry County"],

    data_dir=_DATA_DIR,
    output_dir=_OUTPUT_DIR,

    shapefile_precinct=Path("shapefile/IL24/IL24.shp"),
    shapefile_district=Path("shapefile/congressional_districts.shp"),
    shapefile_crosstab=Path("shapefile/state_senate/1772312996199_Senate Plan.shp"),

    joinfield_format="IL09",

    # Early vote modeling is disabled for IL-09. Election authorities report
    # cumulative early/mail totals by county but give no demographic breakdown,
    # so candidate allocation is unreliable before results are certified.
    banked_vote_mode="none",

    votes_csv_region_map={
        "City of Chicago":       "Chicago",
        "Suburban Cook County":  "Suburban Cook",
        "Lake County":           "Lake County",
        "McHenry County":        "McHenry County",
    },

    # ── Race-specific constraints ─────────────────────────────────────────────
    # Biss has incumbency saturation in Evanston; he draws fewer undecideds there.
    extra_constraints={
        "biss_evanston_undecided_penalty": 0.35,
    },
)
