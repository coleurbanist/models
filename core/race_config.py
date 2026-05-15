from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class RaceConfig:
    """
    All parameters that define a specific election race.
    Every new race provides one instance of this; the core pipeline
    functions accept it instead of reading module-level globals.

    The pipeline outputs JSON files consumed by the separate display project
    (ilforecast_redesign). Models never renders HTML directly.
    """

    # ------------------------------------------------------------------ identity
    race_id: str        # slug used in filenames, e.g. "il09_2026"
    race_label: str     # human label, e.g. "IL-09 Democratic Primary"
    election_date: str  # ISO date string "YYYY-MM-DD"

    # ------------------------------------------------------------------ candidates
    candidates: list[str]               # short display names in desired display order
    colors: dict[str, str]              # {"CandName": "#rrggbb"} — consumed by display project
    ideological_blocs: list[list[str]]  # candidates that share correlated district-level noise

    # ------------------------------------------------------------------ polls & modeling
    polls: list[dict[str, Any]]           # poll definition dicts (see poll_weighting.py)
    undecided_allocation: dict[str, float]  # {candidate: weight} for distributing undecideds
    favorability_blend: float = 0.25        # fraction of undecided weight from favorability
    second_choice_strength: float = 0.60   # fraction of downward deviation routed via SC matrix

    # Discount applied to the commissioning candidate's topline in internal polls.
    # Campaigns only release internal polls when the numbers flatter them, so the
    # candidate who paid for it is likely overstated. Set commissioned_by in the
    # poll dict to activate. 0.0 disables the discount entirely.
    internal_candidate_discount: float = 2.0

    # Viability scaling for undecided allocation.
    # Each candidate's undecided weight is multiplied by baseline_share^alpha before
    # the Dirichlet draw, concentrating undecideds toward polling frontrunners.
    #
    # Historical pattern: undecided voters break for higher-polling candidates (name
    # recognition, perceived viability). IL-09 2026 confirmed this — undecideds went
    # largely to the top 3, not the tail.
    #
    # This only affects the undecided pool, NOT the noise-driven variance that causes
    # surges (fundamental_uncertainty_sigma, bloc shocks). Trailing candidates can still
    # spike above their baseline via noise; they just don't also clean up among undecideds.
    #
    #   0.0 = no viability effect (weights purely from undecided_allocation + favorability)
    #   0.5 = mild frontrunner tilt
    #   1.0 = weight scales linearly with polling share (recommended default)
    #   2.0 = strong concentration toward frontrunners
    undecided_viability_alpha: float = 1.0

    # ------------------------------------------------------------------ simulation
    n_sim_district: int = 1_000_000  # district-level Monte Carlo trials
    n_sim_precinct: int = 50_000     # precinct-level Monte Carlo trials
    moe_district: float = 4.4        # district-level margin of error (percentage points)
    moe_precinct: float = 6.0        # per-precinct noise (percentage points)

    # Independent per-candidate uncertainty beyond sampling error: late momentum
    # shifts, turnout composition errors, structural polling bias per candidate.
    # Historical Illinois primary per-candidate RMSE ≈ 5–6pp; default is 6.0.
    fundamental_uncertainty_sigma: float = 6.0

    # Within-bloc competition: zero-sum redistribution between candidates sharing
    # a bloc.  Expressed as a fraction of sigma_district (moe/2).
    # 0.5 = within-bloc competition is half as large as the district noise.
    sigma_within_bloc_fraction: float = 0.5

    # Systematic shock applied equally to all candidates (turnout composition
    # errors, e.g. sample skewed younger/more progressive than actual electorate).
    # Expressed as a fraction of sigma_district.
    environment_shock_fraction: float = 0.3

    # ------------------------------------------------------------------ geography / regions
    #
    # "region_type" controls what the entries in `regions` represent:
    #
    #   "election_authority" — each region is one election-reporting authority
    #       (a county clerk or one of the 6 independent city election commissions).
    #       Use this for congressional, state legislative, and statewide races.
    #       IL-09 example: ["Chicago", "Suburban Cook", "Lake County", "McHenry County"]
    #       The 6 independent Illinois city commissions (report separately from their county):
    #         City of Chicago, City of Rockford, City of Bloomington,
    #         City of Galesburg, City of Danville, City of East St. Louis
    #
    #   "ward" — each region is one Chicago ward (50 total).
    #       Use this for Chicago mayoral and aldermanic races only.
    #
    region_type: Literal["election_authority", "ward"] = "election_authority"
    regions: list[str] = field(default_factory=list)

    data_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)

    # Paths relative to data_dir
    shapefile_precinct: Path = Path("shapefile/IL24/IL24.shp")
    shapefile_district: Path = Path("shapefile/congressional_districts.shp")
    # Crosstab shapefile: senate districts for IL-09, ward boundaries for Chicago mayor
    shapefile_crosstab: Path | None = None

    # Drives JoinField parsing and election-authority assignment.
    # Each JoinField prefix is an election authority:
    #   "CITY OF CHICAGO:", "COOK:", "LAKE:", "MCHENRY:", "CITY OF ROCKFORD:", etc.
    # For Chicago ward races, precincts are keyed "WARD XX PRECINCT YY" instead.
    joinfield_format: Literal["IL09", "CHICAGO"] = "IL09"

    # ------------------------------------------------------------------ early / mail votes
    #
    # "banked_vote_mode" controls whether and how early/mail votes are modeled:
    #
    #   "none"    — no early vote modeling (default for non-Chicago races).
    #       For IL-09 and most races, election authority totals are available but
    #       demographic composition is unknown until results are certified, so
    #       candidate allocation is unreliable.
    #
    #   "chicago" — use Chicago BOE's election-day demographic release (age + gender)
    #       to distribute early votes via demographic crosstabs.
    #       The city posts a precinct-level age/gender breakdown roughly 30 min
    #       after polls close. This is what makes early vote modeling tractable
    #       for Chicago mayor but not for other races.
    #
    banked_vote_mode: Literal["none", "chicago"] = "none"

    # ------------------------------------------------------------------ runoff (mayoral races)
    has_runoff: bool = False
    runoff_threshold: float = 0.50  # majority fraction needed to win outright

    # ------------------------------------------------------------------ race-specific modeling pins
    # Arbitrary constraints consumed by precinct_pipeline.py step 3 (undecided allocation).
    # Example: {"biss_evanston_undecided_penalty": 0.35}
    extra_constraints: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ pollster ratings
    # Path to pollster_ratings.json produced by pollster_calibration.py.
    # If None, poll_weighting.py uses the pollster_quality scores in polls[].
    pollster_ratings_path: Path | None = None

    # ------------------------------------------------------------------ demographic calibration
    # race_context: key used to look up precinct_progressive_scores in the DB.
    #   e.g. "chicago_mayor", "il09". If None, ideology dimension is skipped.
    # prog_score_scenario: which scenario row to use (default "generic").
    # demo_year: ACS year for precinct_demographics (default 2022).
    race_context: str | None = None
    prog_score_scenario: str = "generic"
    demo_year: int = 2022

    # ------------------------------------------------------------------ region label mapping
    # Maps election-authority strings from results CSV to canonical region names.
    # e.g. {"City of Chicago": "Chicago", "Suburban Cook County": "Suburban Cook"}
    votes_csv_region_map: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def data_path(self, *parts: str | Path) -> Path:
        """Resolve a path relative to data_dir, enforcing pathlib throughout."""
        return self.data_dir.joinpath(*parts)

    def output_path(self, *parts: str | Path) -> Path:
        return self.output_dir.joinpath(*parts)

    def precinct_shapefile(self) -> Path:
        return self.data_path(self.shapefile_precinct)

    def district_shapefile(self) -> Path:
        return self.data_path(self.shapefile_district)

    def crosstab_shapefile(self) -> Path | None:
        if self.shapefile_crosstab is None:
            return None
        return self.data_path(self.shapefile_crosstab)
