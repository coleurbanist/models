"""
Temporary one-off: feed the 2025 poll directly into the district simulator
to see what outputs look like.  Not wired into the main pipeline.

Poll metadata:
  Pollster:     Unknown (2025)
  Field end:    2025 (exact date unknown)
  Sample size:  697
  MOE:          ±3.7pp
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import simulator_runner

# ── Candidates ─────────────────────────────────────────────────────────────
CANDIDATES = [
    "Vallas", "Giannoulias", "Mendoza", "Johnson", "Buckner",
    "Wilson", "Conway", "Gutierrez", "Beale", "Green", "Villegas",
]

# ── Poll metadata ───────────────────────────────────────────────────────────
SAMPLE_SIZE = 697
MOE         = 3.7  # ±pp, as reported

# ── Topline (weighted figures from poll) ────────────────────────────────────
TOPLINE = {
    "Vallas":      27.4,
    "Giannoulias": 21.0,
    "Mendoza":     11.7,
    "Johnson":      8.2,
    "Buckner":      6.3,
    "Wilson":       5.9,
    "Conway":       5.7,
    "Gutierrez":    5.1,
    "Beale":        3.9,
    "Green":        2.9,
    "Villegas":     2.0,
}

# ── Crosstabs ───────────────────────────────────────────────────────────────
# Stored here for future precinct pipeline use.
# Keys match the canonical group names used in demographic_crosstabs.
# Values are percentage points (0–100), None = not reported.

CROSSTABS_IDEOLOGY = {
    "very_conservative": {
        "Vallas": 59, "Giannoulias": 3, "Mendoza": 3, "Johnson": 2,
        "Buckner": 0, "Wilson": 27, "Conway": 1, "Gutierrez": 0,
        "Beale": 1, "Green": 4, "Villegas": 0,
    },
    "somewhat_conservative": {
        "Vallas": 46, "Giannoulias": 10, "Mendoza": 10, "Johnson": 10,
        "Buckner": 1, "Wilson": 8, "Conway": 6, "Gutierrez": 5,
        "Beale": 1, "Green": 2, "Villegas": 0,
    },
    "moderate": {
        "Vallas": 32, "Giannoulias": 19, "Mendoza": 12, "Johnson": 4,
        "Buckner": 2, "Wilson": 8, "Conway": 7, "Gutierrez": 4,
        "Beale": 6, "Green": 2, "Villegas": 4,
    },
    "somewhat_liberal": {
        "Vallas": 22, "Giannoulias": 29, "Mendoza": 14, "Johnson": 9,
        "Buckner": 8, "Wilson": 1, "Conway": 8, "Gutierrez": 2,
        "Beale": 5, "Green": 1, "Villegas": 2,
    },
    "very_liberal": {
        "Vallas": 5, "Giannoulias": 27, "Mendoza": 12, "Johnson": 17,
        "Buckner": 17, "Wilson": 1, "Conway": 1, "Gutierrez": 13,
        "Beale": 0, "Green": 7, "Villegas": 1,
    },
}

CROSSTABS_RACE = {
    "black": {
        "Vallas": 22, "Giannoulias": 18, "Mendoza": 6, "Johnson": 18,
        "Buckner": 4, "Wilson": 13, "Conway": 1, "Gutierrez": 5,
        "Beale": 7, "Green": 4, "Villegas": 3,
    },
    "hispanic": {
        "Vallas": 26, "Giannoulias": 14, "Mendoza": 9, "Johnson": 2,
        "Buckner": 4, "Wilson": 3, "Conway": 15, "Gutierrez": 16,
        "Beale": 8, "Green": 0, "Villegas": 2,
    },
    "white": {
        "Vallas": 31, "Giannoulias": 24, "Mendoza": 17, "Johnson": 6,
        "Buckner": 7, "Wilson": 3, "Conway": 5, "Gutierrez": 1,
        "Beale": 1, "Green": 2, "Villegas": 2,
    },
}

CROSSTABS_VOTE_2024 = {
    "trump": {
        "Vallas": 57, "Giannoulias": 4, "Mendoza": 11, "Johnson": 2,
        "Buckner": 1, "Wilson": 10, "Conway": 7, "Gutierrez": 1,
        "Beale": 4, "Green": 2, "Villegas": 1,
    },
    "harris": {
        "Vallas": 18, "Giannoulias": 26, "Mendoza": 12, "Johnson": 10,
        "Buckner": 7, "Wilson": 4, "Conway": 6, "Gutierrez": 7,
        "Beale": 3, "Green": 3, "Villegas": 2,
    },
}

CROSSTABS_AGE = {
    "18_30": {
        "Vallas": 15, "Giannoulias": 33, "Mendoza": 11, "Johnson": 3,
        "Buckner": 22, "Wilson": 0, "Conway": 0, "Gutierrez": 0,
        "Beale": 8, "Green": 8, "Villegas": 0,
    },
    "31_45": {
        "Vallas": 26, "Giannoulias": 16, "Mendoza": 13, "Johnson": 10,
        "Buckner": 11, "Wilson": 2, "Conway": 11, "Gutierrez": 4,
        "Beale": 1, "Green": 5, "Villegas": 2,
    },
    "46_64": {
        "Vallas": 30, "Giannoulias": 20, "Mendoza": 9, "Johnson": 7,
        "Buckner": 5, "Wilson": 10, "Conway": 5, "Gutierrez": 6,
        "Beale": 3, "Green": 3, "Villegas": 3,
    },
    "65plus": {
        "Vallas": 29, "Giannoulias": 23, "Mendoza": 14, "Johnson": 9,
        "Buckner": 1, "Wilson": 5, "Conway": 4, "Gutierrez": 6,
        "Beale": 6, "Green": 0, "Villegas": 2,
    },
}

CROSSTABS_GENDER = {
    "female": {
        "Vallas": 25, "Giannoulias": 20, "Mendoza": 12, "Johnson": 9,
        "Buckner": 5, "Wilson": 6, "Conway": 7, "Gutierrez": 6,
        "Beale": 4, "Green": 4, "Villegas": 3,
    },
    "male": {
        "Vallas": 31, "Giannoulias": 23, "Mendoza": 12, "Johnson": 7,
        "Buckner": 6, "Wilson": 6, "Conway": 4, "Gutierrez": 4,
        "Beale": 4, "Green": 2, "Villegas": 2,
    },
}

# ── Favorability ────────────────────────────────────────────────────────────
FAVORABILITY = {
    "Giannoulias": {"favorable": 48.6, "unfavorable": 14.3},
    "Mendoza":     {"favorable": 38.0, "unfavorable": 14.6},
    "Conway":      {"favorable": 14.6, "unfavorable": 12.0},
    "Buckner":     {"favorable": 17.6, "unfavorable": 15.4},
    "Vallas":      {"favorable": 40.6, "unfavorable": 34.2},
    "Johnson":     {"favorable":  6.6, "unfavorable": 79.9},
}

# ── Ideological blocs (from coalition analysis) ─────────────────────────────
BLOCS = [
    ["Johnson", "Buckner", "Green", "Gutierrez"],   # progressive
    ["Giannoulias", "Mendoza", "Conway"],            # center-left
    ["Vallas", "Wilson", "Beale", "Villegas"],       # conservative/right
]


def _print_results(result: dict, label: str) -> None:
    advance  = result["advance_probs"]      or {}
    outright = result["outright_win_probs"] or {}
    means    = result["mean_vote_shares"]
    p05      = result["p05_vote_shares"]
    p95      = result["p95_vote_shares"]

    sorted_cands = sorted(CANDIDATES, key=lambda c: advance.get(c, 0.0), reverse=True)

    print(f"  {label}")
    print(f"  {'Candidate':<14} {'Top-2':>7} {'Outright':>9} {'Exp %':>7} {'p05':>6} {'p95':>6}")
    print(f"  {'-'*14} {'-'*7} {'-'*9} {'-'*7} {'-'*6} {'-'*6}")
    for cand in sorted_cands:
        print(
            f"  {cand:<14}"
            f"  {advance.get(cand, 0.0):>6.1%}"
            f"  {outright.get(cand, 0.0):>8.1%}"
            f"  {means[cand]:>6.1%}"
            f"  {p05[cand]:>5.1%}"
            f"  {p95[cand]:>5.1%}"
        )

    print(f"\n  P(outright win / no runoff): {result['prob_no_runoff']:.1%}")
    print(f"  P(goes to runoff):           {1 - result['prob_no_runoff']:.1%}")

    print()
    print("  Runoff conditional probabilities (given a runoff happens):")
    runoff_cond = result["runoff_probs"] or {}
    for cand in sorted_cands:
        rp = runoff_cond.get(cand, 0.0)
        if rp > 0.01:
            print(f"    {cand:<14}  {rp:.1%}")


def _scale_crosstabs(
    crosstabs: dict,
    old_topline: dict[str, float],
    new_topline: dict[str, float],
    candidates: list[str],
) -> dict:
    """
    Adjust crosstabs to be consistent with a new topline.

    For each group, the per-candidate delta (old_crosstab_pct - old_topline_pct)
    is preserved and applied to the new topline.  Values are clipped to [0, 100]
    then renormalized so each group sums to 100%.
    """
    result = {}
    for group, shares in crosstabs.items():
        new_group: dict[str, float] = {}
        for cand in candidates:
            delta = (shares.get(cand) or 0.0) - old_topline.get(cand, 0.0)
            new_group[cand] = max(0.0, new_topline[cand] + delta)
        total = sum(new_group.values())
        if total > 1e-9:
            new_group = {c: v * 100.0 / total for c, v in new_group.items()}
        result[group] = new_group
    return result


def main() -> None:
    print("Chicago Mayor 2027 — 2025 poll snapshot")
    print(f"  n={SAMPLE_SIZE}, MOE=±{MOE}pp, 11 candidates")
    print()

    # ── Scenario A: as-polled (no undecideds) ───────────────────────────────
    baseline_a = {c: v / 100.0 for c, v in TOPLINE.items()}

    result_a = simulator_runner.run_district_sim(
        n_simulations=1_000_000,
        candidates=CANDIDATES,
        baseline=baseline_a,
        moe_district=MOE,
        ideological_blocs=BLOCS,
        second_choice_matrix={},
        second_choice_strength=0.0,
        has_runoff=True,
        runoff_threshold=0.50,
        fundamental_uncertainty_sigma=6.0,
        sigma_within_bloc_fraction=0.5,
        environment_shock_fraction=0.3,
    )

    print("─" * 60)
    _print_results(result_a, "Scenario A — as polled (0% undecided)")
    print()

    # ── Scenario B: ~30% undecided (scale toplines ×0.70) ───────────────────
    scale = 0.70
    baseline_b = {c: v / 100.0 * scale for c, v in TOPLINE.items()}
    undecided_pct = 100.0 * (1.0 - sum(baseline_b.values()))  # ≈ 30pp

    # Build undecided weights from favorability aware_rates; fall back to
    # topline-proportional weight for candidates without favorability data.
    raw_fav: dict[str, float] = {}
    for cand, fdata in FAVORABILITY.items():
        f = fdata["favorable"]
        u = fdata["unfavorable"]
        denom = f + u
        raw_fav[cand] = f / denom if denom > 0 else 0.5

    mean_fav = sum(raw_fav.values()) / len(raw_fav) if raw_fav else 0.5
    undecided_weights: dict[str, float] = {}
    for cand in CANDIDATES:
        if cand in raw_fav:
            undecided_weights[cand] = raw_fav[cand]
        else:
            # No favorability data — use topline share as proxy
            undecided_weights[cand] = TOPLINE[cand] / 100.0 * mean_fav / (sum(TOPLINE.values()) / 100.0 / len(CANDIDATES))

    result_b = simulator_runner.run_district_sim(
        n_simulations=1_000_000,
        candidates=CANDIDATES,
        baseline=baseline_b,
        moe_district=MOE,
        ideological_blocs=BLOCS,
        second_choice_matrix={},
        second_choice_strength=0.0,
        has_runoff=True,
        runoff_threshold=0.50,
        undecided_share=undecided_pct,
        undecided_weights=undecided_weights,
        undecided_concentration=3.0,
        fundamental_uncertainty_sigma=6.0,
        sigma_within_bloc_fraction=0.5,
        environment_shock_fraction=0.3,
    )

    print("─" * 60)
    _print_results(result_b, f"Scenario B — ~{undecided_pct:.0f}% undecided (toplines ×{scale}), Dirichlet allocation")
    print()

    print("  Crosstab dimensions stored (for precinct pipeline):")
    print(f"    ideology: {list(CROSSTABS_IDEOLOGY.keys())}")
    print(f"    race:     {list(CROSSTABS_RACE.keys())}")
    print(f"    vote2024: {list(CROSSTABS_VOTE_2024.keys())}")
    print(f"    age:      {list(CROSSTABS_AGE.keys())}")
    print(f"    gender:   {list(CROSSTABS_GENDER.keys())}")

    # ── Scenario C: hypothetical dead heat (top 5 at 15% each) ─────────────
    print()
    print("─" * 60)
    TOP_5  = ["Vallas", "Giannoulias", "Mendoza", "Johnson", "Buckner"]
    OTHERS = [c for c in CANDIDATES if c not in TOP_5]

    others_old_sum = sum(TOPLINE[c] for c in OTHERS)
    others_remaining = 100.0 - 5 * 15.0  # 25pp left for the field
    topline_c = {c: 15.0 for c in TOP_5}
    for c in OTHERS:
        topline_c[c] = TOPLINE[c] / others_old_sum * others_remaining

    # Scale all crosstab dimensions using the delta method
    ct_c_ideology = _scale_crosstabs(CROSSTABS_IDEOLOGY, TOPLINE, topline_c, CANDIDATES)
    ct_c_race     = _scale_crosstabs(CROSSTABS_RACE,     TOPLINE, topline_c, CANDIDATES)
    ct_c_age      = _scale_crosstabs(CROSSTABS_AGE,      TOPLINE, topline_c, CANDIDATES)
    ct_c_gender   = _scale_crosstabs(CROSSTABS_GENDER,   TOPLINE, topline_c, CANDIDATES)

    print("  Scenario C — hypothetical dead heat (top 5 at 15%, others scaled)")
    print()
    print(f"  {'Candidate':<14}  {'New %':>7}  {'Old %':>7}")
    for c in CANDIDATES:
        print(f"  {c:<14}  {topline_c[c]:>6.1f}%  {TOPLINE[c]:>6.1f}%")
    print()

    # Ideology breakdown for top 5
    ideo_order = ["very_conservative", "somewhat_conservative", "moderate",
                  "somewhat_liberal", "very_liberal"]
    ideo_labels = ["V.Con", "S.Con", "Mod", "S.Lib", "V.Lib"]
    print(f"  Scaled ideology crosstabs (top 5):")
    print(f"  {'Candidate':<14}  " + "  ".join(f"{l:>6}" for l in ideo_labels))
    for c in TOP_5:
        vals = "  ".join(f"{ct_c_ideology[g][c]:>5.1f}%" for g in ideo_order)
        print(f"  {c:<14}  {vals}")
    print()

    baseline_c = {c: v / 100.0 for c, v in topline_c.items()}
    result_c = simulator_runner.run_district_sim(
        n_simulations=1_000_000,
        candidates=CANDIDATES,
        baseline=baseline_c,
        moe_district=MOE,
        ideological_blocs=BLOCS,
        second_choice_matrix={},
        second_choice_strength=0.0,
        has_runoff=True,
        runoff_threshold=0.50,
        fundamental_uncertainty_sigma=6.0,
        sigma_within_bloc_fraction=0.5,
        environment_shock_fraction=0.3,
    )
    _print_results(result_c, "Scenario C — simulation")


if __name__ == "__main__":
    main()
