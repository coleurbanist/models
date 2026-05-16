"""
Poll aggregation, weighting, and versioned history.

Each poll in config.polls is a dict with these keys:
    pollster_id      str
    pollster_name    str
    pollster_quality float           (0–1; overridden by pollster_ratings.json if present)
    field_end        str             YYYY-MM-DD
    sample_size      int
    moe              float           margin of error (percentage points)
    is_internal      bool
    commissioned_by  str | None      candidate name; triggers internal discount when is_internal=True
    topline             dict[str, float]  {candidate: pct}  (values sum to ≤100; undecided = 100 - sum)
    crosstabs           dict | None     {senate_district/ward: {candidate: pct}} or None
    favorability        dict | None     {candidate: {"favorable": x, "unfavorable": y}} or None
    second_choice       dict | None     {from_candidate: {to_candidate: pct}} or None
    sample_composition  dict | None     {group: pct_of_sample}  same keys as demographic_crosstabs
                                        (e.g. {"white": 48, "black": 30, "female": 52, "age_18_30": 15})
                                        Values may be 0–100 (pct) or 0–1 (fraction); either is accepted.
                                        Used to compute subsample n for each crosstab group, which
                                        down-weights unreliable subsamples via sqrt(n_sub / N) scaling.

Internal poll handling
----------------------
is_internal=True halves the poll's composite weight (campaigns only release
polls when the numbers help them, so internal polls carry selection bias).

commissioned_by + is_internal=True triggers an additional automatic discount:
config.internal_candidate_discount pp is subtracted from the commissioning
candidate's topline. This is applied via the house_effect mechanism before
aggregation so it interacts correctly with any other house effect adjustments.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from .race_config import RaceConfig
from . import simulator_runner


# ── Poll identity ─────────────────────────────────────────────────────────

def get_poll_id(poll: dict) -> str:
    """
    Unique identifier for a single poll. Used by --poll-id to target one poll.

    If the poll dict has an explicit "poll_id" key, that is returned as-is.
    Otherwise falls back to "{pollster_id}_{field_end}" (e.g. "m3_2027-01-15"),
    which is unique as long as a pollster doesn't field two polls on the same day.

    "pollster_id" is kept separate and is used only for pollster DB lookup.
    You should never need to add a wave suffix to pollster_id — just use the
    base pollster key (e.g. "m3") for every poll from that firm.
    """
    return poll.get("poll_id") or f"{poll['pollster_id']}_{poll['field_end']}"


# Default scale factor applied to demographic crosstab groups when the poll
# does not report sample_composition for that group.  Equivalent to assuming
# the subgroup is ~25% of the full sample (sqrt(0.25) = 0.5).  This ensures
# polls that withhold composition data are not rewarded with higher crosstab
# weight than polls that report it.
_CROSSTAB_DEFAULT_SCALE = 0.5

# ── Weighting helpers ──────────────────────────────────────────────────────

def _recency_weight(field_end: str, as_of: str | None = None) -> float:
    ref = date.fromisoformat(as_of) if as_of else date.today()
    end = date.fromisoformat(field_end)
    days = (ref - end).days
    if days < 0:
        return 0.0  # future poll (shouldn't appear, but guard)
    if days <= 7:
        return 1.0
    # 14-day half-life after the 7-day flat window
    return 0.5 ** ((days - 7) / 14.0)


def _moe_weight(moe: float) -> float:
    return 100.0 / max(moe, 0.1)


def _composite_weight(
    poll: dict,
    as_of: str | None = None,
    election_day: str | None = None,
    late_poll_multiplier: float = 1.0,
) -> float:
    q = poll.get("pollster_quality", 0.7)
    r = _recency_weight(poll["field_end"], as_of)
    m = _moe_weight(poll["moe"])
    internal = 0.5 if poll.get("is_internal", False) else 1.0
    w = q * r * m * internal
    if late_poll_multiplier != 1.0 and election_day is not None:
        days_before = (date.fromisoformat(election_day) - date.fromisoformat(poll["field_end"])).days
        if 0 <= days_before <= 7:
            w *= late_poll_multiplier
    return w


# ── Pollster ratings override ──────────────────────────────────────────────

def _load_pollster_ratings(path: Path) -> dict[str, dict]:
    """
    Load a pollster ratings file. Handles two formats:

    Rich format (pollster_db.json) — entries have "silver_grade" and/or
    "votehub_pct_within_moe". Quality is derived automatically from whichever
    agencies have rated the pollster; both scores are averaged when available.

    Simple format (pollster_ratings.json) — entries have an explicit "quality"
    float. Kept for backward compatibility.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    # Detect rich format: any entry has agency-specific keys
    is_rich = any(
        "silver_grade" in entry or "votehub_pct_within_moe" in entry
        for entry in raw.values()
    )
    if is_rich:
        from .pollster_ratings import load_pollster_db, to_poll_weighting_format
        return to_poll_weighting_format(load_pollster_db(path))

    return raw


def _apply_pollster_ratings(polls: list[dict], ratings: dict) -> list[dict]:
    result = []
    for poll in polls:
        p = deepcopy(poll)
        pid = p.get("pollster_id", "")
        if pid in ratings:
            p["pollster_quality"] = ratings[pid].get("quality", p.get("pollster_quality", 0.7))
            # Merge pollster-level house effect into any existing poll-level adjustment.
            # All three sources (manual poll, pollster rating, internal discount) are
            # additive — a pollster's systematic bias applies on top of poll-specific ones.
            if "house_effect_adjustment" in ratings[pid]:
                he = dict(p.get("house_effect") or {})
                for cand, adj in ratings[pid]["house_effect_adjustment"].items():
                    he[cand] = he.get(cand, 0.0) + adj
                p["house_effect"] = he
        result.append(p)
    return result


def _apply_lean_adjustments(
    ratings: dict[str, dict],
    candidates: list[str],
    ideological_blocs: list[list[str]],
    bloc_positions: list[float],
) -> dict[str, dict]:
    """
    Translate each pollster's directional lean into per-candidate house effects.

    Formula: house_effect[cand] = lean × bloc_position[cand]

    bloc_positions is parallel to ideological_blocs:
      -1.0 = most conservative,  0.0 = center,  +1.0 = most progressive

    Convention (matches pollster_db.json entry):
      negative lean = R/conservative lean (pollster over-estimates conservative candidates)
      positive lean = D/progressive lean  (pollster over-estimates progressive candidates)

    Stacks additively with any existing per-candidate house_effect_adjustment entries.
    Candidates not assigned to any bloc receive no lean adjustment.
    """
    if not bloc_positions or len(bloc_positions) != len(ideological_blocs):
        return ratings

    cand_position: dict[str, float] = {}
    for bloc, pos in zip(ideological_blocs, bloc_positions):
        for cand in bloc:
            cand_position[cand] = pos

    result = {}
    for pid, rating in ratings.items():
        lean = rating.get("lean")
        if not lean:
            result[pid] = rating
            continue
        r = dict(rating)
        he = dict(r.get("house_effect_adjustment") or {})
        for cand in candidates:
            pos = cand_position.get(cand, 0.0)
            if pos != 0.0:
                he[cand] = he.get(cand, 0.0) + lean * pos
        r["house_effect_adjustment"] = he
        result[pid] = r
    return result


def _apply_internal_discount(polls: list[dict], discount: float) -> list[dict]:
    """
    For internal polls that name a commissioned_by candidate, shade that
    candidate's topline down by `discount` pp via the house_effect mechanism.
    Campaigns only release internal polls when the numbers flatter them, so
    the commissioning candidate's share is likely overstated.
    """
    if discount <= 0:
        return polls
    result = []
    for poll in polls:
        p = deepcopy(poll)
        if p.get("is_internal") and p.get("commissioned_by"):
            cand = p["commissioned_by"]
            he = dict(p.get("house_effect") or {})
            he[cand] = he.get(cand, 0.0) + discount
            p["house_effect"] = he
        result.append(p)
    return result


# ── Topline aggregation ────────────────────────────────────────────────────

def _aggregate_topline(
    polls: list[dict],
    candidates: list[str],
    as_of: str | None = None,
    election_day: str | None = None,
    late_poll_multiplier: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns (baseline, undecided_pct_per_candidate_weight).
    baseline: weighted average vote share for each candidate (fractions, sum ≤ 1).
    Also returns total undecided as a float.
    """
    weighted_sums: dict[str, float] = {c: 0.0 for c in candidates}
    total_weight = 0.0

    for poll in polls:
        w = _composite_weight(poll, as_of, election_day, late_poll_multiplier)
        if w <= 0:
            continue
        topline = poll.get("topline", {})
        for cand in candidates:
            pct = topline.get(cand)
            if pct is None:
                continue
            # Apply house effect adjustment if present (positive = pollster over-estimates this cand)
            he = poll.get("house_effect", {})
            if isinstance(he, dict):
                pct -= he.get(cand, 0.0)
            weighted_sums[cand] += w * pct / 100.0
        total_weight += w

    if total_weight <= 0:
        n = len(candidates)
        baseline = {c: 1.0 / n for c in candidates}
    else:
        baseline = {c: weighted_sums[c] / total_weight for c in candidates}

    return baseline


def _aggregate_crosstabs(
    polls: list[dict],
    candidates: list[str],
    baseline: dict[str, float],
    crosstab_key: str = "crosstabs",
    as_of: str | None = None,
    election_day: str | None = None,
    late_poll_multiplier: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Weighted average of crosstab vote shares by sub-group, anchored to the
    current weighted topline via per-poll deltas.

    For each poll that has both a topline and a crosstab value for a given
    (candidate, group), we compute:
        delta = crosstab_pct - poll_topline_pct
    and accumulate weighted deltas.  The final estimate is:
        baseline[candidate] + weighted_avg(deltas)
    clipped to [0, 1].

    NOTE — additive vs. logit-space deltas:
    Deltas are computed in probability space (percentage points), not logit space.
    This is simpler and works well when all candidates are above ~5%. If a candidate
    drops near zero, additive deltas can produce negative estimates that get clipped
    to 0, which distorts the other candidates' shares. If that happens, switch to
    logit-space deltas: delta = logit(crosstab_pct) - logit(poll_topline_pct), with
    the final estimate = expit(logit(baseline) + weighted_avg(logit_deltas)). That
    approach is more consistent with how compute_precinct_shares works downstream.

    This keeps crosstab estimates consistent with the topline: when a strong
    new poll moves the baseline but carries no crosstab data, the crosstab
    estimates shift with it rather than drifting out of sync.  Polls that
    lack a topline entry for a given candidate are skipped for that candidate
    (a delta cannot be computed without both values).

    Weights are tracked per (group, candidate) so that a poll covering only
    a subset of candidates does not dilute averages for others.
    """
    delta_sums: dict[str, dict[str, float]] = {}
    delta_weights: dict[str, dict[str, float]] = {}

    for poll in polls:
        w = _composite_weight(poll, as_of, election_day, late_poll_multiplier)
        if w <= 0:
            continue
        crosstabs = poll.get(crosstab_key)
        if not crosstabs:
            continue
        poll_topline  = poll.get("topline", {})
        sample_size   = poll.get("sample_size") or 0
        sample_comp   = poll.get("sample_composition") or {}

        for group, shares in crosstabs.items():
            # Normalize demographic group names (race, age) to lowercase so
            # poll JSON capitalization ("White" vs "white") never matters.
            # Geographic crosstabs (ward names) are not affected because they
            # go through the "crosstabs" key, not "demographic_crosstabs".
            if crosstab_key == "demographic_crosstabs":
                group = group.strip().lower()

            # Scale weight by subsample reliability.
            # Known composition: w * sqrt(n_sub / N) — exact.
            # Unknown composition: w * _CROSSTAB_DEFAULT_SCALE — conservative fallback
            # so polls that withhold composition data aren't rewarded over those that report it.
            if crosstab_key == "demographic_crosstabs":
                if sample_size and group in sample_comp:
                    frac    = sample_comp[group]
                    frac    = frac / 100.0 if frac > 1.0 else frac
                    n_sub   = sample_size * frac
                    w_group = w * (n_sub / sample_size) ** 0.5 if n_sub > 0 else 0.0
                else:
                    w_group = w * _CROSSTAB_DEFAULT_SCALE
            else:
                w_group = w

            if group not in delta_sums:
                delta_sums[group] = {c: 0.0 for c in candidates}
                delta_weights[group] = {c: 0.0 for c in candidates}
            for cand in candidates:
                val = shares.get(cand)
                if val is None:
                    continue
                poll_top = poll_topline.get(cand)
                if poll_top is None:
                    continue
                delta = val / 100.0 - poll_top / 100.0
                delta_sums[group][cand]   += w_group * delta
                delta_weights[group][cand] += w_group

    result: dict[str, dict[str, float]] = {}
    for group in delta_sums:
        row: dict[str, float] = {}
        for cand in candidates:
            wt = delta_weights[group][cand]
            if wt > 0:
                avg_delta = delta_sums[group][cand] / wt
                row[cand] = max(0.0, baseline.get(cand, 0.0) + avg_delta)
        if row:
            result[group] = row
    return result


def _aggregate_favorability(
    polls: list[dict],
    candidates: list[str],
    as_of: str | None = None,
    election_day: str | None = None,
    late_poll_multiplier: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Returns {candidate: {"favorable": f, "unfavorable": u, "aware_rate": ar}}
    aware_rate = fav / (fav + unfav), normalized to mean 1.0 across candidates.
    """
    fav_sums = {c: 0.0 for c in candidates}
    unfav_sums = {c: 0.0 for c in candidates}
    total_w = 0.0

    for poll in polls:
        w = _composite_weight(poll, as_of, election_day, late_poll_multiplier)
        if w <= 0:
            continue
        fav_data = poll.get("favorability")
        if not fav_data:
            continue
        for cand in candidates:
            row = fav_data.get(cand, {})
            fav_sums[cand] += w * row.get("favorable", 0.0) / 100.0
            unfav_sums[cand] += w * row.get("unfavorable", 0.0) / 100.0
        total_w += w

    result: dict[str, dict[str, float]] = {}
    if total_w <= 0:
        return result

    raw_aware = {}
    for cand in candidates:
        f = fav_sums[cand] / total_w
        u = unfav_sums[cand] / total_w
        denom = f + u
        raw_aware[cand] = f / denom if denom > 1e-9 else 0.5
        result[cand] = {"favorable": f, "unfavorable": u, "aware_rate": raw_aware[cand]}

    # Normalize aware_rate to mean 1.0
    mean_aware = sum(raw_aware.values()) / len(raw_aware) if raw_aware else 1.0
    if mean_aware > 1e-9:
        for cand in candidates:
            result[cand]["aware_rate"] /= mean_aware

    return result


def _aggregate_second_choice(
    polls: list[dict],
    candidates: list[str],
    as_of: str | None = None,
    election_day: str | None = None,
    late_poll_multiplier: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Returns {from_cand: {to_cand: fraction}} weighted average across polls.
    """
    sc_sums: dict[str, dict[str, float]] = {c: {cc: 0.0 for cc in candidates if cc != c} for c in candidates}
    sc_weights: dict[str, float] = {c: 0.0 for c in candidates}

    for poll in polls:
        w = _composite_weight(poll, as_of, election_day, late_poll_multiplier)
        if w <= 0:
            continue
        sc = poll.get("second_choice")
        if not sc:
            continue
        for from_cand in candidates:
            row = sc.get(from_cand, {})
            for to_cand in candidates:
                if to_cand == from_cand:
                    continue
                val = row.get(to_cand)
                if val is None:
                    continue
                sc_sums[from_cand][to_cand] += w * val / 100.0
            sc_weights[from_cand] += w

    result: dict[str, dict[str, float]] = {}
    for from_cand in candidates:
        wt = sc_weights[from_cand]
        if wt <= 0:
            continue
        result[from_cand] = {
            to_cand: sc_sums[from_cand][to_cand] / wt
            for to_cand in candidates
            if to_cand != from_cand
        }
    return result


# ── Public API ─────────────────────────────────────────────────────────────

def infer_ideology_composition(
    polls: list[dict],
    candidates: list[str],
    ideology_bins: list[str],
    as_of: str | None = None,
    election_day: str | None = None,
    late_poll_multiplier: float = 1.0,
) -> dict[str, float]:
    """
    Estimate the fraction of the electorate in each ideology bin by solving the
    linear system:  topline[c] = sum_g( share[g] * crosstab[g][c] )

    For each poll that has ideology demographic_crosstabs, we either read the
    group shares directly from sample_composition (if present) or solve the
    topline constraint via non-negative least squares.  Results are aggregated
    as a weighted average across polls.

    Returns {bin_name: fraction} summing to 1.0, or {} if no ideology data.
    """
    import numpy as np

    bin_sums   = {b: 0.0 for b in ideology_bins}
    total_w    = 0.0

    for poll in polls:
        w = _composite_weight(poll, as_of, election_day, late_poll_multiplier)
        if w <= 0:
            continue
        ideo_ct = poll.get("demographic_crosstabs") or {}
        bins_present = [b for b in ideology_bins if b in ideo_ct]
        if not bins_present:
            continue

        topline     = poll.get("topline", {})
        sample_comp = poll.get("sample_composition", {})

        # Prefer explicit sample composition when available for ideology bins
        if all(b in sample_comp for b in bins_present):
            raw = {b: float(sample_comp[b]) for b in bins_present}
            total = sum(raw.values())
            if total > 0:
                shares = {b: raw[b] / total for b in bins_present}
            else:
                continue
        else:
            # Solve: topline[c] = sum_b( share[b] * crosstab[b][c] )
            # Only use candidates that appear in both topline and all ideology bins
            cands = [
                c for c in candidates
                if c in topline and all(c in ideo_ct[b] for b in bins_present)
            ]
            if len(cands) < len(bins_present):
                continue  # underdetermined — skip

            A = np.array([
                [ideo_ct[b].get(c, 0.0) / 100.0 for b in bins_present]
                for c in cands
            ])  # shape (n_cands, n_bins)
            t = np.array([topline[c] / 100.0 for c in cands])

            # Non-negative least squares via numpy (no scipy needed)
            # Augment with sum-to-1 constraint weighted heavily
            constraint_weight = 10.0
            A_aug = np.vstack([A, constraint_weight * np.ones((1, len(bins_present)))])
            t_aug = np.append(t, constraint_weight)
            x, _, _, _ = np.linalg.lstsq(A_aug, t_aug, rcond=None)
            x = np.clip(x, 0.0, None)
            if x.sum() < 1e-9:
                continue
            x /= x.sum()
            shares = dict(zip(bins_present, x))

        for b in bins_present:
            bin_sums[b] += w * shares[b]
        total_w += w

    if total_w < 1e-9:
        return {}

    raw = {b: bin_sums[b] / total_w for b in ideology_bins}
    total = sum(raw.values())
    return {b: v / total for b, v in raw.items()} if total > 0 else {}


def aggregate_polls(config: RaceConfig, as_of: str | None = None) -> dict[str, Any]:
    """
    Aggregate all polls in config up to `as_of` date (or today if None).

    Returns a polling snapshot dict:
    {
        "baseline":              {candidate: fraction},
        "undecided_total":       float,
        "senate_district_crosstabs": {group: {candidate: fraction}},
        "demographic_crosstabs": {group: {candidate: fraction}},
        "favorability_topline":  {candidate: {"favorable":, "unfavorable":, "aware_rate":}},
        "second_choice":         {from: {to: fraction}},
        "as_of":                 str (ISO date),
        "n_polls_used":          int,
    }
    """
    polls = config.polls
    if config.pollster_ratings_path:
        ratings = _load_pollster_ratings(config.pollster_ratings_path)
        ratings = _apply_lean_adjustments(
            ratings,
            config.candidates,
            config.ideological_blocs,
            getattr(config, "bloc_ideological_positions", []),
        )
        polls = _apply_pollster_ratings(polls, ratings)
    polls = _apply_internal_discount(
        polls, getattr(config, "internal_candidate_discount", 0.0)
    )

    election_day = config.election_date
    late_mult = getattr(config, "late_poll_multiplier", 1.0)

    baseline = _aggregate_topline(polls, config.candidates, as_of, election_day, late_mult)
    undecided = max(0.0, 1.0 - sum(baseline.values()))

    sd_crosstabs = _aggregate_crosstabs(polls, config.candidates, baseline, "crosstabs", as_of, election_day, late_mult)
    demo_crosstabs = _aggregate_crosstabs(polls, config.candidates, baseline, "demographic_crosstabs", as_of, election_day, late_mult)
    favorability = _aggregate_favorability(polls, config.candidates, as_of, election_day, late_mult)
    second_choice = _aggregate_second_choice(polls, config.candidates, as_of, election_day, late_mult)

    # Count polls with non-zero weight
    n_used = sum(1 for p in polls if _composite_weight(p, as_of, election_day, late_mult) > 0)

    from .precinct_calibration import IDEOLOGY_BINS
    ideology_composition = infer_ideology_composition(
        polls, config.candidates, IDEOLOGY_BINS, as_of, election_day, late_mult
    )

    snapshot = {
        "baseline": baseline,
        "undecided_total": undecided,
        "senate_district_crosstabs": sd_crosstabs,
        "demographic_crosstabs": demo_crosstabs,
        "favorability_topline": favorability,
        "second_choice": second_choice,
        "ideology_composition": ideology_composition,
        "as_of": as_of or date.today().isoformat(),
        "n_polls_used": n_used,
    }
    return snapshot


def run_district_simulation(config: RaceConfig, polling: dict[str, Any]) -> dict[str, Any]:
    """
    Call the Rust simulator for the district-level Monte Carlo.
    Returns the simulator output augmented with polling metadata.

    Undecided handling: when the polling average has undecided share > 0,
    each simulation trial draws a fresh allocation from a Dirichlet distribution
    parameterized by config.undecided_allocation blended with favorability
    aware_rate.  This correctly models uncertainty about where undecideds will
    land rather than locking them into a fixed proportional split every trial.
    """
    baseline  = polling["baseline"]
    undecided = polling.get("undecided_total", 0.0)   # fraction (0–1)
    sc_matrix = polling.get("second_choice", {})

    # Build undecided allocation weights: config prior blended with favorability
    fav   = polling.get("favorability_topline", {})
    blend = getattr(config, "favorability_blend", 0.25)
    alloc = getattr(config, "undecided_allocation", {}).copy()
    for c in config.candidates:
        aware    = fav.get(c, {}).get("aware_rate", 1.0) if fav else 1.0
        alloc[c] = (1.0 - blend) * alloc.get(c, 1.0) + blend * aware

    # Viability scaling: multiply each weight by polling_share^alpha so frontrunners
    # attract a proportionally larger share of undecideds. Applied after the
    # favorability blend so both signals are reflected before scaling.
    alpha = getattr(config, "undecided_viability_alpha", 0.0)
    if alpha > 0.0 and undecided > 0:
        for c in config.candidates:
            share    = max(baseline.get(c, 0.0), 1e-6)  # floor avoids 0^alpha = 0
            alloc[c] = alloc[c] * (share ** alpha)

    sim_result = simulator_runner.run_district_sim(
        n_simulations=config.n_sim_district,
        candidates=config.candidates,
        baseline=baseline,
        moe_district=config.moe_district,
        ideological_blocs=config.ideological_blocs,
        second_choice_matrix=sc_matrix,
        second_choice_strength=config.second_choice_strength,
        has_runoff=getattr(config, "has_runoff", False),
        runoff_threshold=getattr(config, "runoff_threshold", 0.50),
        undecided_share=undecided * 100.0,   # Rust expects percentage points
        undecided_weights=alloc if undecided > 0 else None,
        undecided_concentration=3.0,
        fundamental_uncertainty_sigma=getattr(config, "fundamental_uncertainty_sigma", 0.0),
        sigma_within_bloc_fraction=getattr(config, "sigma_within_bloc_fraction", 0.0),
        environment_shock_fraction=getattr(config, "environment_shock_fraction", 0.0),
    )

    sim_result["baseline"] = baseline
    sim_result["undecided_total"] = undecided
    return sim_result


def build_versioned_history(
    config: RaceConfig,
    history_dates: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a chronological list of polling snapshots, one per date in history_dates.
    If history_dates is None, uses one entry per unique poll field_end date.

    Each snapshot in the returned list has all fields from aggregate_polls()
    plus a "district_sim" key with win_probs and median_vote_shares.

    This powers the banked vote engine's snapshot-matching logic.
    """
    if history_dates is None:
        seen: set[str] = set()
        history_dates = []
        for p in sorted(config.polls, key=lambda x: x["field_end"]):
            d = p["field_end"]
            if d not in seen:
                seen.add(d)
                history_dates.append(d)

    snapshots = []
    for d in history_dates:
        snap = aggregate_polls(config, as_of=d)
        # District simulation for this snapshot
        try:
            district = run_district_simulation(config, snap)
            snap["district_sim"] = district
        except FileNotFoundError:
            snap["district_sim"] = None  # simulator not built yet
        snapshots.append(snap)

    return snapshots
