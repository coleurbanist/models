"""
Poll aggregation, weighting, and versioned history.

Each poll in config.polls is a dict with these keys:
    pollster_id     str
    pollster_name   str
    pollster_quality float          (0–1; overridden by pollster_ratings.json if present)
    field_end       str             YYYY-MM-DD
    sample_size     int
    moe             float           margin of error (percentage points)
    is_internal     bool
    topline         dict[str, float]  {candidate: pct}  (values sum to ≤100; undecided = 100 - sum)
    crosstabs       dict | None     {senate_district/ward: {candidate: pct}} or None
    favorability    dict | None     {candidate: {"favorable": x, "unfavorable": y}} or None
    second_choice   dict | None     {from_candidate: {to_candidate: pct}} or None
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .race_config import RaceConfig
from . import simulator_runner


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


def _composite_weight(poll: dict, as_of: str | None = None) -> float:
    q = poll.get("pollster_quality", 0.7)
    r = _recency_weight(poll["field_end"], as_of)
    m = _moe_weight(poll["moe"])
    internal = 0.5 if poll.get("is_internal", False) else 1.0
    return q * r * m * internal


# ── Pollster ratings override ──────────────────────────────────────────────

def _load_pollster_ratings(path: Path) -> dict[str, dict]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _apply_pollster_ratings(polls: list[dict], ratings: dict) -> list[dict]:
    result = []
    for poll in polls:
        p = deepcopy(poll)
        pid = p.get("pollster_id", "")
        if pid in ratings:
            p["pollster_quality"] = ratings[pid].get("quality", p.get("pollster_quality", 0.7))
            if "house_effect" not in p and "house_effect_adjustment" in ratings[pid]:
                p["house_effect"] = ratings[pid]["house_effect_adjustment"]
        result.append(p)
    return result


# ── Topline aggregation ────────────────────────────────────────────────────

def _aggregate_topline(
    polls: list[dict],
    candidates: list[str],
    as_of: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns (baseline, undecided_pct_per_candidate_weight).
    baseline: weighted average vote share for each candidate (fractions, sum ≤ 1).
    Also returns total undecided as a float.
    """
    weighted_sums: dict[str, float] = {c: 0.0 for c in candidates}
    total_weight = 0.0

    for poll in polls:
        w = _composite_weight(poll, as_of)
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
        w = _composite_weight(poll, as_of)
        if w <= 0:
            continue
        crosstabs = poll.get(crosstab_key)
        if not crosstabs:
            continue
        poll_topline = poll.get("topline", {})
        for group, shares in crosstabs.items():
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
                delta_sums[group][cand] += w * delta
                delta_weights[group][cand] += w

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
) -> dict[str, dict[str, float]]:
    """
    Returns {candidate: {"favorable": f, "unfavorable": u, "aware_rate": ar}}
    aware_rate = fav / (fav + unfav), normalized to mean 1.0 across candidates.
    """
    fav_sums = {c: 0.0 for c in candidates}
    unfav_sums = {c: 0.0 for c in candidates}
    total_w = 0.0

    for poll in polls:
        w = _composite_weight(poll, as_of)
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
) -> dict[str, dict[str, float]]:
    """
    Returns {from_cand: {to_cand: fraction}} weighted average across polls.
    """
    sc_sums: dict[str, dict[str, float]] = {c: {cc: 0.0 for cc in candidates if cc != c} for c in candidates}
    sc_weights: dict[str, float] = {c: 0.0 for c in candidates}

    for poll in polls:
        w = _composite_weight(poll, as_of)
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
        polls = _apply_pollster_ratings(polls, ratings)

    baseline = _aggregate_topline(polls, config.candidates, as_of)
    undecided = max(0.0, 1.0 - sum(baseline.values()))

    sd_crosstabs = _aggregate_crosstabs(polls, config.candidates, baseline, "crosstabs", as_of)
    demo_crosstabs = _aggregate_crosstabs(polls, config.candidates, baseline, "demographic_crosstabs", as_of)
    favorability = _aggregate_favorability(polls, config.candidates, as_of)
    second_choice = _aggregate_second_choice(polls, config.candidates, as_of)

    # Count polls with non-zero weight
    n_used = sum(1 for p in polls if _composite_weight(p, as_of) > 0)

    snapshot = {
        "baseline": baseline,
        "undecided_total": undecided,
        "senate_district_crosstabs": sd_crosstabs,
        "demographic_crosstabs": demo_crosstabs,
        "favorability_topline": favorability,
        "second_choice": second_choice,
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
