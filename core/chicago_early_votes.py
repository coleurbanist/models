"""
Chicago early / mail vote engine.

Why this is Chicago-specific
─────────────────────────────
For most races (IL-09, statewide), election authorities report cumulative
early vote totals by county/city but give no demographic breakdown until
results are certified. We can't reliably say which candidate those votes
are going to, so we don't model them.

Chicago is different: the Chicago Board of Elections posts age and gender
demographics for early/mail ballots roughly 30 minutes after 7pm on election
night. That breakdown, combined with the demographic crosstabs from polls,
lets us distribute the early vote pool by candidate before precinct results
come in.

Two phases
──────────
Phase 1 — Pre-election (no demographic data yet):
    Distribute early votes using the 60/40 ward-geographic-pattern /
    district-snapshot blend, same logic as the original IL-09 approach.
    Treat these estimates as LOW confidence — they're useful for showing
    a running total but not for calling the race.

Phase 2 — Election night (Chicago BOE posts demographics):
    Replace Phase 1 estimates with demographic-weighted allocations.
    Load the posted age+gender breakdown, apply demographic crosstabs
    from the polling snapshot, and distribute votes accordingly.
    These estimates are substantially more reliable.

Data files expected (in config.data_dir / "csv_data"):
    early_votes_by_ward.csv       — cumulative early/mail totals by ward + date
    chicago_boe_demographics.csv  — posted on election night by Chicago BOE
                                    columns: age_group, gender, count
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .race_config import RaceConfig


# ── Phase 1: pre-election snapshot-based estimate ─────────────────────────

def _find_nearest_snapshot(target_date: str, history: list[dict]) -> dict | None:
    candidates = [s for s in history if s["as_of"] <= target_date]
    return max(candidates, key=lambda s: s["as_of"]) if candidates else None


def _round_votes(votes: dict[str, float]) -> dict[str, int]:
    total = round(sum(votes.values()))
    floored = {c: int(v) for c, v in votes.items()}
    remainder = total - sum(floored.values())
    fracs = {c: votes[c] - floored[c] for c in votes}
    for _ in range(remainder):
        leader = max(fracs, key=fracs.__getitem__)
        floored[leader] += 1
        fracs[leader] = 0.0
    return floored


def _load_ward_priors(config: RaceConfig) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Load ward_priors.json written by run_precinct_pipeline().
    Returns (ward_priors, district_baseline).
    ward_priors: {ward_key: {candidate: share}}
    district_baseline: the model's original district-level shares (used to compute ward leans)
    Both are empty dicts if the file doesn't exist yet.
    """
    import json as _json
    path = config.output_dir / "ward_priors.json"
    if not path.exists():
        return {}, {}
    try:
        raw = _json.loads(path.read_text(encoding="utf-8"))
        district = raw.pop("_district", {})
        return raw, district
    except Exception as exc:
        warnings.warn(f"Could not read ward_priors.json: {exc}")
        return {}, {}


def _ward_adjusted_shares(
    ward_key: str,
    ward_priors: dict[str, dict[str, float]],
    district_baseline: dict[str, float],
    current_shares: dict[str, float],
    candidates: list[str],
) -> dict[str, float]:
    """
    Compute ward-specific vote shares for a given polling snapshot.

    The ward priors encode the spatial structure (which wards lean which way)
    from the precinct model. The current_shares encode the district-level totals
    from the most recent poll. We combine them by adding each ward's lean
    (deviation from the original model district average) onto the current
    district shares, then renormalize.

    Falls back to current_shares if no ward prior exists for this ward.
    """
    ward_prior = ward_priors.get(ward_key)
    if not ward_prior or not district_baseline:
        raw = {c: current_shares.get(c, 1.0 / len(candidates)) for c in candidates}
    else:
        raw = {}
        for c in candidates:
            lean = ward_prior.get(c, 0.0) - district_baseline.get(c, 0.0)
            raw[c] = max(0.0, current_shares.get(c, 0.0) + lean)

    total = sum(raw.values())
    if total > 1e-9:
        return {c: raw[c] / total for c in candidates}
    return {c: 1.0 / len(candidates) for c in candidates}


def estimate_early_votes_snapshot(
    config: RaceConfig,
    versioned_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Phase 1: distribute early votes using per-ward priors + daily polling snapshots.

    Iterates over daily increments in the early vote CSV rather than applying a
    single snapshot to the cumulative total. Each day's new votes are allocated
    using the polling snapshot that was current on that day, so a candidate surge
    captured in a new poll only affects votes banked from that point forward.

    Ward priors (ward_priors.json) from run_precinct_pipeline() supply the spatial
    structure: each ward's shares are the current district-level polling numbers
    adjusted by how much that ward deviates from the model's district average.
    Falls back to district-level shares if no ward priors exist.

    Reads: config.data_dir / "csv_data" / "early_votes_by_ward.csv"
    Expected CSV format: index=ward number (int), columns=DD-Mon cumulative totals
    """
    csv_path = config.data_path("csv_data", "early_votes_by_ward.csv")
    if not csv_path.exists():
        return {"available": False, "confidence": "none"}

    df = pd.read_csv(str(csv_path), index_col=0)

    from datetime import date as _date
    date_cols: list[tuple[str, str]] = []
    for col in df.columns:
        try:
            d = pd.to_datetime(col, format="%d-%b").replace(year=_date.today().year)
            date_cols.append((col, d.strftime("%Y-%m-%d")))
        except ValueError:
            pass
    if not date_cols:
        return {"available": False, "confidence": "none"}

    date_cols.sort(key=lambda x: x[1])

    ward_priors, district_baseline = _load_ward_priors(config)
    using_ward_priors = bool(ward_priors)

    # Accumulate votes across days using daily increments
    by_ward_float: dict[str, dict[str, float]] = {}
    by_ward_total: dict[str, float] = {}
    district_votes_float: dict[str, float] = {c: 0.0 for c in config.candidates}
    district_total = 0.0
    snapshots_used: set[str] = set()

    prev_cumulative: dict[str, float] = {str(w): 0.0 for w in df.index}

    for col, iso_date in date_cols:
        snap = _find_nearest_snapshot(iso_date, versioned_history)
        if not snap:
            continue
        snapshots_used.add(snap["as_of"])
        current_shares = snap.get("district_sim", {}).get("median_vote_shares") or snap.get("baseline", {})

        for ward_label in df.index:
            ward_str = str(ward_label)
            cumulative = float(df.loc[ward_label, col])
            increment = cumulative - prev_cumulative.get(ward_str, 0.0)
            if increment <= 0:
                continue

            ward_key = f"{int(ward_label):02d}" if ward_str.isdigit() else ward_str
            shares = _ward_adjusted_shares(
                ward_key, ward_priors, district_baseline, current_shares, config.candidates
            )

            if ward_str not in by_ward_float:
                by_ward_float[ward_str] = {c: 0.0 for c in config.candidates}
                by_ward_total[ward_str] = 0.0
            for c in config.candidates:
                by_ward_float[ward_str][c] += shares[c] * increment
                district_votes_float[c] += shares[c] * increment
            by_ward_total[ward_str] += increment
            district_total += increment

        prev_cumulative = {str(w): float(df.loc[w, col]) for w in df.index}

    if district_total <= 0:
        return {"available": False, "confidence": "none"}

    by_ward: dict[str, Any] = {}
    for ward_str, float_votes in by_ward_float.items():
        rounded = _round_votes(float_votes)
        total = by_ward_total[ward_str]
        by_ward[ward_str] = {
            "total": round(total),
            "votes": rounded,
            "pcts": {c: rounded[c] / total if total > 0 else 0.0 for c in config.candidates},
        }

    district_votes = _round_votes(district_votes_float)
    total_d = sum(district_votes.values())
    latest_iso = date_cols[-1][1]

    return {
        "available": True,
        "confidence": "low",
        "phase": 1,
        "as_of_date": latest_iso,
        "snapshots_used": sorted(snapshots_used),
        "ward_priors_source": "precinct_model" if using_ward_priors else "district_snapshot",
        "district_total": round(district_total),
        "district_votes": district_votes,
        "district_pcts": {c: district_votes[c] / total_d if total_d > 0 else 0.0 for c in config.candidates},
        "by_ward": by_ward,
    }


# ── Phase 2: election-night demographic-weighted estimate ──────────────────

def estimate_early_votes_demographic(
    config: RaceConfig,
    polling_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Phase 2: distribute early votes using Chicago BOE's election-night
    demographic release (age + gender breakdown).

    chicago_boe_demographics.csv format:
        age_group   — e.g. "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
        gender      — "M" or "F"
        count       — number of early/mail ballots in this cell

    The polling snapshot must contain "demographic_crosstabs" with entries
    keyed by age_group and gender matching the above (or compatible groupings).

    Returns district totals and by-(age_group,gender) breakdown,
    flagged as high-confidence.
    """
    demo_path = config.data_path("csv_data", "chicago_boe_demographics.csv")
    if not demo_path.exists():
        warnings.warn(
            f"Chicago BOE demographic file not found at {demo_path}. "
            "Run Phase 1 estimate instead or wait for Chicago BOE to post data."
        )
        return {"available": False, "confidence": "none", "phase": 2}

    demo_df = pd.read_csv(str(demo_path))
    required = {"age_group", "gender", "count"}
    if not required.issubset(demo_df.columns):
        raise ValueError(f"chicago_boe_demographics.csv must have columns: {required}")

    demo_crosstabs = polling_snapshot.get("demographic_crosstabs", {})

    by_cell: list[dict] = []
    district_votes: dict[str, float] = {c: 0.0 for c in config.candidates}
    district_total = 0.0

    for _, row in demo_df.iterrows():
        cell_key = f"{row['age_group']}_{row['gender']}"
        count = float(row["count"])
        if count <= 0:
            continue

        # Look up demographic shares; fall back to district baseline
        shares = (
            demo_crosstabs.get(cell_key)
            or demo_crosstabs.get(row["age_group"])
            or polling_snapshot.get("baseline", {})
        )
        total_shares = sum(shares.get(c, 0.0) for c in config.candidates)
        if total_shares < 1e-9:
            total_shares = 1.0

        cell_votes = {c: (shares.get(c, 0.0) / total_shares) * count for c in config.candidates}
        by_cell.append({
            "age_group": row["age_group"],
            "gender": row["gender"],
            "count": round(count),
            "votes": {c: round(cell_votes[c]) for c in config.candidates},
        })
        for c in config.candidates:
            district_votes[c] += cell_votes[c]
        district_total += count

    rounded_total = _round_votes(district_votes)
    total_d = sum(rounded_total.values())
    return {
        "available": True,
        "confidence": "high",
        "phase": 2,
        "district_total": round(district_total),
        "district_votes": rounded_total,
        "district_pcts": {c: rounded_total[c] / total_d if total_d > 0 else 0.0 for c in config.candidates},
        "by_demographic_cell": by_cell,
    }


# ── Dispatcher ─────────────────────────────────────────────────────────────

def compute_chicago_early_votes(
    config: RaceConfig,
    polling_snapshot: dict[str, Any],
    versioned_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Auto-select Phase 2 if Chicago BOE demographics are available,
    otherwise fall back to Phase 1.

    Raises ValueError if config.banked_vote_mode != "chicago".
    """
    if config.banked_vote_mode != "chicago":
        raise ValueError(
            f"compute_chicago_early_votes() called for race '{config.race_id}' "
            f"which has banked_vote_mode='{config.banked_vote_mode}'. "
            "Only Chicago mayoral / aldermanic races should use this function."
        )

    demo_path = config.data_path("csv_data", "chicago_boe_demographics.csv")
    if demo_path.exists():
        return estimate_early_votes_demographic(config, polling_snapshot)
    else:
        return estimate_early_votes_snapshot(config, versioned_history)
