"""
Diagnostic: show which polls contribute to each demographic group's crosstab
estimate, their recency weights, and the resulting delta vs. baseline.

Run from repo root: python -m races.chicago_mayor_2023.crosstab_coverage
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from races.chicago_mayor_2023.race_config import CONFIG
from core.poll_weighting import _composite_weight, aggregate_polls
from dataclasses import replace

SHOW_CANDIDATES = ["Paul Vallas", "Brandon Johnson", "Lori Lightfoot", "Chuy García", "Willie Wilson"]


def main():
    with CONFIG.polls_round1_path.open() as f:
        raw_polls = json.load(f)

    config = replace(CONFIG, polls=raw_polls)
    as_of  = CONFIG.election_date  # use election day as reference

    # Build polling snapshot to get baseline and crosstab deltas
    snapshot = aggregate_polls(config, as_of=as_of)
    baseline = snapshot["baseline"]
    demo_ct  = snapshot["demographic_crosstabs"]

    election_day  = CONFIG.election_date
    late_mult     = CONFIG.late_poll_multiplier

    # Find all demographic groups across all polls
    all_groups: set[str] = set()
    for poll in raw_polls:
        for g in (poll.get("demographic_crosstabs") or {}):
            all_groups.add(g.strip().lower())

    print("=" * 75)
    print("CROSSTAB COVERAGE DIAGNOSTIC")
    print(f"Reference date: {as_of}  |  late_poll_multiplier: {late_mult}")
    print("=" * 75)

    for group in sorted(all_groups):
        contributing = []
        for poll in raw_polls:
            ideo_ct = poll.get("demographic_crosstabs") or {}
            normalized = {k.strip().lower(): v for k, v in ideo_ct.items()}
            if group not in normalized:
                continue
            w = _composite_weight(poll, as_of, election_day, late_mult)
            if w <= 0:
                continue

            sample_size = poll.get("sample_size") or 0
            sample_comp = poll.get("sample_composition") or {}
            comp_key    = next((k for k in sample_comp if k.strip().lower() == group), None)

            if sample_size and comp_key:
                frac   = sample_comp[comp_key]
                frac   = frac / 100.0 if frac > 1.0 else frac
                n_sub  = int(sample_size * frac)
                w_grp  = w * (frac ** 0.5)
            else:
                n_sub  = None
                w_grp  = w * 0.5  # _CROSSTAB_DEFAULT_SCALE

            shares = normalized[group]
            contributing.append({
                "pollster": poll.get("pollster_name", poll.get("pollster_id", "?")),
                "field_end": poll["field_end"],
                "w_base": w,
                "w_group": w_grp,
                "n_sub": n_sub,
                "shares": shares,
            })

        if not contributing:
            continue

        total_wg = sum(p["w_group"] for p in contributing)

        print(f"\n── {group.upper()} ({len(contributing)} poll(s) with crosstabs)")
        print(f"   {'Pollster':<28} {'Date':<12} {'Rel.Wt':>7} {'n_sub':>6}  ", end="")
        print("  ".join(f"{c[:8]:>8}" for c in SHOW_CANDIDATES))
        print("   " + "-" * (28 + 12 + 7 + 6 + 4 + 10 * len(SHOW_CANDIDATES)))

        for p in sorted(contributing, key=lambda x: x["field_end"]):
            rel_wt = p["w_group"] / total_wg if total_wg > 0 else 0
            n_str  = str(p["n_sub"]) if p["n_sub"] is not None else "unknown"
            row    = f"   {p['pollster']:<28} {p['field_end']:<12} {rel_wt:>6.1%} {n_str:>6}  "
            for c in SHOW_CANDIDATES:
                val = p["shares"].get(c)
                row += f"  {str(round(val)) + '%' if val is not None else '—':>8}"
            print(row)

        # Show final aggregated estimate vs baseline
        print(f"\n   {'Baseline':>42}  ", end="")
        print("  ".join(f"{round(baseline.get(c, 0) * 100):>7}%" for c in SHOW_CANDIDATES))
        if group in demo_ct:
            print(f"   {'Crosstab estimate':>42}  ", end="")
            print("  ".join(f"{round(demo_ct[group].get(c, 0) * 100):>7}%" for c in SHOW_CANDIDATES))
            print(f"   {'Delta (crosstab − baseline)':>42}  ", end="")
            for c in SHOW_CANDIDATES:
                d = (demo_ct[group].get(c, 0) - baseline.get(c, 0)) * 100
                print(f"  {f'{d:+.0f}pp':>8}", end="")
            print()

    print()


if __name__ == "__main__":
    main()
