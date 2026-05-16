"""
Chicago Mayoral 2023 — backtesting pipeline.

Used to validate the model against the known 2023 results:
  Round 1 (2023-02-28): Vallas 33.8 / Johnson 21.7 / Lightfoot 17.1 / García 13.7 / Wilson 8.9
  Runoff   (2023-04-04): Johnson 51.4 / Vallas 48.6

BASIC USAGE
-----------
Run from the repo root (elections/models/):

    python -m races.chicago_mayor_2023.run

FLAGS
-----
--round {round1,runoff}
    Which round to model. Default is round1.

--compare
    After running, print a side-by-side of model mean vote shares vs actual results.

--poll-id <ID>
    Run with a single poll instead of the weighted aggregate.

--list-polls
    Print available poll IDs for the selected round and exit.

ACTUAL RESULTS (for --compare)
-------------------------------
Round 1:
  Paul Vallas       33.8%
  Brandon Johnson   21.7%
  Lori Lightfoot    17.1%
  Chuy García       13.7%
  Willie Wilson      8.9%

Runoff:
  Brandon Johnson   51.4%
  Paul Vallas       48.6%
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from races.chicago_mayor_2023.race_config import CONFIG
from core.poll_weighting import aggregate_polls, run_district_simulation, build_versioned_history, get_poll_id
from core.precinct_pipeline import run_precinct_pipeline
from core.regional_forecast import generate_regional_forecast
from core.chicago_early_votes import compute_chicago_early_votes

_ACTUALS = {
    "round1": {
        "Paul Vallas":      0.338,
        "Brandon Johnson":  0.217,
        "Lori Lightfoot":   0.171,
        "Chuy García":      0.137,
        "Willie Wilson":    0.089,
    },
    "runoff": {
        "Brandon Johnson":  0.514,
        "Paul Vallas":      0.486,
    },
}


def _load_polls_for_round(round_name: Literal["round1", "runoff"]) -> list[dict]:
    path = CONFIG.polls_round1_path if round_name == "round1" else CONFIG.polls_runoff_path
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main(
    include_early_votes: bool = False,
    poll_id: str | None = None,
    round_name: Literal["round1", "runoff"] = "round1",
    compare: bool = False,
) -> None:
    all_polls = _load_polls_for_round(round_name)
    config = replace(CONFIG, polls=all_polls)

    if poll_id:
        matching = [p for p in all_polls if get_poll_id(p) == poll_id]
        if not matching:
            available = sorted(get_poll_id(p) for p in all_polls)
            print(f"Poll ID '{poll_id}' not found in {round_name}. Available poll IDs:")
            for pid in available:
                print(f"  {pid}")
            sys.exit(1)
        config    = replace(config, polls=matching)
        out_dir   = CONFIG.output_dir / "poll_snapshots" / poll_id
        run_label = f"{round_name}:poll:{poll_id}"
    else:
        out_dir   = CONFIG.output_dir
        run_label = f"{round_name}:aggregate"

    print(f"[Chicago Mayor 2023 — backtest] {config.race_label}  ({run_label})")

    print("  Stage 1: Aggregating polls...")
    polling = aggregate_polls(config)

    print(f"  Stage 2: District simulation ({config.n_sim_district:,} trials)...")
    district = run_district_simulation(config, polling)

    advance  = district.get("advance_probs") or {}
    outright = district.get("outright_win_probs") or {}
    means    = district.get("mean_vote_shares") or district.get("median_vote_shares") or {}

    print(f"    {'Candidate':<24} {'Top-2':>7} {'Outright':>9} {'Exp %':>7}")
    print(f"    {'-'*24} {'-'*7} {'-'*9} {'-'*7}")
    for cand in config.candidates:
        print(
            f"    {cand:<24}"
            f"  {advance.get(cand, district['win_probs'].get(cand, 0.0)):>6.1%}"
            f"  {outright.get(cand, 0.0):>8.1%}"
            f"  {means.get(cand, 0.0):>6.1%}"
        )
    prob_no_runoff = district.get("prob_no_runoff")
    if prob_no_runoff is not None:
        print(f"\n    P(outright win, no runoff needed): {prob_no_runoff:.1%}")

    if compare:
        actuals = _ACTUALS.get(round_name, {})
        if actuals:
            print(f"\n    {'Candidate':<24} {'Model':>7} {'Actual':>8} {'Error':>7}")
            print(f"    {'-'*24} {'-'*7} {'-'*8} {'-'*7}")
            for cand in config.candidates:
                model_share  = means.get(cand, 0.0)
                actual_share = actuals.get(cand, 0.0)
                error        = model_share - actual_share
                print(
                    f"    {cand:<24}"
                    f"  {model_share:>6.1%}"
                    f"  {actual_share:>7.1%}"
                    f"  {error * 100:>+6.1f}pp"
                )

    print(f"  Stage 3: Precinct simulation ({config.n_sim_precinct:,} trials)...")
    precincts = run_precinct_pipeline(config, polling, district)
    print(f"    {len(precincts)} precincts")

    print("  Stage 4: Regional forecast (ward-level)...")
    regional = generate_regional_forecast(config, precincts)

    early_votes = None
    if include_early_votes:
        print("  Stage 4b: Early vote estimates...")
        history = build_versioned_history(config)
        early_votes = compute_chicago_early_votes(config, polling, history)
        conf  = early_votes.get("confidence", "none")
        phase = early_votes.get("phase", 0)
        print(f"    Phase {phase}, confidence={conf}, "
              f"total={early_votes.get('district_total', 0):,}")

    print("  Stage 5: Writing outputs...")
    _write_outputs(config, out_dir, poll_id, polling, district, precincts, regional, early_votes)
    print(f"  Done → {out_dir}")


def _write_outputs(config, out_dir, poll_id, polling, district, precincts, regional, early_votes):
    out_dir.mkdir(parents=True, exist_ok=True)

    _save(out_dir / "poll_baseline.json", {
        "current": polling,
        "last_run": polling["as_of"],
        "n_simulations": config.n_sim_district,
    })
    _save(out_dir / "district_win_probabilities.json", district)
    _save(out_dir / "regional_vote_forecast.json", regional)
    if early_votes:
        _save(out_dir / "chicago_early_votes.json", early_votes)

    precinct_csv = out_dir / f"{config.race_id}_precinct_probabilities.csv"
    precincts.to_csv(str(precinct_csv), index=False)

    _update_history(config, poll_id, polling, district, regional, precinct_csv)


def _update_history(config, poll_id, polling, district, regional, precinct_csv):
    main_out     = CONFIG.output_dir
    history_path = main_out / "projection_history.json"
    snapshots    = main_out / "snapshots"
    snapshots.mkdir(exist_ok=True)

    ts      = datetime.now()
    ts_slug = ts.strftime("%Y%m%dT%H%M%S")

    snapshot_name = f"{ts_slug}_{poll_id or 'aggregate'}_precinct_probabilities.csv"
    if precinct_csv.exists():
        shutil.copy2(precinct_csv, snapshots / snapshot_name)

    entry = {
        "run_timestamp": ts.isoformat(timespec="seconds"),
        "as_of_date":    polling.get("as_of"),
        "poll_id":       poll_id,
        "n_polls_used":  polling.get("n_polls_used"),
        "district":      district,
        "regional":      regional,
        "precinct_snapshot": f"snapshots/{snapshot_name}",
    }

    if history_path.exists():
        with history_path.open(encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)
    _save(history_path, history)


def _save(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chicago Mayor 2023 backtesting pipeline")
    parser.add_argument("--round", choices=["round1", "runoff"], default="round1")
    parser.add_argument("--compare", action="store_true",
                        help="Print model vs actual results side-by-side")
    parser.add_argument("--early-votes", action="store_true")
    parser.add_argument("--poll-id", metavar="ID")
    parser.add_argument("--list-polls", action="store_true")
    args = parser.parse_args()

    round_name = args.round

    if args.list_polls:
        polls = _load_polls_for_round(round_name)
        print(f"Available poll IDs ({round_name}):")
        for p in polls:
            print(f"  {get_poll_id(p):<36}  {p['pollster_name']}  ({p['field_end']})")
        sys.exit(0)

    main(
        include_early_votes=args.early_votes,
        poll_id=args.poll_id,
        round_name=round_name,
        compare=args.compare,
    )
