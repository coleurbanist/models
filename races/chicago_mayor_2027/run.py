"""
Chicago Mayoral 2027 pipeline orchestrator.

BASIC USAGE
-----------
Run from the repo root (elections/models/):

    python -m races.chicago_mayor_2027.run

This runs the full pipeline using the weighted average of all polls in
race_config.py and writes results to the output directory configured there.

FLAGS
-----
--early-votes
    Also estimate banked early/mail votes using Chicago BOE demographic data.
    Only useful on or after election night when the BOE releases precinct-level
    age/gender breakdowns (~30 min after polls close).

--poll-id <ID>
    Run with a single poll instead of the weighted aggregate. Useful for
    seeing what one pollster's numbers imply on their own. Results go to
    output/poll_snapshots/<ID>/ and do not overwrite the main aggregate output.

--list-polls
    Print all poll IDs currently in race_config.py and exit. Use this to
    find the right ID to pass to --poll-id.

EXAMPLES
--------
    # Normal aggregate run
    python -m races.chicago_mayor_2027.run

    # See what polls are loaded
    python -m races.chicago_mayor_2027.run --list-polls

    # Run with just a specific PPP poll (use --list-polls to see IDs)
    python -m races.chicago_mayor_2027.run --poll-id ppp_2027-01-15

    # Run aggregate + early vote estimates (election night only)
    python -m races.chicago_mayor_2027.run --early-votes

OUTPUTS
-------
Written to the output directory defined in race_config.py:

    poll_baseline.json              — weighted poll snapshot (topline, crosstabs, etc.)
    district_win_probabilities.json — citywide Monte Carlo results:
                                        advance_probs     (prob to reach top-2 / runoff)
                                        outright_win_probs (prob to win outright >50%)
                                        prob_no_runoff     (prob anyone wins outright)
                                        mean_vote_shares / median_vote_shares
                                        p05_vote_shares / p95_vote_shares
    regional_vote_forecast.json     — per-ward expected turnout, vote shares, raw votes
    chicago_mayor_2027_precinct_probabilities.csv
                                    — per-precinct: median_pct_{cand}, win_prob_{cand},
                                        median_votes_{cand} for every candidate
    chicago_early_votes.json        — early vote estimates (only with --early-votes)

    projection_history.json         — cumulative log of every run; each entry has a
                                        timestamp, which poll(s) were used, and the
                                        full district + regional results. Lets the
                                        website project show how projections changed
                                        over the campaign.
    snapshots/                      — timestamped precinct CSV from every run, e.g.
                                        20260513T143000_aggregate_precinct_probabilities.csv
                                        Used by the website project to show precinct-level
                                        history at any point in the campaign.

    poll_snapshots/<poll_id>/       — same set of files as above but for a single-poll
                                        run (--poll-id). Does not overwrite main outputs.

ADDING A NEW POLL
-----------------
Edit races/chicago_mayor_2027/race_config.py and append a new dict to the polls=[]
list. Required keys: pollster_id, pollster_name, pollster_quality, field_end,
sample_size, moe, is_internal, topline. Optional: crosstabs, demographic_crosstabs,
favorability, second_choice. See existing entries for format. Then re-run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from races.chicago_mayor_2027.race_config import CONFIG
from core.poll_weighting import aggregate_polls, run_district_simulation, build_versioned_history, get_poll_id
from core.precinct_pipeline import run_precinct_pipeline
from core.regional_forecast import generate_regional_forecast
from core.chicago_early_votes import compute_chicago_early_votes


def main(include_early_votes: bool = False, poll_id: str | None = None) -> None:
    # ── Build the config to use for this run ──────────────────────────────────
    if poll_id:
        matching = [p for p in CONFIG.polls if get_poll_id(p) == poll_id]
        if not matching:
            available = sorted(get_poll_id(p) for p in CONFIG.polls)
            print(f"Poll ID '{poll_id}' not found. Available poll IDs:")
            for pid in available:
                print(f"  {pid}")
            sys.exit(1)
        config   = replace(CONFIG, polls=matching)
        out_dir  = CONFIG.output_dir / "poll_snapshots" / poll_id
        run_label = f"poll:{poll_id}"
    else:
        config   = CONFIG
        out_dir  = CONFIG.output_dir
        run_label = "aggregate"

    print(f"[Chicago Mayor 2027] {config.race_label}  ({run_label})")

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
            f"  {advance.get(cand, district['win_probs'][cand]):>6.1%}"
            f"  {outright.get(cand, 0.0):>8.1%}"
            f"  {means.get(cand, 0.0):>6.1%}"
        )
    prob_no_runoff = district.get("prob_no_runoff")
    if prob_no_runoff is not None:
        print(f"\n    P(outright win, no runoff needed): {prob_no_runoff:.1%}")

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


def _write_outputs(
    config,
    out_dir: Path,
    poll_id: str | None,
    polling,
    district,
    precincts,
    regional,
    early_votes,
) -> None:
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

    # History is always written to the main output dir regardless of poll_id,
    # so aggregate and per-poll runs are all in one place for the website project.
    _update_history(config, poll_id, polling, district, regional, precinct_csv)


def _update_history(
    config,
    poll_id: str | None,
    polling,
    district,
    regional,
    precinct_csv: Path,
) -> None:
    """
    Append this run to projection_history.json in the main output dir.
    Also copies the precinct CSV to snapshots/ with a timestamp so every
    run's full precinct-level detail is preserved.
    """
    main_out     = CONFIG.output_dir
    history_path = main_out / "projection_history.json"
    snapshots    = main_out / "snapshots"
    snapshots.mkdir(exist_ok=True)

    ts      = datetime.now()
    ts_slug = ts.strftime("%Y%m%dT%H%M%S")

    # Save a timestamped precinct snapshot (preserves full precinct detail per run)
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


def _save(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chicago Mayor 2027 pipeline")
    parser.add_argument("--early-votes", action="store_true",
                        help="Include Chicago BOE early vote estimates")
    parser.add_argument("--poll-id", metavar="ID",
                        help="Run with a single poll instead of the weighted aggregate. "
                             "Use --list-polls to see available IDs.")
    parser.add_argument("--list-polls", action="store_true",
                        help="Print available poll IDs and exit")
    args = parser.parse_args()

    if args.list_polls:
        print("Available poll IDs:")
        for p in CONFIG.polls:
            print(f"  {get_poll_id(p):<36}  {p['pollster_name']}  ({p['field_end']})")
        sys.exit(0)

    main(include_early_votes=args.early_votes, poll_id=args.poll_id)
