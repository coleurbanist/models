"""
IL-09 2026 pipeline orchestrator.

Runs stages 1–5 and writes JSON outputs for consumption by the display project
(elections/ilforecast_redesign). Does not render HTML directly.

    python -m races.il09_2026.run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from races.il09_2026.race_config import CONFIG
from core.poll_weighting import (
    aggregate_polls,
    run_district_simulation,
    build_versioned_history,
)
from core.precinct_pipeline import run_precinct_pipeline
from core.regional_forecast import generate_regional_forecast


def main() -> None:
    print(f"[IL-09 2026] {CONFIG.race_label}")

    print("  Stage 1: Aggregating polls...")
    polling = aggregate_polls(CONFIG)

    print(f"  Stage 2: District simulation ({CONFIG.n_sim_district:,} trials)...")
    district = run_district_simulation(CONFIG, polling)
    for cand in CONFIG.candidates:
        print(f"    {cand:16s}  win={district['win_probs'][cand]:.1%}  "
              f"median={district['median_vote_shares'][cand]:.1%}")

    print(f"  Stage 3: Precinct simulation ({CONFIG.n_sim_precinct:,} trials)...")
    precincts = run_precinct_pipeline(CONFIG, polling, district)
    print(f"    {len(precincts)} precincts")

    print("  Stage 4: Regional forecast...")
    regional = generate_regional_forecast(CONFIG, precincts)

    print("  Stage 5: Writing outputs...")
    _write_outputs(polling, district, precincts, regional)
    print(f"  Done → {CONFIG.output_dir}")


def _write_outputs(polling, district, precincts, regional) -> None:
    out = CONFIG.output_dir
    out.mkdir(parents=True, exist_ok=True)

    _save(out / "poll_baseline.json", {
        "current": polling,
        "last_run": polling["as_of"],
        "n_simulations": CONFIG.n_sim_district,
    })
    _save(out / "district_win_probabilities.json", district)
    _save(out / "regional_vote_forecast.json", regional)

    # Precinct CSV — the display project reads this for map rendering
    precincts.to_csv(str(out / f"{CONFIG.race_id}_precinct_probabilities.csv"), index=False)


def _save(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
