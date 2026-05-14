"""
Thin wrapper that calls the compiled Rust simulator binary via subprocess.
Python assembles the JSON payload; Rust does the Monte Carlo math; Python
reads the JSON results.

Build the binary first:
    cd simulator && cargo build --release
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Root of the elections-models repo (parent of core/)
_REPO_ROOT = Path(__file__).parent.parent

# Cross-platform binary path
_BIN_NAME = "simulator.exe" if sys.platform == "win32" else "simulator"
_BINARY = _REPO_ROOT / "simulator" / "target" / "release" / _BIN_NAME


def _run(payload: dict) -> dict:
    if not _BINARY.exists():
        raise FileNotFoundError(
            f"Simulator binary not found at {_BINARY}.\n"
            f"Build it with:  cd {_REPO_ROOT / 'simulator'} && cargo build --release"
        )
    result = subprocess.run(
        [str(_BINARY)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Simulator exited with code {result.returncode}.\n"
            f"stderr: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def run_district_sim(
    *,
    n_simulations: int,
    candidates: list[str],
    baseline: dict[str, float],
    moe_district: float,
    ideological_blocs: list[list[str]],
    second_choice_matrix: dict[str, dict[str, float]],
    second_choice_strength: float,
    favorability_weights: dict[str, float] | None = None,
    has_runoff: bool = False,
    runoff_threshold: float = 0.50,
    undecided_share: float = 0.0,
    undecided_weights: dict[str, float] | None = None,
    undecided_concentration: float = 3.0,
    fundamental_uncertainty_sigma: float = 0.0,
    sigma_within_bloc_fraction: float = 0.0,
    environment_shock_fraction: float = 0.0,
) -> dict:
    """
    Run the district-level Monte Carlo simulation.

    Returns:
        {
            "win_probs":           {candidate: float},
            "outright_win_probs":  {candidate: float} | None,
            "runoff_probs":        {candidate: float} | None,
            "advance_probs":       {candidate: float} | None,
            "prob_no_runoff":      float | None,
            "mean_vote_shares":    {candidate: float},
            "median_vote_shares":  {candidate: float},
            "p05_vote_shares":     {candidate: float},
            "p95_vote_shares":     {candidate: float},
        }
    runoff_probs, advance_probs, outright_win_probs, and prob_no_runoff are
    None when has_runoff=False.

    undecided_share: percentage points of undecided voters (e.g. 30.0 for 30%).
      When > 0, baseline should be the decided-only shares and undecided_weights
      controls how undecideds are distributed via Dirichlet sampling.
    """
    payload: dict = {
        "mode": "district",
        "n_simulations": n_simulations,
        "candidates": candidates,
        "baseline": baseline,
        "moe_district": moe_district,
        "ideological_blocs": ideological_blocs,
        "second_choice_matrix": second_choice_matrix,
        "second_choice_strength": second_choice_strength,
        "has_runoff": has_runoff,
        "runoff_threshold": runoff_threshold,
        "undecided_share": undecided_share,
        "undecided_concentration": undecided_concentration,
        "fundamental_uncertainty_sigma": fundamental_uncertainty_sigma,
        "sigma_within_bloc_fraction": sigma_within_bloc_fraction,
        "environment_shock_fraction": environment_shock_fraction,
    }
    if favorability_weights:
        payload["favorability_weights"] = favorability_weights
    if undecided_weights:
        payload["undecided_weights"] = undecided_weights
    return _run(payload)


def run_precinct_sim(
    *,
    n_simulations: int,
    candidates: list[str],
    moe_district: float,
    moe_precinct: float,
    ideological_blocs: list[list[str]],
    precincts: list[dict],
) -> dict:
    """
    Run the precinct-level Monte Carlo simulation.

    Each entry in `precincts` must have:
        {"id": str, "baseline": {candidate: float}, "turnout_weight": int}

    Returns:
        {
            "precincts": [
                {
                    "id": str,
                    "win_probs":    {candidate: float},
                    "median_pcts":  {candidate: float},
                    "median_votes": {candidate: float},
                },
                ...
            ]
        }
    """
    payload = {
        "mode": "precinct",
        "n_simulations": n_simulations,
        "candidates": candidates,
        "moe_district": moe_district,
        "moe_precinct": moe_precinct,
        "ideological_blocs": ideological_blocs,
        "precincts": precincts,
    }
    return _run(payload)
