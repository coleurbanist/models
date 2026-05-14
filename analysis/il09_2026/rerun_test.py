"""
IL-09 2026 model comparison: old assumptions vs new assumptions vs actual results.

Runs the Rust simulator twice with the same poll baseline:

  OLD: Single bloc, no within-bloc competition, no environment shock,
       undecideds pre-allocated (baked into the baseline).

  NEW: Correct ideological blocs (Moderates/Progressives), within-bloc
       competition shocks, environment shock, Dirichlet undecided sampling
       with calibrated viability weights.

Compares each model's predicted vote shares to actual district results
and reports the improvement.
"""

import json
import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
SIM_BIN     = REPO_ROOT / "simulator" / "target" / "release" / "simulator"
CALIB_FILE  = Path(__file__).parent / "calibration_params.json"

# ── Candidates ────────────────────────────────────────────────────────────────
CANDIDATES = ["Fine", "Biss", "Abughazaleh", "Simmons", "Amiwala", "Andrew", "Huynh"]

# ── Poll baseline (PPP/RoundTable March 9-10 2026, Wave 2) ────────────────────
# raw_poll = decided voters only (sum < 100 because ~18pp are undecided)
RAW_POLL = {
    "Fine":        15.008,
    "Biss":        24.220,
    "Abughazaleh": 18.819,
    "Simmons":      8.623,
    "Amiwala":      5.254,
    "Andrew":       6.346,
    "Huynh":        1.448,
}
UNDECIDED_PCT = 18.37  # percentage points of undecideds

# median_forecast = poll baseline with undecideds pre-allocated (old approach)
MEDIAN_FORECAST = {
    "Fine":        17.673,
    "Biss":        29.481,
    "Abughazaleh": 23.479,
    "Simmons":     10.540,
    "Amiwala":      6.896,
    "Andrew":       7.750,
    "Huynh":        2.118,
}

MOE_DISTRICT = 4.4  # percentage points (±2σ interval)

# ── Second-choice matrix (PPP/RoundTable Wave 2, March 2026) ──────────────────
# Values are percentages; rows are the "donor" candidate, columns are recipients.
# "no_second" and "others" are excluded — only the 7 tracked candidates.
SC_MATRIX_RAW = {
    "Biss":        {"Fine": 23, "Abughazaleh": 21, "Simmons": 13, "Amiwala": 10, "Andrew": 8,  "Huynh": 2},
    "Abughazaleh": {"Fine":  6, "Biss":        24, "Simmons": 20, "Amiwala": 22, "Andrew": 4,  "Huynh": 5},
    "Fine":        {"Biss":  39,"Abughazaleh": 14, "Simmons":  6, "Amiwala":  5, "Andrew": 11, "Huynh": 2},
    "Simmons":     {"Biss":  24,"Abughazaleh":  5, "Fine":     7, "Amiwala": 21, "Andrew": 10, "Huynh": 7},
    "Amiwala":     {"Biss":  17,"Abughazaleh": 37, "Fine":     2, "Simmons": 17, "Andrew": 15, "Huynh": 3},
    "Andrew":      {"Biss":  17,"Abughazaleh": 10, "Fine":    16, "Simmons": 10, "Amiwala": 15,"Huynh": 3},
    "Huynh":       {"Biss":  29,"Abughazaleh": 14, "Fine":    26, "Simmons": 23, "Amiwala":  0,"Andrew": 0},
}

def normalize_sc_matrix(raw):
    """Normalize each row to sum to 1.0 (over just the 7 tracked candidates)."""
    out = {}
    for donor, row in raw.items():
        total = sum(row.values())
        out[donor] = {recip: v / total for recip, v in row.items()} if total > 0 else row
    return out

SC_MATRIX = normalize_sc_matrix(SC_MATRIX_RAW)

# ── Actual district results (among the 7 tracked candidates only) ──────────────
# Source: illinois_elections.db, primary 2026, us_house district 9.
# Raw votes: Biss 38804, Abughazaleh 34707, Fine 26384, Simmons 9419, Andrew 7997,
#            Amiwala 6692, Huynh 2343 → total 126346
ACTUAL = {
    "Biss":        0.3071,
    "Abughazaleh": 0.2747,
    "Fine":        0.2088,
    "Simmons":     0.0745,
    "Andrew":      0.0633,
    "Amiwala":     0.0530,
    "Huynh":       0.0185,
}

# ── Undecided allocation weights ──────────────────────────────────────────────
#
# PRE-ELECTION (used in build_new_payload):
#   Proportional to poll share among decided voters.  No ex-post knowledge of who
#   actually absorbed the undecideds.  Normalized to [0,1] so that with
#   concentration=3.0 the Dirichlet sum-of-alphas = 3.0, giving meaningful
#   trial-to-trial variation in how the 18pp undecided pool is split.
#
# POST-MORTEM (kept for reference / calibration analysis):
#   Derived from actual results — confirms top-tier absorbed most undecideds.
#   Do NOT use pre-election; this is hindsight.
_poll_total = sum(RAW_POLL.values())
PRE_ELECTION_UNDECIDED_WEIGHTS = {c: v / _poll_total for c, v in RAW_POLL.items()}

CALIBRATED_UNDECIDED_WEIGHTS = {   # post-mortem only
    "Fine":        0.30,
    "Biss":        3.571,
    "Abughazaleh": 3.074,
    "Simmons":     0.30,
    "Amiwala":     0.30,
    "Andrew":      0.355,
    "Huynh":       0.30,
}

# ── Helper: normalize a dict to sum-to-1 fractions ────────────────────────────
def to_fractions(pct_dict):
    total = sum(pct_dict.values())
    return {k: v / total for k, v in pct_dict.items()}

# ── Helper: run the Rust simulator with a given payload ───────────────────────
def run_sim(payload: dict) -> dict:
    raw = json.dumps(payload).encode()
    result = subprocess.run(
        [str(SIM_BIN)],
        input=raw,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Simulator exited {result.returncode}:\n{result.stderr.decode()}")
    return json.loads(result.stdout)

# ── Helper: compute MAE between predicted and actual shares ───────────────────
def mae(predicted: dict, actual: dict) -> float:
    return sum(abs(predicted[c] - actual[c]) for c in CANDIDATES) / len(CANDIDATES)

# ── Helper: print a comparison table ─────────────────────────────────────────
def print_table(label: str, predicted: dict, actual: dict):
    print(f"\n  {label}")
    print(f"  {'Candidate':<16} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
    print(f"  {'-'*50}")
    for c in sorted(CANDIDATES, key=lambda x: -actual[x]):
        pred_pct = predicted[c] * 100
        act_pct  = actual[c] * 100
        err      = pred_pct - act_pct
        print(f"  {c:<16} {pred_pct:>9.1f}% {act_pct:>9.1f}% {err:>+9.1f}pp")
    print(f"  {'MAE':<16} {mae(predicted, actual)*100:>9.2f}pp")

# ─────────────────────────────────────────────────────────────────────────────
# OLD MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Mirrors the final pre-election run (March 14, 2026):
#   - Baseline = median_forecast (undecideds already baked in)
#   - No ideological blocs (every candidate is in its own singleton)
#   - No within-bloc competition shocks
#   - No environment shock
#   - Second-choice constraint ON (was active in original run)
def build_old_payload(n_sim: int) -> dict:
    baseline = to_fractions(MEDIAN_FORECAST)
    return {
        "mode":                    "district",
        "n_simulations":           n_sim,
        "candidates":              CANDIDATES,
        "baseline":                baseline,
        "moe_district":            MOE_DISTRICT,
        "ideological_blocs":       [],          # no correlated bloc shocks
        "second_choice_matrix":    SC_MATRIX,
        "second_choice_strength":  0.60,
        # all new features default to disabled (0 / false)
        "undecided_share":         0.0,
        "sigma_within_bloc_fraction": 0.0,
        "environment_shock_fraction": 0.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# NEW MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Incorporates four improvements from the IL-09 2026 post-mortem:
#   1. Correct ideological blocs (Biss with Progressives, not Moderates)
#   2. Within-bloc competition shocks (one progressive gaining = others losing)
#   3. Environment shock (systematic polling miss affecting all candidates)
#   4. Dirichlet undecided sampling with poll-share proportional weights
#
# fundamental_sigma=9.0pp:
#   Sampling MOE captures ~2.2pp of error; the rest is non-sampling (late
#   movement, turnout composition, structural bias).  Historical IL primary
#   RMSE ≈ 5–8pp per candidate.  At 9pp independent sigma with poll-share
#   undecided weights, Biss win prob ≈ 65%, matching the election-day model.
def build_new_payload(n_sim: int, fundamental_sigma: float = 9.0) -> dict:
    # Use decided-only baseline (raw poll numbers); simulator adds undecideds
    decided_baseline = to_fractions(RAW_POLL)
    return {
        "mode":                    "district",
        "n_simulations":           n_sim,
        "candidates":              CANDIDATES,
        "baseline":                decided_baseline,
        "moe_district":            MOE_DISTRICT,
        # Correct blocs: Fine/Andrew competed for moderate voters;
        # Biss/Abughazaleh/Simmons/Amiwala/Huynh competed for progressive voters.
        "ideological_blocs": [
            ["Fine", "Andrew"],
            ["Biss", "Abughazaleh", "Simmons", "Amiwala", "Huynh"],
        ],
        "second_choice_matrix":    SC_MATRIX,
        "second_choice_strength":  0.60,
        # Within-bloc competition: half as large as the district-level shock
        "sigma_within_bloc_fraction": 0.5,
        # Environment shock: 30% of district noise — models systematic polling miss
        "environment_shock_fraction": 0.30,
        # Independent per-candidate noise: non-sampling error (late movement,
        # turnout surprises, structural bias not captured by polling MOE).
        "fundamental_uncertainty_sigma": fundamental_sigma,
        # Dirichlet undecided sampling — proportional to poll share (pre-election).
        # concentration=3.0 → sum-of-alphas=3.0, giving meaningful per-trial variance.
        "undecided_share":         UNDECIDED_PCT,
        "undecided_weights":       PRE_ELECTION_UNDECIDED_WEIGHTS,
        "undecided_concentration": 3.0,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not SIM_BIN.exists():
        sys.exit(f"Simulator binary not found at {SIM_BIN}. Run: cargo build --release")

    N_SIM = 500_000
    print(f"Running {N_SIM:,} simulations per model …\n")

    # Run old model
    print("  [1/3] Old model (no blocs, pre-allocated undecideds) …", flush=True)
    old_result = run_sim(build_old_payload(N_SIM))
    old_shares = old_result["median_vote_shares"]

    # Run new model without fundamental uncertainty (baseline comparison)
    print("  [2/3] New model (correct blocs + Dirichlet, no fundamental uncertainty) …", flush=True)
    new_result = run_sim(build_new_payload(N_SIM, fundamental_sigma=0.0))
    new_shares = new_result["median_vote_shares"]

    # Run new model with fundamental uncertainty calibrated to IL primary history.
    # 9pp sigma + poll-share undecided weights → Biss ≈ 65%, matching election-day model.
    print("  [3/3] New model + fundamental uncertainty (9.0pp independent sigma) …", flush=True)
    full_result = run_sim(build_new_payload(N_SIM, fundamental_sigma=9.0))
    full_shares = full_result["median_vote_shares"]

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  IL-09 2026 — MODEL COMPARISON")
    print("═" * 60)

    print_table("OLD MODEL (pre-election assumptions)", old_shares, ACTUAL)
    print_table("NEW MODEL (blocs + Dirichlet, no fund. uncertainty)", new_shares, ACTUAL)
    print_table("NEW MODEL + fundamental uncertainty (9pp)", full_shares, ACTUAL)

    old_mae  = mae(old_shares,  ACTUAL) * 100
    new_mae  = mae(new_shares,  ACTUAL) * 100
    full_mae = mae(full_shares, ACTUAL) * 100
    print(f"\n{'─'*60}")
    print(f"  MAE: old={old_mae:.2f}pp  new(no fund.)={new_mae:.2f}pp  new+fund.={full_mae:.2f}pp")
    print(f"{'─'*60}")

    # ── Win probabilities ─────────────────────────────────────────────────────
    print("\n  WIN PROBABILITIES")
    print(f"  {'Candidate':<16} {'Old':>10} {'New(tight)':>12} {'New+9pp':>11}  {'Actual winner':>5}")
    print(f"  {'-'*56}")
    actual_winner = max(ACTUAL, key=ACTUAL.get)
    for c in sorted(CANDIDATES, key=lambda x: -ACTUAL[x]):
        old_wp  = old_result["win_probs"].get(c, 0.0) * 100
        new_wp  = new_result["win_probs"].get(c, 0.0) * 100
        full_wp = full_result["win_probs"].get(c, 0.0) * 100
        marker  = " ← won" if c == actual_winner else ""
        print(f"  {c:<16} {old_wp:>9.1f}% {new_wp:>11.1f}% {full_wp:>10.1f}%{marker}")

    # ── Uncertainty ranges ────────────────────────────────────────────────────
    print("\n  NEW MODEL + 9pp FUNDAMENTAL UNCERTAINTY — P05/MEDIAN/P95")
    print(f"  {'Candidate':<16} {'P5':>8} {'Median':>8} {'P95':>8} {'Actual':>8}")
    print(f"  {'-'*50}")
    for c in sorted(CANDIDATES, key=lambda x: -ACTUAL[x]):
        p05 = full_result["p05_vote_shares"].get(c, 0.0) * 100
        med = full_result["median_vote_shares"].get(c, 0.0) * 100
        p95 = full_result["p95_vote_shares"].get(c, 0.0) * 100
        act = ACTUAL[c] * 100
        in_range = "✓" if p05 <= act <= p95 else "✗"
        print(f"  {c:<16} {p05:>7.1f}% {med:>7.1f}% {p95:>7.1f}% {act:>7.1f}% {in_range}")

    # ── Key insights ──────────────────────────────────────────────────────────
    print("\n  KEY MODELING NOTES")
    print("  • Systematic biases NOT addressed here (require precinct-level fixes):")
    print("    – Fine severely underestimated in Ward 50 / Niles Orthodox Jewish precincts")
    print("    – Andrew severely underestimated in Cook 8100 series (Morton Grove / Lincolnwood)")
    print("    – Amiwala underestimated in Niles Township South Asian precincts")
    print("    These require extra_constraints in the precinct pipeline, not district-level tuning.")
    print("  • Undecided allocation: calibrated weights confirm top-tier absorbed ~316% of")
    print("    predicted undecideds; Simmons/Huynh got essentially none.")
    print("  • Bloc correlation confirmed: Fine↔Abughazaleh (r=−0.61) opposing blocs;")
    print("    Abughazaleh↔Simmons (r=+0.32) co-moving within progressive bloc.")


if __name__ == "__main__":
    main()
