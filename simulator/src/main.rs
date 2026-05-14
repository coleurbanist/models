// The simulator is a standalone program that Python calls as a subprocess.
// Python sends a JSON blob to its stdin, it runs the math, and prints JSON to stdout.
// It never reads files or talks to a database — all data comes in through stdin.

use std::collections::HashMap;
use std::io::{self, Read};

use rand::SeedableRng;
use rand_distr::{Dirichlet, Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ─── Default value functions ───────────────────────────────────────────────
// serde needs named functions (not closures) for default field values.

fn default_undecided_concentration() -> f64 { 3.0 }
fn default_runoff_threshold() -> f64 { 0.5 }

// ─── Input / output types ──────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum Request {
    District(DistrictRequest),
    Precinct(PrecinctRequest),
}

#[derive(Deserialize)]
struct DistrictRequest {
    n_simulations:       usize,
    candidates:          Vec<String>,

    // Poll-derived vote share per candidate (0.0–1.0).
    // If undecided_share > 0, this should be the DECIDED-only baseline
    // (i.e. the raw poll numbers normalized to just the decided voters).
    // If undecided_share = 0 (default), this is the full baseline including
    // pre-allocated undecideds, matching the original behavior.
    baseline:            HashMap<String, f64>,

    // ── Undecided Dirichlet sampling ────────────────────────────────────────
    // When undecided_share > 0, each simulation trial draws a fresh undecided
    // allocation from a Dirichlet distribution rather than using a fixed split.
    // This models the uncertainty in WHERE undecideds will land, not just
    // whether the polling baseline is correct.
    //
    // undecided_share:       total undecided percentage points (e.g. 8.0)
    // undecided_weights:     relative weight for each candidate claiming undecideds
    //                        (higher = gets more; calibrate from historical races)
    // undecided_concentration: Dirichlet alpha multiplier — higher = tighter
    //                        distribution, less trial-to-trial variance in allocation.
    //                        3.0 = moderate variance; 1.0 = high variance.
    #[serde(default)]
    undecided_share:          f64,
    #[serde(default)]
    undecided_weights:        HashMap<String, f64>,
    #[serde(default = "default_undecided_concentration")]
    undecided_concentration:  f64,

    moe_district:        f64,

    // ── Two-level bloc shocks ───────────────────────────────────────────────
    // Candidates in the same ideological bloc share a district-level shock
    // (they move together when the overall environment favors their bloc).
    // On TOP of that, within-bloc competition shocks redistribute share between
    // candidates IN the same bloc — a zero-sum perturbation so one progressive
    // gaining means other progressives losing, even on a good progressive night.
    //
    // sigma_within_bloc_fraction: within-bloc sigma as a fraction of sigma_district.
    //   0.0 = disabled (old behavior: all bloc-mates move perfectly together)
    //   0.5 = within-bloc competition is half as large as the district-level noise
    #[serde(default)]
    sigma_within_bloc_fraction: f64,

    // ── Environment shock ───────────────────────────────────────────────────
    // A single shock applied equally to ALL candidates, modeling systematic
    // polling error (e.g. the sample skewed younger/more progressive than actual
    // turnout). This widens the tails of all distributions appropriately.
    //
    // environment_shock_fraction: env sigma as a fraction of sigma_district.
    //   0.0 = disabled (old behavior)
    //   0.3 = environment shock is 30% as large as district noise
    #[serde(default)]
    environment_shock_fraction: f64,

    // ── Fundamental uncertainty ─────────────────────────────────────────────
    // Independent per-candidate noise representing sources of uncertainty
    // beyond sampling error: late momentum shifts, turnout composition errors,
    // structural polling bias specific to individual candidates.
    //
    // Unlike bloc shocks (which are shared) or the environment shock (which
    // affects everyone equally), this is fully independent — Candidate A could
    // run 6pp above expectations while Candidate B runs 4pp below, with no
    // correlation between them.
    //
    // In percentage points (same units as moe_district).
    // Historical Illinois primary RMSE ≈ 5–6pp per candidate, so a value in
    // that range is appropriate when moe_district already covers sampling error.
    //   0.0   = disabled (default — pure sampling-error model)
    //   5.0   = 5pp independent sigma; matches old IL-09 model volatility
    #[serde(default)]
    fundamental_uncertainty_sigma: f64,

    ideological_blocs:   Vec<Vec<String>>,
    second_choice_matrix: HashMap<String, HashMap<String, f64>>,
    second_choice_strength: f64,

    // ── Runoff tracking ─────────────────────────────────────────────────────
    // When has_runoff=true, tracks top-two finishers separately from outright
    // winners.  Outputs runoff_probs (probability of being in top-two when no
    // one clears the threshold) and advance_probs (win outright OR make runoff).
    #[serde(default)]
    has_runoff:          bool,
    #[serde(default = "default_runoff_threshold")]
    runoff_threshold:    f64,

    #[serde(default)]
    #[allow(dead_code)]
    favorability_weights: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct PrecinctRequest {
    n_simulations:       usize,
    candidates:          Vec<String>,
    moe_district:        f64,
    moe_precinct:        f64,
    #[serde(default)]
    sigma_within_bloc_fraction: f64,
    #[serde(default)]
    environment_shock_fraction: f64,
    #[serde(default)]
    fundamental_uncertainty_sigma: f64,
    ideological_blocs:   Vec<Vec<String>>,
    precincts:           Vec<PrecinctInput>,
}

#[derive(Deserialize)]
struct PrecinctInput {
    id:              String,
    baseline:        HashMap<String, f64>,
    turnout_weight:  f64,
}

#[derive(Serialize)]
struct DistrictOutput {
    // Probability of finishing with the plurality (top vote-getter) in round 1.
    win_probs:               HashMap<String, f64>,
    // Probability of clearing the runoff threshold outright (>50%).
    // None when has_runoff=false.
    outright_win_probs:      Option<HashMap<String, f64>>,
    // Probability of being in top-2 when no outright winner (conditional on runoff).
    // None when has_runoff=false.
    runoff_probs:            Option<HashMap<String, f64>>,
    // outright_win_prob + P(in top-2 when runoff): unconditional advance probability.
    // None when has_runoff=false.
    advance_probs:           Option<HashMap<String, f64>>,
    // Probability that any candidate wins outright (no runoff needed).
    // None when has_runoff=false.
    prob_no_runoff:          Option<f64>,
    // Central tendency and spread of vote share distribution.
    mean_vote_shares:        HashMap<String, f64>,
    median_vote_shares:      HashMap<String, f64>,
    p05_vote_shares:         HashMap<String, f64>,
    p95_vote_shares:         HashMap<String, f64>,
}

#[derive(Serialize)]
struct PrecinctOutput {
    precincts: Vec<PrecinctResult>,
}

#[derive(Serialize)]
struct PrecinctResult {
    id:           String,
    win_probs:    HashMap<String, f64>,
    median_pcts:  HashMap<String, f64>,
    median_votes: HashMap<String, f64>,
}

// ─── Helper functions ──────────────────────────────────────────────────────

// After adding random noise, vote shares can go negative or not sum to 1.
// Negatives become 0, then everything is scaled so the total is exactly 1.
// If somehow everything is 0, falls back to equal shares.
fn normalize(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 1e-12 {
        for x in v.iter_mut() { *x /= s; }
    } else {
        let n = v.len() as f64;
        for x in v.iter_mut() { *x = 1.0 / n; }
    }
}

// Returns the index of the largest value — used to find the trial winner.
fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// Returns the indices that would sort v in descending order.
fn argsort_desc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 { (v[n/2-1] + v[n/2]) / 2.0 } else { v[n/2] }
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return 0.0; }
    let idx = ((p / 100.0) * (n - 1) as f64).round() as usize;
    v[idx.min(n - 1)]
}

// Builds a per-candidate lookup: which bloc index does this candidate belong to?
// Named blocs get indices 0..n_blocs.  Unassigned candidates each get a unique
// singleton index so they move independently.
fn build_bloc_map(candidates: &[String], blocs: &[Vec<String>]) -> Vec<usize> {
    let mut bloc_map = vec![usize::MAX; candidates.len()];
    for (bi, bloc) in blocs.iter().enumerate() {
        for name in bloc {
            if let Some(ci) = candidates.iter().position(|c| c == name) {
                bloc_map[ci] = bi;
            }
        }
    }
    let mut next = blocs.len();
    for b in bloc_map.iter_mut() {
        if *b == usize::MAX { *b = next; next += 1; }
    }
    bloc_map
}

// ─── Shared simulation logic ───────────────────────────────────────────────
//
// run_one_trial applies all noise sources to a baseline and returns vote shares.
// Used by both district and precinct modes to avoid code duplication.
//
// Parameters
// ──────────
// baseline:           per-candidate decided-voter baseline (fraction, sums to ≤1)
// undecided_extra:    per-candidate undecided addition for this trial (fraction)
// env_shock:          single shock added to all candidates equally
// bloc_shocks:        one shock per bloc (index by bloc_map[ci])
// within_shocks:      zero-sum within-bloc shocks (mean-centered per bloc)
// independent_shocks: fully independent per-candidate shock (fundamental uncertainty)
// sc_matrix:          second-choice matrix [from_cand][to_cand]
// sc_strength:        fraction of deficit routed via second-choice
fn run_one_trial(
    baseline:          &[f64],
    undecided_extra:   &[f64],
    env_shock:         f64,
    bloc_shocks:       &[f64],
    within_shocks:     &[f64],
    independent_shocks: &[f64],
    bloc_map:          &[usize],
    sc_matrix:         &[Vec<f64>],
    sc_strength:       f64,
) -> Vec<f64> {
    let n = baseline.len();

    // Apply all noise sources.  .max(0) prevents negative shares before normalization.
    let mut shares: Vec<f64> = baseline
        .iter()
        .enumerate()
        .map(|(ci, &b)| {
            let shock = env_shock
                + bloc_shocks[bloc_map[ci]]
                + within_shocks[ci]
                + independent_shocks[ci];
            (b + undecided_extra[ci] + shock).max(0.0)
        })
        .collect();

    // Second-choice redistribution: when a candidate falls below baseline,
    // route sc_strength × deficit through their second-choice row.
    for ci in 0..n {
        let deficit = (baseline[ci] + undecided_extra[ci]) - shares[ci];
        if deficit > 0.0 {
            let routed = deficit * sc_strength;
            let row_sum: f64 = sc_matrix[ci].iter().sum();
            if row_sum > 1e-12 {
                for cj in 0..n {
                    shares[cj] += routed * sc_matrix[ci][cj] / row_sum;
                }
            }
        }
    }

    normalize(&mut shares);
    shares
}

// Build zero-sum within-bloc competition shocks.
// Each candidate gets a raw draw from N(0, sigma_within), then the mean
// within their bloc is subtracted so the total within-bloc shock sums to zero.
// Candidates in singleton blocs get 0 (no intra-bloc competition to model).
fn sample_within_shocks(
    n_cands:     usize,
    bloc_map:    &[usize],
    n_bloc_slots: usize,
    sigma:       f64,
    rng:         &mut impl rand::Rng,
    normal:      &Normal<f64>,
) -> Vec<f64> {
    if sigma <= 0.0 {
        return vec![0.0; n_cands];
    }

    let mut raw: Vec<f64> = (0..n_cands)
        .map(|_| normal.sample(rng) * sigma)
        .collect();

    // Mean-center within each bloc
    let mut bloc_sum   = vec![0.0f64; n_bloc_slots];
    let mut bloc_count = vec![0usize; n_bloc_slots];
    for ci in 0..n_cands {
        bloc_sum[bloc_map[ci]]   += raw[ci];
        bloc_count[bloc_map[ci]] += 1;
    }
    for ci in 0..n_cands {
        let b = bloc_map[ci];
        if bloc_count[b] > 1 {
            raw[ci] -= bloc_sum[b] / bloc_count[b] as f64;
        } else {
            raw[ci] = 0.0; // singleton bloc — no within-competition
        }
    }
    raw
}

// ─── District simulation ──────────────────────────────────────────────────

fn run_district(req: DistrictRequest) -> DistrictOutput {
    let n_cands  = req.candidates.len();
    let n_blocs  = req.ideological_blocs.len().max(1);
    let bloc_map = build_bloc_map(&req.candidates, &req.ideological_blocs);
    let n_bloc_slots = n_blocs + n_cands; // covers named blocs + singleton slots

    let baseline: Vec<f64> = req.candidates.iter()
        .map(|c| *req.baseline.get(c).unwrap_or(&0.0))
        .collect();

    // sigma_district: moe is a ±2-sigma interval in pp → 1 sigma in fraction
    let sigma_district    = req.moe_district / 100.0 / 2.0;
    let sigma_within      = sigma_district * req.sigma_within_bloc_fraction;
    let sigma_env         = sigma_district * req.environment_shock_fraction;
    let sigma_fundamental = req.fundamental_uncertainty_sigma / 100.0;

    // Build second-choice matrix as indexed 2D Vec
    let sc_matrix: Vec<Vec<f64>> = req.candidates.iter()
        .map(|from| {
            let row = req.second_choice_matrix.get(from);
            req.candidates.iter()
                .map(|to| {
                    if from == to { 0.0 }
                    else { row.and_then(|r| r.get(to)).copied().unwrap_or(0.0) }
                })
                .collect()
        })
        .collect();

    // Build Dirichlet for undecided allocation (if enabled).
    // alpha[i] = undecided_weight[i] * concentration.
    // Clamp weights to 0.1 minimum so Dirichlet never degenerates.
    let undecided_alphas: Vec<f64> = if req.undecided_share > 0.0 {
        req.candidates.iter()
            .map(|c| {
                let w = req.undecided_weights.get(c).copied().unwrap_or(1.0).max(0.1);
                w * req.undecided_concentration
            })
            .collect()
    } else {
        vec![]
    };
    let maybe_dirichlet: Option<Dirichlet<f64>> = if req.undecided_share > 0.0 {
        Dirichlet::<f64>::new(&undecided_alphas).ok()
    } else {
        None
    };
    let undecided_fraction = req.undecided_share / 100.0;

    let n_sim = req.n_simulations;
    let chunk = (n_sim / rayon::current_num_threads()).max(1);

    // Each thread returns: (win_counts, runoff_counts, share_samples)
    // runoff_counts[ci] = times candidate was in top-two when no one won outright
    let results: Vec<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<Vec<f64>>)> =
        (0..rayon::current_num_threads())
            .into_par_iter()
            .map(|thread_id| {
                let mut rng = rand::rngs::SmallRng::seed_from_u64(
                    thread_id as u64 * 6364136223846793005,
                );
                let normal = Normal::new(0.0_f64, 1.0).unwrap();

                let start = thread_id * chunk;
                let end = if thread_id == rayon::current_num_threads() - 1 {
                    n_sim
                } else {
                    start + chunk
                };
                let local_n = end - start;

                let mut win_counts    = vec![0u64; n_cands];
                let mut runoff_counts = vec![0u64; n_cands]; // top-two when no outright winner
                let mut outright_counts = vec![0u64; n_cands]; // won with > threshold
                let mut share_samples: Vec<Vec<f64>> =
                    vec![Vec::with_capacity(local_n); n_cands];

                for _ in 0..local_n {
                    // ── Noise: environment shock (same for all candidates) ──
                    let env_shock = if sigma_env > 0.0 {
                        normal.sample(&mut rng) * sigma_env
                    } else {
                        0.0
                    };

                    // ── Noise: district-level bloc shocks ──────────────────
                    let bloc_shocks: Vec<f64> = (0..n_bloc_slots)
                        .map(|_| normal.sample(&mut rng) * sigma_district)
                        .collect();

                    // ── Noise: within-bloc competition shocks (zero-sum) ───
                    let within_shocks = sample_within_shocks(
                        n_cands, &bloc_map, n_bloc_slots, sigma_within,
                        &mut rng, &normal,
                    );

                    // ── Noise: independent per-candidate (fundamental uncertainty) ──
                    let independent_shocks: Vec<f64> = if sigma_fundamental > 0.0 {
                        (0..n_cands)
                            .map(|_| normal.sample(&mut rng) * sigma_fundamental)
                            .collect()
                    } else {
                        vec![0.0; n_cands]
                    };

                    // ── Undecided allocation (Dirichlet per trial) ─────────
                    let undecided_extra: Vec<f64> = if let Some(ref d) = maybe_dirichlet {
                        let alloc: Vec<f64> = d.sample(&mut rng);
                        alloc.iter().map(|&a| a * undecided_fraction).collect()
                    } else {
                        vec![0.0; n_cands]
                    };

                    // ── Simulate trial ─────────────────────────────────────
                    let shares = run_one_trial(
                        &baseline, &undecided_extra,
                        env_shock, &bloc_shocks, &within_shocks, &independent_shocks,
                        &bloc_map, &sc_matrix, req.second_choice_strength,
                    );

                    // ── Record results ─────────────────────────────────────
                    let winner = argmax(&shares);
                    win_counts[winner] += 1;

                    if req.has_runoff {
                        if shares[winner] >= req.runoff_threshold {
                            // Outright winner — no runoff needed
                            outright_counts[winner] += 1;
                        } else {
                            // No outright winner — top-two advance to runoff
                            let ranked = argsort_desc(&shares);
                            runoff_counts[ranked[0]] += 1;
                            runoff_counts[ranked[1]] += 1;
                        }
                    }

                    for (ci, &s) in shares.iter().enumerate() {
                        share_samples[ci].push(s);
                    }
                }

                (win_counts, runoff_counts, outright_counts, share_samples)
            })
            .collect();

    // Combine results from all threads
    let mut total_wins    = vec![0u64; n_cands];
    let mut total_runoff  = vec![0u64; n_cands];
    let mut total_outright = vec![0u64; n_cands];
    let mut all_shares: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sim); n_cands];

    for (wins, runoff, outright, shares) in results {
        for ci in 0..n_cands {
            total_wins[ci]     += wins[ci];
            total_runoff[ci]   += runoff[ci];
            total_outright[ci] += outright[ci];
            all_shares[ci].extend_from_slice(&shares[ci]);
        }
    }

    let n = n_sim as f64;

    let win_probs: HashMap<String, f64> = req.candidates.iter().enumerate()
        .map(|(ci, name)| (name.clone(), total_wins[ci] as f64 / n))
        .collect();

    let mean_vote_shares: HashMap<String, f64> = req.candidates.iter().enumerate()
        .map(|(ci, name)| {
            let mean = all_shares[ci].iter().sum::<f64>() / n;
            (name.clone(), mean)
        })
        .collect();

    let median_vote_shares: HashMap<String, f64> = req.candidates.iter().enumerate()
        .map(|(ci, name)| (name.clone(), median(all_shares[ci].clone())))
        .collect();

    let p05: HashMap<String, f64> = req.candidates.iter().enumerate()
        .map(|(ci, name)| (name.clone(), percentile(all_shares[ci].clone(), 5.0)))
        .collect();

    let p95: HashMap<String, f64> = req.candidates.iter().enumerate()
        .map(|(ci, name)| (name.clone(), percentile(all_shares[ci].clone(), 95.0)))
        .collect();

    // Runoff outputs — only populated when has_runoff=true
    let (outright_win_probs, runoff_probs, advance_probs, prob_no_runoff) = if req.has_runoff {
        // n_runoff_trials: trials where no one cleared the threshold.
        // Each such trial increments exactly 2 candidates' runoff_counts.
        let n_runoff_trials = total_runoff.iter().sum::<u64>() as f64 / 2.0;

        let owp: HashMap<String, f64> = req.candidates.iter().enumerate()
            .map(|(ci, name)| (name.clone(), total_outright[ci] as f64 / n))
            .collect();

        let rp: HashMap<String, f64> = req.candidates.iter().enumerate()
            .map(|(ci, name)| {
                let prob = if n_runoff_trials > 0.0 {
                    total_runoff[ci] as f64 / (n_runoff_trials * 2.0)
                } else { 0.0 };
                (name.clone(), prob)
            })
            .collect();

        let ap: HashMap<String, f64> = req.candidates.iter().enumerate()
            .map(|(ci, name)| {
                let outright = total_outright[ci] as f64 / n;
                let in_runoff = total_runoff[ci] as f64 / n;
                (name.clone(), outright + in_runoff)
            })
            .collect();

        let no_runoff = total_outright.iter().sum::<u64>() as f64 / n;

        (Some(owp), Some(rp), Some(ap), Some(no_runoff))
    } else {
        (None, None, None, None)
    };

    DistrictOutput {
        win_probs,
        outright_win_probs,
        runoff_probs,
        advance_probs,
        prob_no_runoff,
        mean_vote_shares,
        median_vote_shares,
        p05_vote_shares: p05,
        p95_vote_shares: p95,
    }
}

// ─── Precinct simulation ──────────────────────────────────────────────────

fn run_precinct(req: PrecinctRequest) -> PrecinctOutput {
    let n_cands  = req.candidates.len();
    let n_blocs  = req.ideological_blocs.len().max(1);
    let bloc_map = build_bloc_map(&req.candidates, &req.ideological_blocs);
    let n_bloc_slots = n_blocs + n_cands;

    let sigma_district    = req.moe_district / 100.0 / 2.0;
    let sigma_precinct    = req.moe_precinct / 100.0 / 2.0;
    let sigma_within      = sigma_district * req.sigma_within_bloc_fraction;
    let sigma_env         = sigma_district * req.environment_shock_fraction;
    let sigma_fundamental = req.fundamental_uncertainty_sigma / 100.0;

    // Precinct sim has no second-choice constraint (district-level effect only)
    let sc_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_cands]; n_cands];

    let precincts: Vec<PrecinctResult> = req.precincts
        .into_par_iter()
        .map(|precinct| {
            let baseline: Vec<f64> = req.candidates.iter()
                .map(|c| *precinct.baseline.get(c).unwrap_or(&0.0))
                .collect();

            let turnout = precinct.turnout_weight;
            let n_sim   = req.n_simulations;

            // Deterministic seed from JoinField hash — same input always gives same output
            let seed = precinct.id.bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
            let normal  = Normal::new(0.0_f64, 1.0).unwrap();

            let mut win_counts    = vec![0u64; n_cands];
            let mut share_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_sim); n_cands];

            for _ in 0..n_sim {
                // Environment shock — same for all candidates in this trial
                let env_shock = if sigma_env > 0.0 {
                    normal.sample(&mut rng) * sigma_env
                } else { 0.0 };

                // District-level bloc shocks (shared across all precincts in this trial
                // conceptually, but each precinct draws independently for simplicity)
                let bloc_shocks: Vec<f64> = (0..n_bloc_slots)
                    .map(|_| normal.sample(&mut rng) * sigma_district)
                    .collect();

                // Within-bloc competition shocks (zero-sum per bloc)
                let within_shocks = sample_within_shocks(
                    n_cands, &bloc_map, n_bloc_slots, sigma_within,
                    &mut rng, &normal,
                );

                // Additional precinct-specific noise (independent per candidate)
                let precinct_shocks: Vec<f64> = (0..n_cands)
                    .map(|_| normal.sample(&mut rng) * sigma_precinct)
                    .collect();

                // Combine within-bloc and precinct noise
                let combined_within: Vec<f64> = within_shocks.iter()
                    .zip(precinct_shocks.iter())
                    .map(|(w, p)| w + p)
                    .collect();

                // Fundamental uncertainty: independent per-candidate district-level shock
                let independent_shocks: Vec<f64> = if sigma_fundamental > 0.0 {
                    (0..n_cands)
                        .map(|_| normal.sample(&mut rng) * sigma_fundamental)
                        .collect()
                } else {
                    vec![0.0; n_cands]
                };

                let no_undecided = vec![0.0f64; n_cands];
                let shares = run_one_trial(
                    &baseline, &no_undecided,
                    env_shock, &bloc_shocks, &combined_within, &independent_shocks,
                    &bloc_map, &sc_matrix, 0.0,
                );

                let winner = argmax(&shares);
                win_counts[winner] += 1;
                for (ci, &s) in shares.iter().enumerate() {
                    share_samples[ci].push(s);
                }
            }

            let n = n_sim as f64;

            let win_probs: HashMap<String, f64> = req.candidates.iter().enumerate()
                .map(|(ci, name)| (name.clone(), win_counts[ci] as f64 / n))
                .collect();

            let median_pcts: HashMap<String, f64> = req.candidates.iter().enumerate()
                .map(|(ci, name)| (name.clone(), median(share_samples[ci].clone())))
                .collect();

            let median_votes: HashMap<String, f64> = req.candidates.iter().enumerate()
                .map(|(ci, name)| {
                    let m = median(share_samples[ci].clone());
                    (name.clone(), m * turnout)
                })
                .collect();

            PrecinctResult { id: precinct.id, win_probs, median_pcts, median_votes }
        })
        .collect();

    PrecinctOutput { precincts }
}

// ─── Entry point ──────────────────────────────────────────────────────────

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("Failed to read stdin");

    let request: Request = serde_json::from_str(&input).expect("Failed to parse JSON input");

    let output = match request {
        Request::District(req) => {
            let result = run_district(req);
            serde_json::to_string(&result).expect("Failed to serialize district output")
        }
        Request::Precinct(req) => {
            let result = run_precinct(req);
            serde_json::to_string(&result).expect("Failed to serialize precinct output")
        }
    };

    println!("{}", output);
}
