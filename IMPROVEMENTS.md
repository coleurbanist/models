# Known Gaps and Planned Improvements

Issues identified from IL-09 2026 post-mortem and design review.
Roughly ordered by impact.

---

## 0. Undecided voter allocation (high priority)

**Gap:** Undecideds are currently allocated as a fixed pre-computation in Python
using weighted shares before the simulator runs. The *allocation* of undecideds
is itself a major source of uncertainty that the MOE-based noise doesn't capture.

**Observed in IL-09 2026:** Lower-tier candidates (4th place and below) basically
hit their polling averages exactly, meaning almost all undecided voters broke to
the top three candidates. The current model treats undecided allocation weights
as fixed inputs, so it cannot reproduce or anticipate this pattern.

**What this means:** The model systematically underestimates variance for
lower-tier candidates (they could beat expectations if undecideds break their
way) and underestimates the ceiling for top-tier candidates (who may absorb
most of the undecided pool).

**Fix (two parts):**

1. **Model undecided allocation as a random variable in the simulator.**
   Instead of passing a single fixed baseline to Rust, pass a baseline that
   reflects only the decided voters, plus an `undecided_share` per candidate
   representing their *weight* in claiming undecideds. In each trial, draw a
   random undecided allocation (Dirichlet distribution over weights) and add
   it to the decided baseline before applying shocks. This adds a second,
   independent source of variance on top of the polling MOE.

2. **Calibrate undecided allocation from historical races.**
   Use the DB to compare final results vs. pre-election polling averages across
   past Illinois primaries. Fit a model of how undecideds historically distributed
   (by candidate tier, by ideological bloc, by incumbency) and use that as the
   prior for the Dirichlet draw. The IL-09 2026 result — undecideds breaking
   overwhelmingly to top-tier candidates — should be a calibration data point.

**Implementation note:** The `undecided_allocation` dict already exists in
`RaceConfig` as weights. The change is moving the randomization from Python
into the Rust simulator so it varies trial-by-trial rather than being fixed.

---

## 1. Election night: update baseline from live results

**Gap:** On election night the simulator still runs from the pre-election polling
baseline even as results arrive. If a candidate is running well ahead of model
in early results, that information is ignored.

**The mechanism differs by race type:**

**IL-09 and similar (precinct results available):**
As each batch of precincts reports, compute each candidate's actual share in
reported precincts vs. their modeled share in those same precincts. Apply that
delta to the baseline passed to the simulator for the remaining unreported
precincts.

**Chicago mayoral (no precinct results on election night):**
Chicago Board of Elections does not release precinct-level mayoral results
on election night. However, aldermanic races are on the same ballot and DO
report by ward. Use ward-level aldermanic results as a proxy to infer mayoral
performance by ward — this is already the design of `core/election_night/ward_inference.py`.

`WardInferenceEngine` runs two phases:
- Phase A: turnout tracking — compare actual aldermanic turnout per ward to
  prior expectations to detect which candidate's geographic strongholds are
  over/under-turning out.
- Phase B: credibility-weighted share blend — as mayoral results trickle in
  at the ward level, blend them with the aldermanic proxy signal.

**What still needs to be built:**
- `ward_group_map.json`, `ward_turnout_prior.json`, `ward_share_prior.json`
  data files (require 2023 mayoral + aldermanic results from the DB)
- Wire `WardInferenceEngine` output into the baseline adjustment that gets
  passed to the simulator on each tick
- The tick loop in `election_logger.py` needs a Chicago-specific branch that
  calls `WardInferenceEngine` instead of the direct precinct delta approach

---

## 2. Differential turnout by candidate stronghold

**Gap:** The simulator treats `turnout_weight` per precinct as a fixed input.
It doesn't model the scenario where Biss precincts turn out at 110% of
expectation while Abughazaleh precincts turn out at 90%. That turnout
imbalance shifts the district result even if every candidate hits their
within-precinct share exactly.

**Fix (Python, then pass adjusted weights to Rust):**
Add a turnout model that draws correlated turnout multipliers by region before
each simulation call. Natural fit for the existing ideological bloc structure:
draw a turnout shock per bloc (same Normal draw used for vote share shocks),
apply it to `turnout_weight` for precincts where that bloc is strong, then pass
the adjusted weights to the simulator. Relevant for both pre-election modeling
and election night, where early ward-level turnout is one of the most
informative early signals.

---

## 3. Home territory / geographic loyalty bonus

**Gap:** In IL-09 every top candidate overperformed their polling baseline in
their geographic stronghold — a well-documented pattern in multi-candidate
primaries. The simulator doesn't model this. For Chicago mayoral it's likely
more pronounced due to ward organization effects.

**Fix (Python, `core/precinct_pipeline.py`):**
Before calling the simulator, apply a geographic loyalty bonus in the precinct
pipeline: nudge each candidate's baseline upward in their home precincts
(identified by ward, senate district, or neighborhood cluster) and compensate
elsewhere to keep the district total consistent. The `extra_constraints` dict
in `RaceConfig` is the right place to encode these pins per race — the existing
`biss_evanston_undecided_penalty` is the same pattern.

**Important distinction — two separate effects that look similar in error analysis:**

1. **Home territory** (e.g. Fine in Northfield Township, Biss in Evanston): generalizable
   pattern where a candidate overperforms near where they live due to name recognition
   and local relationships. Belongs in the generic geographic loyalty bonus framework
   and can be estimated from the gap between actual results and district-wide share.

2. **Community voting bloc** (e.g. Fine in Ward 50 / Niles Township): a specific
   demographic community voting heavily as a bloc for a candidate, independent of
   home territory. These produce extreme precinct-level effects (Fine got 55–80%
   in Orthodox Jewish precincts vs a predicted ~18–20%) and are candidate-specific
   and non-transferable to other races.

   These belong in `extra_constraints` keyed to the specific candidate and precinct
   set — not in the generic loyalty bonus. The 2026 results identify the affected
   precincts exactly: Ward 50 precincts and specific Niles Township (Cook 8200 series)
   precincts where Fine's actual share exceeded ~45%. ACS data has no religion field,
   so the election results themselves are the best available signal for where these
   communities are. If Fine runs again, hardcode those precincts directly.

   **Niles Township is especially complex:** it contains multiple distinct community
   blocs pulling in different directions. Fine overperformed heavily in the Orthodox
   Jewish precincts, but Amiwala also overperformed in different Niles precincts with
   a large South Asian population. A township-level loyalty bonus would be wrong here —
   constraints must be at the individual precinct level, identified from actual results
   rather than township-wide demographic averages. The 2026 results are the ground
   truth for which precincts belong to which community signal.

---

## 4. Minor candidates below the tracked field

**Gap:** Candidates outside the top tracked field are currently ignored entirely.
Their votes don't appear in any precinct baseline, so the top candidates'
shares are implicitly over-normalized. In a ward where a minor candidate pulls
3–4%, the model misattributes that share to the top candidates.

**Fix (Python, `core/precinct_pipeline.py`):**
Include minor candidates in the precinct baseline at a small estimated share
(based on historical comparable-race minor candidate totals from the DB).
Pass them through normalization so the top candidates' shares are reduced
accordingly. They don't need to appear in simulation outputs — just include
them in the denominator.

---

## 5. Pre-election candidate surges not detected in polling

**Gap:** The simulator's independent noise is symmetric and centered on the
polling baseline. It can simulate a candidate outperforming in a given trial,
but it can't model directional late momentum (e.g. a candidate catching fire
in the final week with no new poll to capture it).

**Partial fix:**
No clean solution pre-election without a poll. On election night this is
handled by gap #1 (live result adjustment). Pre-election, the best proxy is
monitoring social media / earned media volume as a leading indicator and
manually adjusting the baseline if warranted — not something to automate.

---

## 6. MOE shrinkage as election night progresses

**Gap:** Related to #1. `moe_district` is fixed at the pre-election value
(e.g. 4.4pp) all night even as hundreds of precincts report in and actual
uncertainty collapses.

**Fix:** In the election night tick loop, recompute `moe_district` passed to
the simulator as:

```
effective_moe = moe_district * sqrt(1 - fraction_reported)
```

This is a simple approximation but much better than holding MOE constant.
Implement alongside gap #1 in `core/election_night/`.

---

## 7. Runoff probability as a first-class output (Chicago mayoral)

**Gap:** The simulator reports win probabilities as if there's a single winner.
For Chicago mayoral with `has_runoff=True`, the meaningful number is "probability
of making the runoff" for each candidate — which is very different from
"probability of winning outright" in a crowded field where nobody is near 50%.

**Fix (Rust, district mode):**
In each district simulation trial, identify the top-two finishers. Track both
"won outright" (>50%) and "made runoff" counts per candidate. Add
`runoff_probs` to `DistrictOutput`. Python already has `has_runoff` and
`runoff_threshold` in `RaceConfig` — pass them through to the simulator in
the district request payload.

---

## 8. Correlated polling error (systematic environment shock)

**Gap:** The bloc shock models ideological correlation but misses a different
kind of error: systematic polling miss that affects all candidates in the same
direction. If the poll sample skews younger/more progressive than actual
turnout composition, every candidate's baseline is wrong in the same direction.

**Fix (Rust, district and precinct modes):**
Add a small "environment shock" — a single Normal draw applied equally to all
candidates before bloc shocks — representing overall sample composition error.
Magnitude should be smaller than `moe_district` (maybe 30–40% of it). This
widens the tails of the distribution appropriately and produces more realistic
scenarios where the whole field runs ahead of or behind polling.

---

## 9. Candidate dropout / consolidation scenario modeling

**Gap:** In multi-candidate primaries, especially with a runoff, late
consolidation is common — a lower-tier candidate drops out and endorses
someone. The model has no way to handle this.

**Fix (Python, `core/poll_weighting.py` or race config):**
Add an optional `consolidation_scenarios` field to `RaceConfig` specifying
redistribution rules (e.g. "redistribute 70% of Amiwala's support to Biss,
30% to Fine"). Apply before aggregating polls and passing baselines to the
simulator. Useful for scenario modeling and for updating the model quickly
if a candidate actually drops out before election day.

---

## 11. Two-level bloc shock: district-level vs within-bloc competition

**Gap (confirmed by IL-09 2026 post-mortem):**
The current Rust simulator draws one shock per bloc and applies it in the same
direction to all bloc members. This captures the district-level dynamic correctly
(if progressives have a good night, the total progressive vote share goes up), but
misses that within the progressive bloc individual candidates *compete* against each
other. If Biss absorbs most of the progressive surge, Simmons/Amiwala/Huynh lose
share even though they're in the same bloc.

In IL-09 the correct ideological blocs were:
  Moderates:   Fine, Andrew
  Progressives: Biss, Abughazaleh, Simmons, Amiwala, Huynh

The old model had Biss grouped with Fine as "moderate/establishment" which was wrong.
Abughazaleh↔Simmons error correlation (r=+0.32) confirmed they genuinely moved
together. Fine↔Abughazaleh (r=-0.61) confirmed they were in opposing blocs.

**Fix (Rust, district and precinct modes):**
Replace the single bloc shock with a two-level model:
  1. **Bloc-level shock**: draw one Normal shock per bloc, apply to the *total*
     vote share of the bloc. This preserves the existing behavior.
  2. **Within-bloc redistribution shock**: draw a second shock that shifts share
     *between* candidates within the bloc (one candidate's gain = others' loss).
     This is a zero-sum perturbation within the bloc.

The within-bloc shock should be smaller than the bloc shock — it represents
candidate-level variation within an ideological coalition, not a district-wide swing.

---

## 12. Undecided allocation should reflect perceived viability, not just ideology

**Gap (confirmed by IL-09 2026 post-mortem):**
Undecideds broke heavily toward the three candidates perceived as potentially
winning (Biss, Fine, Abughazaleh), not toward ideological proximity. The current
model allocates undecideds by `undecided_allocation` weights which are set manually
and approximate tier but don't capture the viability signal.

In IL-09, top-tier candidates absorbed essentially all undecideds. Lower-tier
candidates met their polling averages almost exactly — they got no undecided lift.

**Calibration from post-mortem:**
The `calibration_params.json` from `analysis/il09_2026/postmortem.py` contains
empirically derived `undecided_allocation` weights based on actual 2026 results.
Use those as the starting point for any future IL-09 or similar race.

More broadly: weight undecided allocation by poll share (a proxy for perceived
viability) rather than treating all candidates equally within a tier. Candidates
polling above ~15% should receive substantially higher undecided weights than
candidates polling below ~10%.

---

## 13. Small moderate blocs behave differently from large progressive blocs

**Gap (confirmed by IL-09 2026 post-mortem):**
With only two moderate candidates (Fine and Andrew), a district-wide moderate bloc
shock adds noise without signal. Fine's result was dominated by her community bloc
effects (Orthodox Jewish precincts), and Andrew had his own community bloc in the
Cook 8100 series (Morton Grove/Lincolnwood area). Neither of their results was
primarily driven by a district-wide moderate swing.

For a two-candidate moderate bloc, the meaningful modeling question isn't
"how did moderates do tonight?" but "which specific communities did each candidate
lock up?" — a question that requires precinct-level constraints, not a bloc shock.

**Implication for future races:**
When a bloc has only 1–2 candidates, skip the bloc shock and rely on precinct-level
geographic constraints instead. The bloc shock framework is most valuable for blocs
of 3+ candidates where genuine co-movement across the field is likely.

---

## 10. Ward organization effects (Chicago mayoral)

**Gap:** Some Chicago wards have strong aldermanic organizations that reliably
deliver turnout and vote share for endorsed candidates regardless of citywide
polling. This is distinct from geographic loyalty — it's an active ground game
effect that historically persists across election cycles.

**Fix (Python, `core/precinct_pipeline.py`):**
Build per-ward organization strength priors from 2019 and 2023 mayoral results
in the DB: compare each candidate's ward-level overperformance vs. their
citywide share, controlling for demographics. Encode as ward-level baseline
adjustments in `extra_constraints`. This likely won't be possible until the
2027 candidate field is known and endorsements are announced.
