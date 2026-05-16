"""
Race-agnostic logit-space demographic calibration for precinct vote share estimation.

Signals from multiple demographic crosstab dimensions are combined in logit space
so they compete multiplicatively in probability space rather than stacking
additively. This prevents conflicting signals (e.g. a heavily-Black precinct
with strongly conservative ideology) from producing nonsense estimates.

Supported dimension types (auto-detected from demographic_crosstabs keys):

  Race      "black", "hispanic", "white", "asian"
              → requires pct_{group} columns in the precinct DataFrame
              → loaded from precinct_demographics via enrich_precinct_df()

  Age       keys from AGE_BAND_COLS (e.g. "age_18_34", "age_65_plus")
              → requires pct_{key} columns in the precinct DataFrame
              → computed from granular 5-year DB buckets by enrich_precinct_df()
              → polls may use any supported breakdown; mixing breakdowns across
                polls is fine — each key is aggregated independently

  Ideology  Granular (5-bin): "very_conservative", "somewhat_conservative",
                              "moderate", "somewhat_liberal", "very_liberal"
            Simplified (3-bin): "conservative", "moderate", "progressive"
                                 (also accepts "liberal" as alias for "progressive")
              → requires score_pp column in the precinct DataFrame
              → loaded from precinct_progressive_scores via enrich_precinct_df()
              → a Gaussian kernel on prog_score maps each precinct to a
                distribution over the five ideology bins
              → "conservative" aggregates bins 0+1; "progressive"/"liberal" bins 3+4
              → polls may use either breakdown; mixing across polls is fine

Keys in demographic_crosstabs are normalized to lowercase before matching, so
poll JSON capitalization ("White" vs "white", "age_18_34" vs "Age_18_34") does
not matter.

Typical usage (from precinct_pipeline.py step1):

    from .precinct_calibration import enrich_precinct_df, compute_precinct_shares

    df = enrich_precinct_df(df, config)
    shares_df = compute_precinct_shares(
        df,
        polling["demographic_crosstabs"],
        polling["baseline"],
        config.candidates,
    )
    for c in config.candidates:
        df[f"demo_est_{c}"] = shares_df[c]
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .db import get_precinct_demographics, get_progressive_scores

# ── Dimension classification ───────────────────────────────────────────────

RACE_GROUPS = frozenset({"black", "hispanic", "white", "asian"})

# Age band keys for demographic_crosstabs.
#
# Each entry maps a key → {db_column: weight}.  Weight is 1.0 for full 5-year
# buckets, or a fraction when a poll boundary falls inside a census bucket
# (e.g. age_18_30 needs only 1/5 of the 30-34 bucket for age 30).
# Fractional weights assume uniform distribution within a 5-year bucket.
#
# Keys must be lowercase and start with "age_".
# Add new entries here to support additional poll age breakdowns.
def _band(*pairs) -> dict[str, float]:
    """Build a band dict from (col_suffix, weight) pairs, expanding M+F."""
    out: dict[str, float] = {}
    for suffix, w in pairs:
        out[f"total_male_{suffix}"]   = w
        out[f"total_female_{suffix}"] = w
    return out

AGE_BAND_COLS: dict[str, dict[str, float]] = {
    # Standard 4-band (most common)
    "age_18_34": _band(
        ("18_and_19_years", 1), ("20_years", 1), ("21_years", 1),
        ("22_to_24_years", 1), ("25_to_29_years", 1), ("30_to_34_years", 1),
    ),
    "age_35_49": _band(
        ("35_to_39_years", 1), ("40_to_44_years", 1), ("45_to_49_years", 1),
    ),
    "age_50_64": _band(
        ("50_to_54_years", 1), ("55_to_59_years", 1),
        ("60_and_61_years", 1), ("62_to_64_years", 1),
    ),
    "age_65_plus": _band(
        ("65_and_66_years", 1), ("67_to_69_years", 1), ("70_to_74_years", 1),
        ("75_to_79_years", 1), ("80_to_84_years", 1), ("85_years_and_over", 1),
    ),
    # Split-bucket bands: boundaries fall inside a 5-year census bucket.
    # 1/5 weight = one year out of five; 4/5 = four years out of five.
    "age_18_30": _band(
        ("18_and_19_years", 1), ("20_years", 1), ("21_years", 1),
        ("22_to_24_years", 1), ("25_to_29_years", 1),
        ("30_to_34_years", 0.2),   # only age 30 from this bucket
    ),
    "age_31_45": _band(
        ("30_to_34_years", 0.8),   # ages 31-34
        ("35_to_39_years", 1), ("40_to_44_years", 1),
        ("45_to_49_years", 0.2),   # only age 45
    ),
    "age_46_64": _band(
        ("45_to_49_years", 0.8),   # ages 46-49
        ("50_to_54_years", 1), ("55_to_59_years", 1),
        ("60_and_61_years", 1), ("62_to_64_years", 1),
    ),
    # Other alternative breakdowns
    "age_18_29": _band(
        ("18_and_19_years", 1), ("20_years", 1), ("21_years", 1),
        ("22_to_24_years", 1), ("25_to_29_years", 1),
    ),
    "age_30_44": _band(
        ("30_to_34_years", 1), ("35_to_39_years", 1), ("40_to_44_years", 1),
    ),
    "age_18_44": _band(
        ("18_and_19_years", 1), ("20_years", 1), ("21_years", 1),
        ("22_to_24_years", 1), ("25_to_29_years", 1), ("30_to_34_years", 1),
        ("35_to_39_years", 1), ("40_to_44_years", 1),
    ),
    "age_45_64": _band(
        ("45_to_49_years", 1), ("50_to_54_years", 1), ("55_to_59_years", 1),
        ("60_and_61_years", 1), ("62_to_64_years", 1),
    ),
    "age_45_59": _band(
        ("45_to_49_years", 1), ("50_to_54_years", 1), ("55_to_59_years", 1),
    ),
    "age_60_plus": _band(
        ("60_and_61_years", 1), ("62_to_64_years", 1),
        ("65_and_66_years", 1), ("67_to_69_years", 1), ("70_to_74_years", 1),
        ("75_to_79_years", 1), ("80_to_84_years", 1), ("85_years_and_over", 1),
    ),
    "age_50_plus": _band(
        ("50_to_54_years", 1), ("55_to_59_years", 1),
        ("60_and_61_years", 1), ("62_to_64_years", 1),
        ("65_and_66_years", 1), ("67_to_69_years", 1), ("70_to_74_years", 1),
        ("75_to_79_years", 1), ("80_to_84_years", 1), ("85_years_and_over", 1),
    ),
}

# All granular DB columns needed to compute any age band above
_AGE_LOAD_COLS: list[str] = sorted({col for cols in AGE_BAND_COLS.values() for col in cols.keys()})

IDEOLOGY_BINS = [
    "very_conservative",
    "somewhat_conservative",
    "moderate",
    "somewhat_liberal",
    "very_liberal",
]
IDEOLOGY_CENTERS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
IDEOLOGY_SIGMA   = 1.0    # spread in ideology units (±1 unit = one full bin)
IDEOLOGY_K       = 25.0   # prog_score pp per ideology unit (±50pp ≈ ±2 units)

# Maps every accepted ideology crosstab key → the IDEOLOGY_BINS indices it covers.
# "conservative" spans bins 0+1; "progressive"/"liberal" span bins 3+4.
# The granular 5-bin keys are each single-bin entries.
# Polls may use either the 3-bin or 5-bin breakdown; they're handled identically.
IDEOLOGY_ALIASES: dict[str, list[int]] = {
    "conservative":          [0, 1],
    "liberal":               [3, 4],
    "progressive":           [3, 4],
    "very_conservative":     [0],
    "somewhat_conservative": [1],
    "moderate":              [2],
    "somewhat_liberal":      [3],
    "somewhat_progressive":  [3],
    "very_liberal":          [4],
    "very_progressive":      [4],
}

# DB column name for each race group
_RACE_DB_COLS: dict[str, str] = {
    "black":    "total_not_hispanic_or_latino_black_or_african_american_alone",
    "hispanic": "total_hispanic_or_latino",
    "white":    "total_not_hispanic_or_latino_white_alone",
    "asian":    "total_not_hispanic_or_latino_asian_alone",
}
_RACE_LOAD_COLS = ["total"] + list(_RACE_DB_COLS.values())

GENDER_GROUPS = frozenset({"male", "female"})
_GENDER_DB_COLS: dict[str, str] = {
    "male":   "total_male",
    "female": "total_female",
}
# total is already in _RACE_LOAD_COLS; gender columns piggyback on the race query
_GENDER_LOAD_COLS = list(_GENDER_DB_COLS.values())


# ── Math helpers ───────────────────────────────────────────────────────────

def _logit(p: np.ndarray, eps: float = 0.005) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _ideology_weights(scores: np.ndarray) -> np.ndarray:
    """Convert prog_score array (pp) to ideology bin weights. Returns (n, 5)."""
    mu = scores[:, None] / IDEOLOGY_K
    w  = np.exp(-0.5 * ((IDEOLOGY_CENTERS - mu) / IDEOLOGY_SIGMA) ** 2)
    return w / w.sum(axis=1, keepdims=True)


# ── Data enrichment ────────────────────────────────────────────────────────

def enrich_precinct_df(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Add pct_black, pct_hispanic, pct_white, pct_asian, and score_pp columns
    to df by querying the DB. Columns already present are left untouched.

    config must have:
        race_context (str | None)      — key for precinct_progressive_scores
        prog_score_scenario (str)      — scenario to use ("generic" by default)
        demo_year (int)                — ACS year for demographics (default 2022)

    The JoinField column is found by looking for "joinfield", "JoinField", etc.
    """
    df = df.copy()

    jf_col = next(
        (c for c in ["joinfield", "JoinField", "JOINFIELD", "join_field"] if c in df.columns),
        None,
    )
    if jf_col is None:
        warnings.warn("No JoinField column found in precinct_df; skipping demographic enrichment")
        return df

    jf_values = df[jf_col].dropna().unique().tolist()
    demo_year  = getattr(config, "demo_year", 2022)

    # ── Race fractions ────────────────────────────────────────────────────
    needs_race   = any(f"pct_{g}" not in df.columns for g in _RACE_DB_COLS)
    needs_gender = any(f"pct_{g}" not in df.columns for g in _GENDER_DB_COLS)
    if (needs_race or needs_gender) and jf_values:
        try:
            load_cols = _RACE_LOAD_COLS + [c for c in _GENDER_LOAD_COLS if c not in _RACE_LOAD_COLS]
            demo = get_precinct_demographics(
                joinfields=jf_values,
                year=demo_year,
                columns=load_cols,
            )
            t = demo["total"].replace(0, np.nan)
            pct_cols: list[str] = []
            for group, db_col in {**_RACE_DB_COLS, **_GENDER_DB_COLS}.items():
                col = f"pct_{group}"
                if col not in df.columns:
                    demo[col] = demo[db_col] / t
                    pct_cols.append(col)

            demo["_jf_key"] = demo["JoinField"].str.upper()
            df["_jf_key"]   = df[jf_col].str.upper()
            df = df.merge(
                demo[["_jf_key", "total"] + pct_cols].drop_duplicates("_jf_key"),
                on="_jf_key", how="left",
            ).drop(columns=["_jf_key"])

        except Exception as exc:
            warnings.warn(f"Could not load precinct demographics: {exc}")

    # ── Age band fractions ────────────────────────────────────────────────
    needs_age = any(f"pct_{band}" not in df.columns for band in AGE_BAND_COLS)
    if needs_age and jf_values:
        try:
            age_demo = get_precinct_demographics(
                joinfields=jf_values,
                year=demo_year,
                columns=["total"] + _AGE_LOAD_COLS,
            )
            t = age_demo["total"].replace(0, np.nan)
            age_pct_cols: list[str] = []
            for band, band_weights in AGE_BAND_COLS.items():
                pct_col = f"pct_{band}"
                if pct_col not in df.columns:
                    present = {col: w for col, w in band_weights.items() if col in age_demo.columns}
                    if present:
                        weighted = sum(age_demo[col].fillna(0) * w for col, w in present.items())
                        age_demo[pct_col] = weighted / t
                        age_pct_cols.append(pct_col)

            if age_pct_cols:
                age_demo["_jf_key"] = age_demo["JoinField"].str.upper()
                df["_jf_key"]       = df[jf_col].str.upper()
                df = df.merge(
                    age_demo[["_jf_key"] + age_pct_cols].drop_duplicates("_jf_key"),
                    on="_jf_key", how="left",
                ).drop(columns=["_jf_key"])
        except Exception as exc:
            warnings.warn(f"Could not load precinct age demographics: {exc}")

    # ── Progressive scores (ideology proxy) ──────────────────────────────
    race_context = getattr(config, "race_context", None)
    if "score_pp" not in df.columns and race_context is not None:
        scenario = getattr(config, "prog_score_scenario", "generic")
        try:
            scores = get_progressive_scores(race_context, scenario=scenario)[
                ["JoinField", "score_pp"]
            ].copy()
            scores["_jf_key"] = scores["JoinField"].str.upper()
            df["_jf_key"]     = df[jf_col].str.upper()
            df = df.merge(
                scores[["_jf_key", "score_pp"]].drop_duplicates("_jf_key"),
                on="_jf_key", how="left",
            ).drop(columns=["_jf_key"])
        except Exception as exc:
            warnings.warn(f"Could not load progressive scores for '{race_context}': {exc}")

    return df


# ── Core calibration ───────────────────────────────────────────────────────

def compute_precinct_shares(
    df: pd.DataFrame,
    demographic_crosstabs: dict[str, dict[str, float]],
    baseline: dict[str, float],
    candidates: list[str],
    weight_col: str = "turnout_weight",
    ideological_blocs: list[list[str]] | None = None,
    bloc_ideological_positions: list[float] | None = None,
    bloc_ideology_strength: float = 0.0,
    candidate_precinct_boosts: dict[str, dict[str, float]] | None = None,
    ideology_race_weights: dict[str, float] | None = None,
    ideology_prog_slope: float = 0.0,
) -> pd.DataFrame:
    """
    Estimate per-precinct vote shares by combining demographic crosstab signals
    in logit space.

    demographic_crosstabs: {group_name: {candidate: fraction}}  (0–1 scale)
        as returned by poll_weighting.aggregate_polls().
    baseline: {candidate: fraction}  (0–1 scale), the weighted poll topline.

    Returns a DataFrame indexed like df with:
      - one column per candidate (predicted pct, 0–100)
      - "predicted_winner" column

    Dimensions used:
      - Race: groups in RACE_GROUPS that are both in demographic_crosstabs
        and have a matching pct_{group} column in df.
      - Ideology: any IDEOLOGY_BINS keys present in demographic_crosstabs,
        if df has a score_pp column.

    Unrecognized groups are silently ignored. If no usable data is found,
    falls back to the district baseline for every precinct.
    """
    n = len(df)

    avail_race = [
        g for g in RACE_GROUPS
        if g in demographic_crosstabs and f"pct_{g}" in df.columns
    ]
    avail_age = [
        k for k in AGE_BAND_COLS
        if k in demographic_crosstabs and f"pct_{k}" in df.columns
    ]
    avail_gender = [
        g for g in GENDER_GROUPS
        if g in demographic_crosstabs and f"pct_{g}" in df.columns
    ]
    avail_ideo = [g for g in demographic_crosstabs if g in IDEOLOGY_ALIASES]
    has_ideology = bool(avail_ideo) and "score_pp" in df.columns

    if not avail_race and not avail_age and not avail_gender and not has_ideology:
        out = pd.DataFrame(
            {c: baseline.get(c, 1.0 / len(candidates)) * 100.0 for c in candidates},
            index=df.index,
        )
        winner = max(baseline, key=baseline.get) if baseline else candidates[0]
        out["predicted_winner"] = winner
        return out

    # Ideology bin weights (n, 5) from prog_score
    ideo_w = _ideology_weights(df["score_pp"].fillna(0.0).values) if has_ideology else None

    # Z-score score_pp once; used by both ideo_scale prog slope and bloc adjustment.
    score_z = np.zeros(n)
    if "score_pp" in df.columns:
        _raw = df["score_pp"].fillna(0.0).values
        _std = _raw.std()
        score_z = (_raw - _raw.mean()) / _std if _std > 1e-6 else score_z

    # Build per-candidate bloc position map for the direct bloc-ideology adjustment
    cand_bloc_pos: dict[str, float] = {}
    has_bloc_ideo = (
        bloc_ideology_strength != 0.0
        and "score_pp" in df.columns
        and ideological_blocs is not None
        and bloc_ideological_positions is not None
        and len(bloc_ideological_positions) == len(ideological_blocs)
    )
    if has_bloc_ideo:
        for bloc, pos in zip(ideological_blocs, bloc_ideological_positions):
            for c in bloc:
                cand_bloc_pos[c] = pos

    # Per-precinct ideology attenuation factor from racial composition.
    # ideo_scale[i] ∈ (0, 1]: how much the ideology signal is trusted in precinct i.
    # Black/Hispanic precincts get lower scale because racial solidarity attenuates
    # ideological sorting. Computed once; applied to both crosstab and bloc signals.
    ideo_scale = np.ones(n)
    if ideology_race_weights:
        _scale   = np.zeros(n)
        _covered = np.zeros(n)
        for _col, _wt in ideology_race_weights.items():
            if _col in df.columns:
                _frac     = df[_col].fillna(0.0).values.clip(0.0, 1.0)
                _scale   += _frac * _wt
                _covered += _frac
        _scale   += (1.0 - _covered).clip(0.0) * 1.0  # remainder at full strength
        ideo_scale = _scale

    # Progressive-score amplification: ideology matters more in high-prog precincts.
    # Factor = (1 + slope × score_z), clipped so it never goes negative.
    if ideology_prog_slope != 0.0 and "score_pp" in df.columns:
        ideo_scale = (ideo_scale * (1.0 + ideology_prog_slope * score_z)).clip(0.05)

    # Fraction of precinct population covered by each dimension's known groups
    if avail_race:
        pct_race_covered = sum(df[f"pct_{g}"].fillna(0.0).values for g in avail_race)
        pct_race_other   = (1.0 - pct_race_covered).clip(0)
    if avail_age:
        pct_age_covered = sum(df[f"pct_{k}"].fillna(0.0).values for k in avail_age)
        pct_age_other   = (1.0 - pct_age_covered).clip(0)
    if avail_gender:
        pct_gender_covered = sum(df[f"pct_{g}"].fillna(0.0).values for g in avail_gender)
        pct_gender_other   = (1.0 - pct_gender_covered).clip(0)

    # Turnout weights for calibration
    if weight_col in df.columns:
        w = df[weight_col].fillna(1.0).values
    elif "total" in df.columns:
        w = df["total"].fillna(1.0).values
    else:
        w = np.ones(n)

    out_data: dict[str, np.ndarray] = {}
    for cand in candidates:
        tl_f     = baseline.get(cand, 1.0 / len(candidates))
        tl_logit = float(_logit(np.array([tl_f]))[0])
        delta    = np.zeros(n)

        # Race: weighted avg of per-group crosstabs → logit delta
        if avail_race:
            race_f = pct_race_other * tl_f
            for g in avail_race:
                ct     = demographic_crosstabs[g].get(cand, tl_f)
                race_f = race_f + df[f"pct_{g}"].fillna(0.0).values * ct
            delta += _logit(race_f) - tl_logit

        # Age: same weighted-average logit approach as race
        if avail_age:
            age_f = pct_age_other * tl_f
            for k in avail_age:
                ct    = demographic_crosstabs[k].get(cand, tl_f)
                age_f = age_f + df[f"pct_{k}"].fillna(0.0).values * ct
            delta += _logit(age_f) - tl_logit

        # Gender
        if avail_gender:
            gender_f = pct_gender_other * tl_f
            for g in avail_gender:
                ct       = demographic_crosstabs[g].get(cand, tl_f)
                gender_f = gender_f + df[f"pct_{g}"].fillna(0.0).values * ct
            delta += _logit(gender_f) - tl_logit

        # Ideology: Gaussian-weighted avg over covered ideology bins → logit delta.
        # Each key in avail_ideo maps to one or more bin indices via IDEOLOGY_ALIASES.
        # "conservative" sums kernel weights for bins 0+1; "progressive" for bins 3+4.
        # Bins not covered by any reported key fall back to topline (zero delta contribution).
        if has_ideology and ideo_w is not None:
            ideo_f  = np.zeros(n)
            covered = np.zeros(n)
            for key in avail_ideo:
                ct    = demographic_crosstabs[key].get(cand, tl_f)
                idxs  = IDEOLOGY_ALIASES[key]
                key_w = ideo_w[:, idxs].sum(axis=1)
                ideo_f  += key_w * ct
                covered += key_w
            ideo_f += (1.0 - covered).clip(0) * tl_f
            delta += (_logit(ideo_f) - tl_logit) * ideo_scale

        # Bloc × progressive score: direct logit adjustment.
        # score_z is z-scored across precincts; bloc_pos is the candidate's ideological position.
        # A progressive candidate (pos > 0) gets boosted in high-score_pp precincts and vice versa.
        if has_bloc_ideo and cand in cand_bloc_pos:
            delta += bloc_ideology_strength * score_z * cand_bloc_pos[cand] * ideo_scale

        # Per-candidate demographic boosts (e.g. incumbency advantage among a specific group).
        # delta += logit_units * pct_col, where pct_col is a 0–1 demographic fraction.
        if candidate_precinct_boosts and cand in candidate_precinct_boosts:
            for pct_col, logit_units in candidate_precinct_boosts[cand].items():
                if pct_col in df.columns:
                    delta += logit_units * df[pct_col].fillna(0.0).values

        combined_pct = _expit(tl_logit + delta) * 100.0

        # Small additive calibration so turnout-weighted mean == topline
        mean_pct = (combined_pct * w).sum() / w.sum()
        out_data[cand] = np.clip(combined_pct + (tl_f * 100.0 - mean_pct), 0.0, None)

    out = pd.DataFrame(out_data, index=df.index)

    # Renormalize rows to sum to 100
    row_sum = out[candidates].sum(axis=1).replace(0, np.nan)
    for cand in candidates:
        out[cand] = out[cand] / row_sum * 100.0

    out["predicted_winner"] = out[candidates].idxmax(axis=1)
    return out


# ── Convenience: check whether calibration will do anything ───────────────

def has_usable_crosstabs(
    demographic_crosstabs: dict,
    df: pd.DataFrame,
) -> bool:
    """Return True if compute_precinct_shares will use at least one dimension."""
    race_ok = any(
        g in demographic_crosstabs and f"pct_{g}" in df.columns
        for g in RACE_GROUPS
    )
    age_ok = any(
        k in demographic_crosstabs and f"pct_{k}" in df.columns
        for k in AGE_BAND_COLS
    )
    gender_ok = any(
        g in demographic_crosstabs and f"pct_{g}" in df.columns
        for g in GENDER_GROUPS
    )
    ideo_ok = (
        any(g in demographic_crosstabs for g in IDEOLOGY_ALIASES)
        and "score_pp" in df.columns
    )
    return race_ok or age_ok or gender_ok or ideo_ok
