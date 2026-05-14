"""
Five-step precinct modeling pipeline.

Steps 1–4 run in Python (data loading, spatial joins, calibration).
Step 5 calls the Rust simulator via simulator_runner.

Requires: geopandas, pandas, numpy, shapely
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .race_config import RaceConfig
from . import simulator_runner
from .precinct_calibration import (
    enrich_precinct_df,
    compute_precinct_shares,
    has_usable_crosstabs,
)

try:
    import geopandas as gpd
    from shapely.validation import make_valid
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    warnings.warn("geopandas not available — shapefile operations will fail")


# ── JoinField normalization ────────────────────────────────────────────────

def normalize_joinfield(raw: str) -> str:
    return str(raw).strip().upper()


def assign_region_from_joinfield(jf: str, joinfield_format: str) -> str:
    """
    Derive region name from the JoinField prefix.
    Avoids the misaligned in_chicago / in_evanston flag columns.
    """
    jf = normalize_joinfield(jf)
    if joinfield_format == "IL09":
        if jf.startswith("CITY OF CHICAGO:"):
            return "Chicago"
        if jf.startswith("COOK:"):
            # Evanston precincts are in Cook but have specific identifiers
            # (75xxxxx range); keep as Suburban Cook — caller can override
            return "Suburban Cook"
        if jf.startswith("LAKE:"):
            return "Lake County"
        if jf.startswith("MCHENRY:"):
            return "McHenry County"
        return "Suburban Cook"
    elif joinfield_format == "CHICAGO":
        # "WARD XX PRECINCT YY" — extract ward number for ward-group lookup
        parts = jf.split()
        if len(parts) >= 2 and parts[0] == "WARD":
            try:
                ward = int(parts[1])
                return f"Ward {ward:02d}"
            except ValueError:
                pass
        return "Unknown"
    return "Unknown"


# ── Shapefile loading ──────────────────────────────────────────────────────

def _load_shapefile(path: Path, crs_target: str = "EPSG:26916") -> "gpd.GeoDataFrame":
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")
    gdf = gpd.read_file(str(path))
    gdf.geometry = gdf.geometry.apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )
    return gdf.to_crs(crs_target)


# ── Step 1: Demographic modeling ──────────────────────────────────────────

def _area_weighted_crosstab_overlap(
    precinct_gdf: "gpd.GeoDataFrame",
    crosstab_gdf: "gpd.GeoDataFrame",
    group_col: str,
    candidates: list[str],
    crosstabs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    For each precinct, compute area-weighted blend of crosstab groups.
    Returns {precinct_index: {candidate: estimated_pct}}.
    """
    result: dict[int, dict[str, float]] = {}
    intersection = gpd.overlay(
        precinct_gdf[["geometry"]].reset_index(),
        crosstab_gdf[[group_col, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    intersection["area"] = intersection.geometry.area

    for idx, group in intersection.groupby("index"):
        total_area = group["area"].sum()
        if total_area <= 0:
            continue
        blended: dict[str, float] = {c: 0.0 for c in candidates}
        for _, row in group.iterrows():
            grp_name = str(row[group_col]).strip()
            w = row["area"] / total_area
            grp_shares = crosstabs.get(grp_name, {})
            for c in candidates:
                blended[c] += w * grp_shares.get(c, 0.0)
        result[int(idx)] = blended

    return result


def step1_demographic_modeling(
    precinct_df: pd.DataFrame,
    crosstab_gdf: "gpd.GeoDataFrame | None",
    polling: dict[str, Any],
    config: RaceConfig,
) -> pd.DataFrame:
    """
    Estimate per-precinct candidate support using whichever crosstab data is
    available, in priority order:

    1. Logit demographic calibration (preferred when demographic_crosstabs has
       race/ideology groups and config.race_context is set).  Combines all
       available dimensions in logit space so conflicting signals compete
       multiplicatively rather than stacking.

    2. Area-weighted senate-district crosstabs (fallback for congressional/
       state races where SD-level geographic crosstabs are available).

    3. District baseline for every precinct (last resort).

    Returns precinct_df with new columns 'demo_est_{cand}' for each candidate.
    """
    df = precinct_df.copy()
    demo_crosstabs = polling.get("demographic_crosstabs", {})
    sd_crosstabs   = polling.get("senate_district_crosstabs", {})
    baseline       = polling["baseline"]

    # ── Path 1: logit demographic calibration ────────────────────────────
    if demo_crosstabs and (config.race_context is not None or any(
        f"pct_{g}" in df.columns for g in ["black", "hispanic", "white", "asian"]
    )):
        enriched = enrich_precinct_df(df, config)
        if has_usable_crosstabs(demo_crosstabs, enriched):
            shares = compute_precinct_shares(
                enriched, demo_crosstabs, baseline, config.candidates
            )
            for c in config.candidates:
                df[f"demo_est_{c}"] = shares[c].values
            # Carry enriched demographic columns forward for downstream steps
            for col in ["pct_black", "pct_hispanic", "pct_white", "pct_asian",
                        "score_pp", "total"]:
                if col in enriched.columns and col not in df.columns:
                    df[col] = enriched[col].values
            return df

    # ── Path 2: area-weighted SD crosstabs ───────────────────────────────
    if sd_crosstabs and "sd_group" in df.columns:
        for c in config.candidates:
            df[f"demo_est_{c}"] = df["sd_group"].map(
                lambda g, cand=c: sd_crosstabs.get(str(g), {}).get(cand, baseline.get(cand, 0.0))
            )
        return df

    # ── Path 3: district baseline ─────────────────────────────────────────
    for c in config.candidates:
        df[f"demo_est_{c}"] = baseline.get(c, 1.0 / len(config.candidates))
    return df


# ── Step 2: Calibrate to baseline ────────────────────────────────────────

def _calibrate_to_target(
    df: pd.DataFrame,
    est_cols: list[str],
    target: dict[str, float],
    candidates: list[str],
    turnout_col: str = "turnout_weight",
    tolerance: float = 0.005,
    max_iter: int = 50,
) -> pd.DataFrame:
    """
    Iterative additive calibration.
    Adjusts est_cols until turnout-weighted average matches target within tolerance.
    """
    df = df.copy()
    weights = df[turnout_col].values if turnout_col in df.columns else np.ones(len(df))
    total_w = weights.sum()

    for _ in range(max_iter):
        converged = True
        for cand, col in zip(candidates, est_cols):
            current_avg = (df[col].values * weights).sum() / total_w
            target_val = target.get(cand, 0.0)
            delta = target_val - current_avg
            if abs(delta) > tolerance:
                df[col] = (df[col] + delta).clip(lower=0.0)
                converged = False
        if converged:
            break

    return df


def step2_calibrate_to_baseline(
    df: pd.DataFrame,
    polling: dict[str, Any],
    config: RaceConfig,
) -> pd.DataFrame:
    est_cols = [f"demo_est_{c}" for c in config.candidates]
    return _calibrate_to_target(
        df, est_cols, polling["baseline"], config.candidates
    )


# ── Step 3: Allocate undecideds ────────────────────────────────────────────

def _build_undecided_prior(
    config: RaceConfig,
    polling: dict[str, Any],
) -> dict[str, float]:
    """
    Build the citywide undecided allocation prior: config weights blended
    with favorability aware_rate.  Returns normalized fractions summing to 1.
    """
    fav   = polling.get("favorability_topline", {})
    blend = config.favorability_blend
    alloc = config.undecided_allocation.copy()

    for c in config.candidates:
        aware    = fav.get(c, {}).get("aware_rate", 1.0) if fav else 1.0
        alloc[c] = (1.0 - blend) * alloc.get(c, 1.0) + blend * aware

    total = sum(alloc.values())
    if total <= 0:
        n = len(config.candidates)
        return {c: 1.0 / n for c in config.candidates}
    return {c: alloc[c] / total for c in config.candidates}


def step3_allocate_undecideds(
    df: pd.DataFrame,
    polling: dict[str, Any],
    config: RaceConfig,
) -> pd.DataFrame:
    """
    Distribute undecided voters across precincts.

    When demographic crosstabs and precinct demographic data are available,
    undecideds are allocated per-precinct using the same logit calibration
    as step1: the citywide undecided prior (config.undecided_allocation blended
    with favorability) is used as the baseline, and race/ideology crosstabs
    shift the allocation for each precinct individually.

    Without demographic data, falls back to a flat citywide allocation.
    Applies any extra_constraints pins from config in the fallback path.
    """
    df = df.copy()
    undecided = polling.get("undecided_total", 0.0)
    if undecided <= 0:
        for c in config.candidates:
            df[f"final_est_{c}"] = df[f"demo_est_{c}"]
        return df

    prior = _build_undecided_prior(config, polling)

    # ── Preferred path: per-precinct demographic adjustment ───────────────
    demo_crosstabs = polling.get("demographic_crosstabs", {})
    if demo_crosstabs and has_usable_crosstabs(demo_crosstabs, df):
        # Treat `prior` as the undecided-voter baseline and apply the same
        # logit demographic adjustment used for decided voters in step1.
        # The calibration in compute_precinct_shares ensures the
        # turnout-weighted city average still equals `prior`, so the
        # total undecided mass is conserved — it's just distributed
        # unevenly across precincts based on their demographics.
        precinct_alloc = compute_precinct_shares(
            df, demo_crosstabs, prior, config.candidates,
            weight_col="turnout_weight",
        )
        for c in config.candidates:
            df[f"final_est_{c}"] = (
                df[f"demo_est_{c}"] + undecided * precinct_alloc[c].values / 100.0
            )
        return df

    # ── Fallback: flat citywide allocation ────────────────────────────────
    # Apply race-specific extra constraints before normalizing
    biss_penalty = config.extra_constraints.get("biss_evanston_undecided_penalty")
    if biss_penalty is not None and "Biss" in config.candidates:
        is_evanston = df.get("is_evanston", pd.Series(False, index=df.index))
        if "joinfield" in df.columns:
            is_evanston = df["joinfield"].str.upper().str.startswith("COOK:75")

        total_ev = is_evanston.sum()
        total_ot = (~is_evanston).sum()
        if total_ev > 0 and total_ot > 0:
            boost = prior.get("Biss", 0.0) * total_ev * biss_penalty / total_ot
            df.loc[is_evanston, "_biss_alloc"] = prior.get("Biss", 0.0) * (1.0 - biss_penalty)
            df.loc[~is_evanston, "_biss_alloc"] = prior.get("Biss", 0.0) + boost
        else:
            df["_biss_alloc"] = prior.get("Biss", 0.0)

    for c in config.candidates:
        if c == "Biss" and "_biss_alloc" in df.columns:
            df[f"final_est_{c}"] = df[f"demo_est_{c}"] + undecided * df["_biss_alloc"]
        else:
            df[f"final_est_{c}"] = df[f"demo_est_{c}"] + undecided * prior.get(c, 0.0)

    if "_biss_alloc" in df.columns:
        df = df.drop(columns=["_biss_alloc"])

    return df


# ── Step 4: Final calibration to district median ──────────────────────────

def step4_final_calibration(
    df: pd.DataFrame,
    district_results: dict[str, Any],
    config: RaceConfig,
) -> pd.DataFrame:
    """
    Second calibration pass targeting the district simulation's median vote shares
    (not the raw polling baseline). This ensures the precinct map is consistent
    with the displayed win-probability output.
    """
    target = district_results.get("median_vote_shares", district_results.get("baseline", {}))
    est_cols = [f"final_est_{c}" for c in config.candidates]
    return _calibrate_to_target(df, est_cols, target, config.candidates)


# ── Step 5: Monte Carlo via Rust ──────────────────────────────────────────

def step5_monte_carlo(
    df: pd.DataFrame,
    config: RaceConfig,
) -> pd.DataFrame:
    """
    Call the Rust simulator with per-precinct baselines.
    Adds win_prob_{cand}, median_pct_{cand}, median_votes_{cand} columns.
    """
    id_col = "joinfield" if "joinfield" in df.columns else df.index.name or "index"

    precincts = []
    for _, row in df.iterrows():
        pid = str(row.get("joinfield", row.name))
        baseline = {c: float(row.get(f"final_est_{c}", 0.0)) for c in config.candidates}
        turnout = float(row.get("turnout_weight", 100.0))
        precincts.append({"id": pid, "baseline": baseline, "turnout_weight": turnout})

    result = simulator_runner.run_precinct_sim(
        n_simulations=config.n_sim_precinct,
        candidates=config.candidates,
        moe_district=config.moe_district,
        moe_precinct=config.moe_precinct,
        ideological_blocs=config.ideological_blocs,
        precincts=precincts,
    )

    # Index results by precinct id
    result_map = {r["id"]: r for r in result["precincts"]}

    df = df.copy()
    for c in config.candidates:
        df[f"win_prob_{c}"] = 0.0
        df[f"median_pct_{c}"] = 0.0
        df[f"median_votes_{c}"] = 0.0

    for _, row in df.iterrows():
        pid = str(row.get("joinfield", row.name))
        if pid not in result_map:
            continue
        r = result_map[pid]
        for c in config.candidates:
            df.at[row.name, f"win_prob_{c}"] = r["win_probs"].get(c, 0.0)
            df.at[row.name, f"median_pct_{c}"] = r["median_pcts"].get(c, 0.0)
            df.at[row.name, f"median_votes_{c}"] = r["median_votes"].get(c, 0.0)

    return df


# ── Spatial join helpers ───────────────────────────────────────────────────

def load_and_join_precinct_shapefile(
    config: RaceConfig,
    precinct_csv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load the precinct shapefile, spatially clip to the district boundary,
    join to the CSV data via JoinField, and return a merged DataFrame.

    The shapefile contributes ONLY geometry and JoinField_norm.
    The CSV is the left frame to avoid column naming conflicts.
    """
    if not HAS_GEO:
        raise ImportError("geopandas is required for shapefile operations")

    precinct_shp = _load_shapefile(config.precinct_shapefile())
    district_shp = _load_shapefile(config.district_shapefile())

    # Clip to district boundary
    district_union = district_shp.geometry.union_all()
    precinct_shp = precinct_shp[precinct_shp.geometry.intersects(district_union)].copy()
    precinct_shp["geometry"] = precinct_shp.geometry.intersection(district_union)

    # Normalize JoinField — try common column names
    for col in ["JoinField", "JOINFIELD", "joinfield", "JOIN_FIELD"]:
        if col in precinct_shp.columns:
            precinct_shp["JoinField_norm"] = precinct_shp[col].apply(normalize_joinfield)
            break
    else:
        raise KeyError("No JoinField column found in precinct shapefile")

    # Join: CSV left, shapefile contributes geometry + JoinField_norm only
    merged = precinct_csv.copy()
    if "joinfield" in merged.columns:
        merged["joinfield_norm"] = merged["joinfield"].apply(normalize_joinfield)
    else:
        raise KeyError("precinct_csv must have a 'joinfield' column")

    geo_slim = precinct_shp[["JoinField_norm", "geometry"]].rename(
        columns={"JoinField_norm": "joinfield_norm"}
    )
    merged = merged.merge(geo_slim, on="joinfield_norm", how="left")
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:26916")

    # Assign region
    merged["region"] = merged["joinfield"].apply(
        lambda jf: assign_region_from_joinfield(jf, config.joinfield_format)
    )

    return merged


# ── Ward prior aggregation (Chicago) ──────────────────────────────────────

def compute_ward_priors(df: pd.DataFrame, config: RaceConfig) -> dict[str, dict[str, float]]:
    """
    Aggregate precinct-level final_est_{cand} to ward level using turnout_weight.
    Returns {ward: {candidate: share}} keyed by zero-padded ward string ("01"…"50").
    Only meaningful for joinfield_format="CHICAGO".
    """
    ward_sums: dict[str, dict[str, float]] = {}
    ward_weights: dict[str, float] = {}

    for _, row in df.iterrows():
        jf = str(row.get("joinfield", "")).upper()
        parts = jf.split()
        if len(parts) < 2 or parts[0] != "WARD":
            continue
        try:
            ward = f"{int(parts[1]):02d}"
        except ValueError:
            continue

        w = float(row.get("turnout_weight", 1.0))
        if ward not in ward_sums:
            ward_sums[ward] = {c: 0.0 for c in config.candidates}
            ward_weights[ward] = 0.0
        ward_weights[ward] += w
        for c in config.candidates:
            ward_sums[ward][c] += float(row.get(f"final_est_{c}", 0.0)) * w

    priors: dict[str, dict[str, float]] = {}
    district_sums: dict[str, float] = {c: 0.0 for c in config.candidates}
    district_total_w = 0.0

    for ward, total_w in ward_weights.items():
        if total_w <= 0:
            continue
        raw = {c: ward_sums[ward][c] / total_w for c in config.candidates}
        total = sum(raw.values())
        if total > 1e-9:
            normalized = {c: raw[c] / total for c in config.candidates}
            priors[ward] = {c: round(normalized[c], 6) for c in config.candidates}
            for c in config.candidates:
                district_sums[c] += normalized[c] * total_w
            district_total_w += total_w

    # Store district-level weighted average so early vote code can compute ward leans
    if district_total_w > 0:
        district = {c: round(district_sums[c] / district_total_w, 6) for c in config.candidates}
        total = sum(district.values())
        if total > 1e-9:
            priors["_district"] = {c: district[c] / total for c in config.candidates}

    return priors


# ── Full pipeline entry point ──────────────────────────────────────────────

def run_precinct_pipeline(
    config: RaceConfig,
    polling: dict[str, Any],
    district_results: dict[str, Any],
    precinct_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run all five steps and return the fully annotated precinct DataFrame.

    If precinct_csv_path is None, uses config.data_path("csv_data/expectations/IL_09_precinct_probabilities.csv")
    or an empty DataFrame if the file doesn't exist yet.
    """
    if precinct_csv_path is None:
        precinct_csv_path = config.data_path(
            "csv_data", "expectations", f"{config.race_id}_precinct_probabilities.csv"
        )

    if precinct_csv_path.exists():
        df = pd.read_csv(str(precinct_csv_path))
        if "joinfield" not in df.columns:
            # Try alternate column names
            for alt in ["JoinField", "JOINFIELD", "join_field"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "joinfield"})
                    break
    else:
        warnings.warn(f"Precinct CSV not found at {precinct_csv_path}; starting from empty DataFrame")
        df = pd.DataFrame({"joinfield": []})

    # Ensure turnout_weight column exists
    if "turnout_weight" not in df.columns:
        df["turnout_weight"] = 100.0

    # Load crosstab shapefile if configured
    crosstab_gdf = None
    crosstab_path = config.crosstab_shapefile()
    if crosstab_path is not None and HAS_GEO:
        try:
            crosstab_gdf = _load_shapefile(crosstab_path)
        except FileNotFoundError as e:
            warnings.warn(str(e))

    # Step 1 — demographic modeling
    df = step1_demographic_modeling(df, crosstab_gdf, polling, config)

    # Step 2 — calibrate to polling baseline
    df = step2_calibrate_to_baseline(df, polling, config)

    # Step 3 — allocate undecideds
    df = step3_allocate_undecideds(df, polling, config)

    # Step 4 — final calibration to district simulation median
    df = step4_final_calibration(df, district_results, config)

    # Step 5 — Monte Carlo via Rust
    df = step5_monte_carlo(df, config)

    # Write ward priors for early-vote estimation (Chicago only)
    if config.joinfield_format == "CHICAGO":
        import json as _json
        priors = compute_ward_priors(df, config)
        if priors:
            priors_path = config.output_dir / "ward_priors.json"
            priors_path.parent.mkdir(parents=True, exist_ok=True)
            priors_path.write_text(_json.dumps(priors, indent=2), encoding="utf-8")

    return df
