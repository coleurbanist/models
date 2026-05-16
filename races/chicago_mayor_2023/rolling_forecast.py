"""
Rolling forecast: re-run the full pipeline at each poll date using only
polls available up to that date. Generates ward and precinct maps for each
snapshot, then plots vote share and runoff probability over time.

Snapshot outputs land in:
  outputs/rolling/YYYY-MM-DD/ward_map.png
  outputs/rolling/YYYY-MM-DD/precinct_map.png
  outputs/rolling/YYYY-MM-DD/regional_vote_forecast.json
  outputs/rolling/YYYY-MM-DD/<race_id>_precinct_probabilities.csv

Run from repo root: python -m races.chicago_mayor_2023.rolling_forecast
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from races.chicago_mayor_2023.race_config import CONFIG
from races.chicago_mayor_2023.map_precincts import draw_precinct_map
from races.chicago_mayor_2023.map_wards import draw_ward_map
from core.poll_weighting import aggregate_polls, run_district_simulation
from core.precinct_pipeline import run_precinct_pipeline
from core.regional_forecast import generate_regional_forecast

POLLS_PATH    = CONFIG.polls_round1_path
ROLLING_DIR   = CONFIG.output_dir / "rolling"
ELECTION_DAY  = date(2023, 2, 28)

ACTUALS = {
    "Paul Vallas":     0.338,
    "Brandon Johnson": 0.217,
    "Lori Lightfoot":  0.171,
    "Chuy García":     0.137,
    "Willie Wilson":   0.089,
}

SHOW = ["Paul Vallas", "Brandon Johnson", "Lori Lightfoot", "Chuy García", "Willie Wilson"]


def run_at_date(all_polls: list[dict], cutoff: date) -> dict:
    polls = [p for p in all_polls if date.fromisoformat(p["field_end"]) <= cutoff]
    if not polls:
        return {}

    snapshot_dir = ROLLING_DIR / cutoff.isoformat()
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    config = replace(CONFIG, polls=polls, n_sim_district=200_000, output_dir=snapshot_dir)

    polling  = aggregate_polls(config, as_of=cutoff.isoformat())
    district = run_district_simulation(config, polling)

    precincts = run_precinct_pipeline(config, polling, district)
    regional  = generate_regional_forecast(config, precincts)

    # Write snapshot outputs
    precinct_csv = snapshot_dir / f"{config.race_id}_precinct_probabilities.csv"
    precincts.to_csv(str(precinct_csv), index=False)
    regional_json = snapshot_dir / "regional_vote_forecast.json"
    regional_json.write_text(json.dumps(regional, indent=2), encoding="utf-8")

    # Generate maps
    label = cutoff.strftime("%b %-d, %Y")
    draw_ward_map(regional_json, snapshot_dir, title_suffix=label)
    draw_precinct_map(precinct_csv, snapshot_dir, title_suffix=label)

    means   = district.get("mean_vote_shares") or district.get("median_vote_shares") or {}
    advance = district.get("advance_probs") or district.get("win_probs") or {}
    return {"means": means, "advance": advance, "n_polls": len(polls)}


def make_gifs() -> None:
    """Assemble per-date map PNGs into animated GIFs."""
    snapshot_dirs = sorted(
        d for d in ROLLING_DIR.iterdir()
        if d.is_dir() and d.name.count("-") == 2  # YYYY-MM-DD
    )
    if not snapshot_dirs:
        print("No snapshots found — run rolling forecast first.")
        return

    for map_file, gif_name in [("ward_map.png", "ward_maps.gif"),
                                ("precinct_map.png", "precinct_maps.gif")]:
        frames = []
        for d in snapshot_dirs:
            p = d / map_file
            if p.exists():
                frames.append(Image.open(p).convert("RGBA"))

        if not frames:
            print(f"No {map_file} frames found.")
            continue

        out = CONFIG.output_dir / gif_name
        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=800,   # ms per frame
            loop=0,         # loop forever
            disposal=2,
        )
        print(f"Saved → {out}  ({len(frames)} frames)")


def main():
    with POLLS_PATH.open() as f:
        all_polls = json.load(f)

    dates = sorted({date.fromisoformat(p["field_end"]) for p in all_polls})

    rows = []
    for d in dates:
        n_polls_at = sum(1 for p in all_polls if date.fromisoformat(p["field_end"]) <= d)
        print(f"  {d}  ({n_polls_at} polls)...")
        result = run_at_date(all_polls, d)
        if not result:
            continue
        row = {"date": d, "n_polls": result["n_polls"]}
        for c in SHOW:
            row[f"share_{c}"]   = result["means"].get(c, 0) * 100
            row[f"advance_{c}"] = result["advance"].get(c, 0) * 100
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    colors = {c: CONFIG.colors.get(c, "#888") for c in SHOW}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")

    for cand in SHOW:
        ax1.plot(df["date"], df[f"share_{cand}"], color=colors[cand], linewidth=2,
                 marker="o", markersize=5, label=cand)
        if cand in ACTUALS:
            ax1.axhline(ACTUALS[cand] * 100, color=colors[cand],
                        linestyle="--", linewidth=1, alpha=0.5)

    ax1.axvline(pd.Timestamp(ELECTION_DAY), color="white", linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_ylabel("Projected vote share (%)", color="white")
    ax1.set_title("Chicago Mayor 2023 — Rolling forecast", color="white", fontsize=14, pad=10)
    ax1.legend(fontsize=9, framealpha=0.3, labelcolor="white",
               facecolor="#1a1a2e", edgecolor="none", loc="upper left")
    ax1.text(0.99, 0.97, "Dashed = actual result", transform=ax1.transAxes,
             color="white", alpha=0.5, fontsize=8, ha="right", va="top")
    ax1.yaxis.label.set_color("white")

    for cand in SHOW:
        ax2.plot(df["date"], df[f"advance_{cand}"], color=colors[cand], linewidth=2,
                 marker="o", markersize=5, label=cand)

    ax2.axhline(50, color="white", linestyle=":", linewidth=0.8, alpha=0.4)
    ax2.axvline(pd.Timestamp(ELECTION_DAY), color="white", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Prob. advance to runoff (%)", color="white")
    ax2.set_xlabel("Poll field end date", color="white")
    ax2.yaxis.label.set_color("white")
    ax2.xaxis.label.set_color("white")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    for _, row in df.iterrows():
        ax1.annotate(str(int(row["n_polls"])), (row["date"], ax1.get_ylim()[0]),
                     textcoords="offset points", xytext=(0, 4),
                     fontsize=6, color="white", alpha=0.5, ha="center")

    plt.tight_layout()

    out = CONFIG.output_dir / "rolling_forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")
    print(f"Snapshot maps → {ROLLING_DIR}/")
    plt.show()

    print("\nBuilding GIFs...")
    make_gifs()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gifs-only", action="store_true", help="Build GIFs from existing snapshots without re-running the forecast")
    args = parser.parse_args()
    if args.gifs_only:
        make_gifs()
    else:
        main()
