"""
2×2 choropleth of per-precinct turnout statistics for Chicago mayoral races
(2015, 2019, 2023).

Layout:
  Top    left : Mean turnout — first round (municipal)
  Top    right: Mean turnout — runoff (municipal_runoff)
  Bottom left : CV (std/mean) — first round      ← fluctuation metric
  Bottom right: CV (std/mean) — runoff

CV (coefficient of variation) is unit-free — a precinct with CV=0.10 swings
±10% around its mean from cycle to cycle; a precinct with CV=0.25 is much
more volatile.  Color is inverted for CV: low (stable) = blue, high
(volatile) = red.

Reads:  outputs/precinct_turnout_stats.csv
Writes: outputs/precinct_turnout_maps.png
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
STATS_CSV = SCRIPT_DIR / "outputs" / "precinct_turnout_stats.csv"
SHAPEFILE = Path(
    "/home/cole/elections/il9prediction_and_tracker/data/shapefile/IL24/IL24.shp"
)
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "precinct_turnout_maps.png"

BG = "#1a1a2e"


def main() -> None:
    print(f"Loading turnout stats from {STATS_CSV}")
    stats = pd.read_csv(STATS_CSV)
    # Normalize JoinField to uppercase for shapefile merge
    stats["joinfield_upper"] = stats["JoinField"].str.upper()

    print(f"Loading shapefile from {SHAPEFILE}")
    gdf = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
    gdf["joinfield_upper"] = gdf["JoinField"].str.upper()
    gdf = gdf[gdf["joinfield_upper"].str.startswith("CITY OF CHICAGO:", na=False)].copy()

    gdf = gdf.merge(stats, on="joinfield_upper", how="left")
    matched = gdf["mean_turnout_1r"].notna().sum()
    print(f"Matched {matched} / {len(gdf)} precincts to turnout stats")

    # ── colour ranges ──────────────────────────────────────────────────────────
    # Round to sensible limits; both mean_turnout panels share the same scale.
    t_max = max(gdf["mean_turnout_1r"].quantile(0.99),
                gdf["mean_turnout_ro"].quantile(0.99))
    t_max = round(t_max / 50) * 50  # snap to nearest 50

    cv_max = max(gdf["cv_turnout_1r"].quantile(0.99),
                 gdf["cv_turnout_ro"].quantile(0.99))
    cv_max = round(cv_max, 2)

    panels = [
        # (row, col, data_col, title, cmap, vmin, vmax, label)
        (0, 0, "mean_turnout_1r", "Mean Turnout — First Round",  "YlOrRd", 0,     t_max,  "avg. votes cast"),
        (0, 1, "mean_turnout_ro", "Mean Turnout — Runoff",        "YlOrRd", 0,     t_max,  "avg. votes cast"),
        (1, 0, "cv_turnout_1r",   "Volatility (CV) — First Round", "coolwarm", 0, cv_max, "std / mean"),
        (1, 1, "cv_turnout_ro",   "Volatility (CV) — Runoff",      "coolwarm", 0, cv_max, "std / mean"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(hspace=0.08, wspace=0.04)

    for row, col, data_col, title, cmap, vmin, vmax, bar_label in panels:
        ax = axes[row, col]
        ax.set_facecolor(BG)
        gdf.plot(
            ax=ax,
            column=data_col,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="#222233",
            linewidth=0.1,
            legend=False,
            missing_kwds={"color": "#444455", "edgecolor": "#222233", "linewidth": 0.1},
        )
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")

        s = gdf[data_col].dropna()
        stat_line = (
            f"n={len(s)}  mean={s.mean():.1f}  "
            f"p10={s.quantile(0.10):.2f}  p90={s.quantile(0.90):.2f}"
        )
        ax.text(
            0.02, 0.02, stat_line,
            transform=ax.transAxes, color="white", fontsize=8.5,
            va="bottom", fontfamily="monospace",
            bbox=dict(facecolor=BG, edgecolor="#555566", alpha=0.7, pad=4),
        )

        sm = plt.cm.ScalarMappable(
            cmap=plt.get_cmap(cmap),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.75)
        cbar.set_label(bar_label, color="white", fontsize=9)
        cbar.ax.tick_params(colors="white", labelsize=8)
        cbar.outline.set_edgecolor("#555566")

    fig.suptitle(
        "Chicago Mayoral Precinct Turnout — Mean & Volatility (2015 · 2019 · 2023)",
        color="white", fontsize=15, fontweight="bold", y=0.99,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
