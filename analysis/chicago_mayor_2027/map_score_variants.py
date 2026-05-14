"""
Three-panel map: generic vs. Black-progressive vs. Latino-progressive scores.

Shows how Chicago's progressive lean shifts depending on which candidate
scenario you're modeling. The generic panel pools all reference races
(with 2019 runoff dropped). The Black and Latino panels restrict to
races where the progressive bloc was led by a Black or Latino candidate.

Reads:  outputs/precinct_progressive_scores.csv
Writes: outputs/progressive_scores_by_scenario.png

Colormap: PiYG_r — pink = more progressive than the city, green = less.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
SCORES_CSV = SCRIPT_DIR / "outputs" / "precinct_progressive_scores.csv"
SHAPEFILE = Path(
    "/home/cole/elections/il9prediction_and_tracker/data/shapefile/IL24/IL24.shp"
)
OUTPUT_PATH = SCRIPT_DIR / "outputs" / "progressive_scores_by_scenario.png"

# Symmetric range so neutral (=city) sits exactly at the colormap midpoint (white).
# Wide enough to capture most of the Black and Latino distributions without
# letting a couple of extreme precincts wash out the rest.
VMIN, VMAX = -60.0, 60.0


def main() -> None:
    print(f"Loading scores from {SCORES_CSV}")
    scores = pd.read_csv(SCORES_CSV)

    print(f"Loading shapefile from {SHAPEFILE}")
    gdf = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
    gdf["joinfield_norm"] = gdf["JoinField"].str.upper()
    gdf = gdf[gdf["joinfield_norm"].str.startswith("CITY OF CHICAGO:", na=False)].copy()

    gdf = gdf.merge(scores, left_on="joinfield_norm", right_on="joinfield", how="left")
    matched = gdf["score_generic"].notna().sum()
    print(f"Matched {matched} / {len(gdf)} Chicago precincts to scores")

    variants = [
        ("score_generic", "Generic\n(all reference races)"),
        ("score_black",   "Black-progressive scenario\n(2023 mayoral + 2026 Senate)"),
        ("score_latino",  "Latino-progressive scenario\n(2015 mayoral + 2026 Comptroller)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.patch.set_facecolor("#1a1a2e")

    cmap = plt.cm.PiYG_r  # low (negative) = green, high (positive) = pink

    for ax, (col, label) in zip(axes, variants):
        ax.set_facecolor("#1a1a2e")
        gdf.plot(
            ax=ax,
            column=col,
            cmap=cmap,
            vmin=VMIN,
            vmax=VMAX,
            edgecolor="#222233",
            linewidth=0.1,
            legend=False,
            missing_kwds={"color": "#444455", "edgecolor": "#222233", "linewidth": 0.1},
        )
        ax.set_title(label, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

        s = gdf[col].dropna()
        ax.text(
            0.02, 0.02,
            f"n={len(s)}   mean={s.mean():+.1f}   std={s.std():.1f}   "
            f"min={s.min():+.1f}   max={s.max():+.1f}",
            transform=ax.transAxes, color="white", fontsize=9,
            va="bottom", fontfamily="monospace",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#555566", alpha=0.7, pad=4),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=VMIN, vmax=VMAX))
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.tolist(), orientation="horizontal",
        fraction=0.035, pad=0.04, aspect=60,
    )
    cbar.set_label(
        "Progressive lean vs. citywide (pp)  —  pink = more progressive, green = less",
        color="white", fontsize=11,
    )
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("#555566")

    fig.suptitle(
        "Chicago Progressive Precinct Lean — by Candidate Scenario",
        color="white", fontsize=16, fontweight="bold", y=0.98,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
