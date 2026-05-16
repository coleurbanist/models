"""
Precinct-level choropleth of model estimates vs actual results.
Run from repo root: python -m races.chicago_mayor_2023.map_precincts
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

try:
    import folium
    from branca.element import MacroElement
    from jinja2 import Template
    _FOLIUM = True
except ImportError:
    _FOLIUM = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from races.chicago_mayor_2023.race_config import CONFIG
from races.chicago_mayor_2023.compare_results import CAND_MAP, DB_PATH

PRECINCT_SHP = Path(__file__).parent.parent.parent / "Shapefiles" / "chicago_precincts.geojson"
WARD_SHP     = Path(__file__).parent.parent.parent / "Shapefiles" / "chicago_wards.geojson"

COLORS = CONFIG.colors
FALLBACK_COLOR = "#888888"


def load_actual_leaders() -> pd.DataFrame:
    """Returns JoinField, _jf, actual_leader, and act_{LastName} float columns (0-100 share)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT JoinField, candidate_name, votes
        FROM election_results
        WHERE race_type = 'chicago_mayor'
          AND election_type = 'municipal'
          AND year = 2023
    """, conn)
    conn.close()
    df["candidate"] = df["candidate_name"].map(CAND_MAP)
    df = df.dropna(subset=["candidate"])
    wide = df.pivot_table(index="JoinField", columns="candidate", values="votes", aggfunc="sum", fill_value=0)
    wide.columns.name = None
    cand_cols = list(wide.columns)
    total = wide[cand_cols].sum(axis=1).replace(0, float("nan"))
    wide["actual_leader"] = wide[cand_cols].idxmax(axis=1)
    for cand in cand_cols:
        wide[f"act_{cand.split()[-1]}"] = (wide[cand] / total * 100).round(1)
    wide["_jf"] = wide.index.str.strip().str.upper()
    act_cols = [c for c in wide.columns if c.startswith("act_")]
    return wide[["_jf", "actual_leader"] + act_cols].reset_index()


def draw_precinct_map(precinct_csv: Path, save_dir: Path, title_suffix: str = "") -> None:
    """
    Draw a two-panel precinct choropleth (model vs actual) and save to save_dir/precinct_map.png.
    precinct_csv: path to the *_precinct_probabilities.csv for this snapshot.
    """
    df = pd.read_csv(precinct_csv)

    final_cols = [c for c in df.columns if c.startswith("final_est_")]
    df["leader"] = df[final_cols].idxmax(axis=1).str.replace("final_est_", "", regex=False)
    df["_jf"] = df["joinfield"].str.strip().str.upper()

    est_cols = []
    for col in final_cols:
        cand = col.replace("final_est_", "")
        ecol = f"est_{cand.split()[-1]}"
        df[ecol] = (df[col] * 100).round(1)
        est_cols.append(ecol)

    actual  = load_actual_leaders()
    act_cols = [c for c in actual.columns if c.startswith("act_")]

    gdf = gpd.read_file(PRECINCT_SHP)
    gdf["_jf"] = gdf["JoinField"].str.strip().str.upper()
    gdf = gdf.merge(df[["_jf", "leader"] + est_cols], on="_jf", how="left")
    gdf = gdf.merge(actual[["_jf", "actual_leader"] + act_cols], on="_jf", how="left")
    gdf = gdf.drop(columns=["_jf"]).to_crs("EPSG:4326")

    all_leaders = sorted(set(gdf["leader"].dropna()) | set(gdf["actual_leader"].dropna()))
    cand_color  = {c: COLORS.get(c, FALLBACK_COLOR) for c in all_leaders}

    wards = gpd.read_file(WARD_SHP).to_crs("EPSG:4326") if WARD_SHP.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.set_axis_off()

    def plot_panel(ax, leader_col, title):
        for cand in all_leaders:
            sub = gdf[gdf[leader_col] == cand]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=cand_color[cand], edgecolor="none", linewidth=0)
        unmatched = gdf[gdf[leader_col].isna()]
        if not unmatched.empty:
            unmatched.plot(ax=ax, color="#333", edgecolor="none")
        if wards is not None:
            wards.boundary.plot(ax=ax, color="white", linewidth=0.5, alpha=0.4)
        leaders_present = gdf[leader_col].dropna().unique()
        patches = [
            mpatches.Patch(color=cand_color[c], label=c)
            for c in sorted(leaders_present, key=lambda c: -(gdf[leader_col] == c).sum())
        ]
        ax.legend(handles=patches, loc="lower left", fontsize=8,
                  framealpha=0.3, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="none")
        ax.set_title(title, color="white", fontsize=13, pad=8)

    plot_panel(axes[0], "leader",        "Model: leading candidate by precinct")
    plot_panel(axes[1], "actual_leader", "Actual: leading candidate by precinct")

    suptitle = "Chicago Mayor 2023 — Precinct-level forecast vs. actual"
    if title_suffix:
        suptitle += f"  ({title_suffix})"
    fig.suptitle(suptitle, color="white", fontsize=15, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "precinct_map.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")

    gdf["tooltip_html"] = gdf.apply(
        lambda row: _build_precinct_tooltip(row, cand_color, est_cols, act_cols), axis=1
    )
    _save_interactive_precinct_map(gdf, cand_color, save_dir, title_suffix)


def _build_precinct_tooltip(row: pd.Series, cand_color: dict, est_cols: list, act_cols: list) -> str:
    pred = row.get("leader") or "Unknown"
    act  = row.get("actual_leader") or "Unknown"
    pc   = cand_color.get(pred, "#888888")
    ac   = cand_color.get(act,  "#888888")
    mark = "✓" if pred == act else "✗"
    mark_color = "#2ecc71" if pred == act else "#e74c3c"

    est_by_last = {c.replace("est_", ""): row.get(c) for c in est_cols}
    act_by_last = {c.replace("act_", ""): row.get(c) for c in act_cols}

    def _fv(v):
        return float(v) if v is not None and str(v) not in ("nan", "None") else None

    lasts = sorted(est_by_last, key=lambda l: _fv(est_by_last[l]) or -1.0, reverse=True)
    entries = [(l, _fv(est_by_last.get(l)), _fv(act_by_last.get(l))) for l in lasts]

    jf = row.get("JoinField", "")
    h  = f'<div style="font-family:monospace;font-size:12px;padding:6px 8px;min-width:240px">'
    h += f'<div style="color:#aaa;font-size:10px">{jf}</div>'
    h += f'<div><span style="color:{pc}">■</span> <b>{pred}</b> <span style="color:#aaa">model</span></div>'
    h += f'<div><span style="color:{ac}">■</span> <b>{act}</b> <span style="color:#aaa">actual</span> '
    h += f'<span style="color:{mark_color}">{mark}</span></div>'
    h += '<hr style="margin:5px 0;border:none;border-top:1px solid #444">'
    h += '<table style="border-collapse:collapse;width:100%">'
    h += '<tr><th style="text-align:left;color:#777;padding:1px 6px 1px 0">Candidate</th>'
    h += '<th style="color:#777;padding:1px 6px">Model</th><th style="color:#777;padding:1px 6px">Actual</th></tr>'
    for last, ev, av in entries:
        es  = f"{ev:.1f}%" if ev is not None else "—"
        as_ = f"{av:.1f}%" if av is not None else "—"
        h += f'<tr><td style="padding:1px 6px 1px 0">{last}</td>'
        h += f'<td style="text-align:right;padding:1px 6px">{es}</td>'
        h += f'<td style="text-align:right;padding:1px 6px;color:#aaa">{as_}</td></tr>'
    h += '</table></div>'
    return h


def _save_interactive_precinct_map(
    gdf: gpd.GeoDataFrame,
    cand_color: dict,
    save_dir: Path,
    title_suffix: str = "",
) -> None:
    if not _FOLIUM:
        print("  (folium not installed — skipping interactive HTML precinct map)")
        return

    plot = gdf[["geometry", "JoinField", "leader", "actual_leader", "tooltip_html"]].copy()
    plot["leader"]        = plot["leader"].fillna("Unknown")
    plot["actual_leader"] = plot["actual_leader"].fillna("Unknown")
    geojson = plot.to_json()

    m = folium.Map(location=[41.84, -87.65], zoom_start=11, tiles="CartoDB dark_matter")

    def _style(col):
        def fn(f):
            cand = f["properties"].get(col) or "Unknown"
            return {"fillColor": cand_color.get(cand, "#333333"),
                    "color": "none", "weight": 0, "fillOpacity": 0.82}
        return fn

    class _HtmlTooltip(MacroElement):
        def __init__(self, layer):
            super().__init__()
            self._layer = layer
            self._template = Template("""
                {% macro script(this, kwargs) %}
                {{ this._layer.get_name() }}.eachLayer(function(layer) {
                    if (layer.feature && layer.feature.properties.tooltip_html) {
                        layer.bindTooltip(layer.feature.properties.tooltip_html,
                            {sticky: true, maxWidth: 320, opacity: 0.97});
                    }
                });
                {% endmacro %}
            """)

    model_group  = folium.FeatureGroup(name="Model prediction", show=True).add_to(m)
    actual_group = folium.FeatureGroup(name="Actual results",  show=False).add_to(m)
    miss_group   = folium.FeatureGroup(name="Wrong predictions (actual winner)", show=False).add_to(m)

    # Ward boundaries for spatial context
    if WARD_SHP.exists():
        wards = gpd.read_file(WARD_SHP).to_crs("EPSG:4326")
        folium.GeoJson(
            wards.to_json(),
            style_function=lambda _: {"fillColor": "none", "color": "white",
                                      "weight": 0.6, "fillOpacity": 0},
            name="Ward boundaries",
        ).add_to(m)

    for group, col in [(model_group, "leader"), (actual_group, "actual_leader")]:
        layer = folium.GeoJson(geojson, style_function=_style(col))
        layer.add_to(group)
        layer.add_child(_HtmlTooltip(layer))

    def _style_misses(f):
        props = f.get("properties", {})
        pred = props.get("leader") or "Unknown"
        act  = props.get("actual_leader") or "Unknown"
        if pred != act:
            return {"fillColor": cand_color.get(act, "#ff6600"),
                    "color": "white", "weight": 0.5, "fillOpacity": 0.9}
        return {"fillColor": "#111111", "color": "none", "weight": 0, "fillOpacity": 0.15}

    miss_layer = folium.GeoJson(geojson, style_function=_style_misses)
    miss_layer.add_to(miss_group)
    miss_layer.add_child(_HtmlTooltip(miss_layer))

    folium.LayerControl(collapsed=False).add_to(m)

    out = save_dir / "precinct_map.html"
    m.save(str(out))
    print(f"Saved → {out}")


def main():
    precinct_csv = CONFIG.output_dir / f"{CONFIG.race_id}_precinct_probabilities.csv"
    draw_precinct_map(precinct_csv, CONFIG.output_dir)
    plt.show()


if __name__ == "__main__":
    main()
