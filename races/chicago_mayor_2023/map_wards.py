"""
Ward-level choropleth: model forecast vs actual results.
Run from repo root: python -m races.chicago_mayor_2023.map_wards
"""
from __future__ import annotations

import json
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

WARDS_SHP = Path(__file__).parent.parent.parent / "Shapefiles" / "chicago_wards.geojson"

COLORS = CONFIG.colors


def load_actual_ward_leaders() -> pd.DataFrame:
    """Returns ward, actual_leader, and act_{LastName} float columns (0-100 share)."""
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

    def ward_num(jf: str) -> str | None:
        parts = jf.split(":", 1)[-1].strip().split()
        if len(parts) >= 2 and parts[0].upper() == "WARD":
            try:
                return str(int(parts[1]))
            except ValueError:
                pass
        return None

    df["ward"] = df["JoinField"].apply(ward_num)
    df = df.dropna(subset=["ward"])

    totals = df.groupby(["ward", "candidate"])["votes"].sum().reset_index()
    wide = totals.pivot_table(index="ward", columns="candidate", values="votes", fill_value=0)
    wide.columns.name = None
    cand_cols = list(wide.columns)
    ward_total = wide[cand_cols].sum(axis=1).replace(0, float("nan"))
    wide["actual_leader"] = wide[cand_cols].idxmax(axis=1)
    for cand in cand_cols:
        wide[f"act_{cand.split()[-1]}"] = (wide[cand] / ward_total * 100).round(1)
    wide = wide.reset_index()
    act_cols = [c for c in wide.columns if c.startswith("act_")]
    return wide[["ward", "actual_leader"] + act_cols]


def draw_ward_map(regional_json: Path, save_dir: Path, title_suffix: str = "") -> None:
    """
    Draw a two-panel ward choropleth (model vs actual) and save to save_dir/ward_map.png.
    regional_json: path to regional_vote_forecast.json for this snapshot.
    """
    with regional_json.open() as f:
        forecast = json.load(f)

    rows = []
    for ward_label, data in forecast.items():
        shares = data.get("vote_shares", {})
        if not shares:
            continue
        leader = max(shares, key=shares.get)
        row = {"ward": str(int(ward_label.split()[-1])), "leader": leader}
        for cand, share in shares.items():
            row[f"est_{cand.split()[-1]}"] = round(share * 100, 1)
        rows.append(row)
    model_df = pd.DataFrame(rows)
    est_cols  = [c for c in model_df.columns if c.startswith("est_")]

    actual_df = load_actual_ward_leaders()
    act_cols  = [c for c in actual_df.columns if c.startswith("act_")]

    gdf = gpd.read_file(WARDS_SHP)
    gdf["ward"] = gdf["ward"].astype(str)
    gdf = gdf.merge(model_df[["ward", "leader"] + est_cols], on="ward", how="left")
    gdf = gdf.merge(actual_df[["ward", "actual_leader"] + act_cols], on="ward", how="left")
    gdf = gdf.to_crs("EPSG:4326")

    all_leaders = sorted(set(gdf["leader"].dropna()) | set(gdf["actual_leader"].dropna()))
    cand_color  = {c: COLORS.get(c, "#888888") for c in all_leaders}

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.set_axis_off()

    def plot_panel(ax, leader_col, title):
        for cand in all_leaders:
            sub = gdf[gdf[leader_col] == cand]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=cand_color[cand], edgecolor="white", linewidth=0.4)
        na_wards = gdf[gdf[leader_col].isna()]
        if not na_wards.empty:
            na_wards.plot(ax=ax, color="#555", edgecolor="white", linewidth=0.4)
        leaders_present = gdf[leader_col].dropna().unique()
        patches = [
            mpatches.Patch(color=cand_color[c], label=c)
            for c in sorted(leaders_present, key=lambda c: -(gdf[leader_col] == c).sum())
        ]
        ax.legend(handles=patches, loc="lower left", fontsize=8,
                  framealpha=0.3, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="none")
        for _, row in gdf.iterrows():
            if row.geometry is None:
                continue
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.annotate(row["ward"], (cx, cy),
                        ha="center", va="center", fontsize=5, color="white", alpha=0.7)
        ax.set_title(title, color="white", fontsize=12, pad=8)

    plot_panel(axes[0], "leader",        "Model: leading candidate by ward")
    plot_panel(axes[1], "actual_leader", "Actual: leading candidate by ward")

    suptitle = "Chicago Mayor 2023 — Ward-level forecast vs. actual"
    if title_suffix:
        suptitle += f"  ({title_suffix})"
    fig.suptitle(suptitle, color="white", fontsize=14, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "ward_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")

    gdf["tooltip_html"] = gdf.apply(
        lambda row: _build_ward_tooltip(row, cand_color, est_cols, act_cols), axis=1
    )
    _save_interactive_ward_map(gdf, cand_color, save_dir, title_suffix)


def _build_ward_tooltip(row: pd.Series, cand_color: dict, est_cols: list, act_cols: list) -> str:
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

    lasts   = sorted(est_by_last, key=lambda l: _fv(est_by_last[l]) or -1.0, reverse=True)
    entries = [(l, _fv(est_by_last.get(l)), _fv(act_by_last.get(l))) for l in lasts]

    h  = f'<div style="font-family:monospace;font-size:12px;padding:6px 8px;min-width:240px">'
    h += f'<div><span style="color:{pc}">■</span> <b>{pred}</b> <span style="color:#aaa">model</span></div>'
    h += f'<div><span style="color:{ac}">■</span> <b>{act}</b> <span style="color:#aaa">actual</span> '
    h += f'<span style="color:{mark_color}">{mark}</span></div>'
    h += f'<div style="color:#aaa;font-size:11px">Ward {row.get("ward", "")}</div>'
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


def _save_interactive_ward_map(
    gdf: gpd.GeoDataFrame,
    cand_color: dict,
    save_dir: Path,
    title_suffix: str = "",
) -> None:
    if not _FOLIUM:
        print("  (folium not installed — skipping interactive HTML ward map)")
        return

    plot = gdf[["geometry", "ward", "leader", "actual_leader", "tooltip_html"]].copy()
    plot["leader"]        = plot["leader"].fillna("Unknown")
    plot["actual_leader"] = plot["actual_leader"].fillna("Unknown")
    geojson = plot.to_json()

    m = folium.Map(location=[41.84, -87.65], zoom_start=11, tiles="CartoDB dark_matter")

    def _style(col):
        def fn(f):
            cand = f["properties"].get(col) or "Unknown"
            return {"fillColor": cand_color.get(cand, "#555555"),
                    "color": "white", "weight": 0.8, "fillOpacity": 0.78}
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
                    "color": "white", "weight": 0.8, "fillOpacity": 0.9}
        return {"fillColor": "#111111", "color": "none", "weight": 0, "fillOpacity": 0.15}

    miss_layer = folium.GeoJson(geojson, style_function=_style_misses)
    miss_layer.add_to(miss_group)
    miss_layer.add_child(_HtmlTooltip(miss_layer))

    folium.LayerControl(collapsed=False).add_to(m)

    out = save_dir / "ward_map.html"
    m.save(str(out))
    print(f"Saved → {out}")


def main():
    regional_json = CONFIG.output_dir / "regional_vote_forecast.json"
    draw_ward_map(regional_json, CONFIG.output_dir)
    plt.show()


if __name__ == "__main__":
    main()
