"""
Build an interactive HTML choropleth of predicted 2027 mayoral vote shares
per precinct, calibrated from the 2025 poll topline + crosstabs.

Prediction model: logit-space demographic calibration (core.precinct_calibration).
  Race and ideology crosstab signals combine in logit space so conflicting
  signals compete multiplicatively rather than stacking additively.

Win probabilities are computed via the Rust precinct Monte Carlo simulator
(10k trials per precinct, moe_precinct=8pp additional local uncertainty).

Outputs: outputs/chicago_mayor_2027_precinct_map.html
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.db import get_progressive_scores
from core import simulator_runner
from core.precinct_calibration import enrich_precinct_df, compute_precinct_shares

# ── Constants ───────────────────────────────────────────────────────────────
DB_PATH      = Path("/home/cole/databases/illinois_elections.db")
SHP_PATH     = Path("/home/cole/elections/il9prediction_and_tracker/data/shapefile/IL24/IL24.shp")
OUT_DIR      = Path(__file__).parent / "outputs"
SIMPLIFY     = 0.0003   # degrees; ~30m — keeps file small
MOE          = 3.7      # district-level poll MOE (pp)
MOE_PRECINCT = 8.0      # additional precinct-level uncertainty (pp)
N_SIM        = 10_000   # precinct simulation trials


CANDIDATES = [
    "Vallas", "Giannoulias", "Mendoza", "Johnson", "Buckner",
    "Wilson", "Conway", "Gutierrez", "Beale", "Green", "Villegas",
]

BLOCS = [
    ["Johnson", "Buckner", "Green", "Gutierrez"],
    ["Giannoulias", "Mendoza", "Conway"],
    ["Vallas", "Wilson", "Beale", "Villegas"],
]

COLORS = {
    "Vallas":      "#d32f2f",
    "Giannoulias": "#1565c0",
    "Mendoza":     "#6a1fa0",
    "Johnson":     "#2e7d32",
    "Buckner":     "#00838f",
    "Wilson":      "#e65100",
    "Conway":      "#4e342e",
    "Gutierrez":   "#ad1457",
    "Beale":       "#558b2f",
    "Green":       "#1b5e20",
    "Villegas":    "#880e4f",
}

TOPLINE = {
    "Vallas": 27.4, "Giannoulias": 21.0, "Mendoza": 11.7, "Johnson": 8.2,
    "Buckner": 6.3, "Wilson": 5.9, "Conway": 5.7, "Gutierrez": 5.1,
    "Beale": 3.9, "Green": 2.9, "Villegas": 2.0,
}

# Ideology crosstab: candidate → [very_con, somewhat_con, moderate, somewhat_lib, very_lib]
IDEO = {
    "Vallas":      [59, 46, 32, 22,  5],
    "Giannoulias": [ 3, 10, 19, 29, 27],
    "Mendoza":     [ 3, 10, 12, 14, 12],
    "Johnson":     [ 2, 10,  4,  9, 17],
    "Buckner":     [ 0,  1,  2,  8, 17],
    "Wilson":      [27,  8,  8,  1,  1],
    "Conway":      [ 1,  6,  7,  8,  1],
    "Gutierrez":   [ 0,  5,  4,  2, 13],
    "Beale":       [ 1,  1,  6,  5,  0],
    "Green":       [ 4,  2,  2,  1,  7],
    "Villegas":    [ 0,  0,  4,  2,  1],
}

# Race crosstab: candidate → {race: pct}
RACE_CT = {
    "Vallas":      {"black": 22, "hispanic": 26, "white": 31},
    "Giannoulias": {"black": 18, "hispanic": 14, "white": 24},
    "Mendoza":     {"black":  6, "hispanic":  9, "white": 17},
    "Johnson":     {"black": 18, "hispanic":  2, "white":  6},
    "Buckner":     {"black":  4, "hispanic":  4, "white":  7},
    "Wilson":      {"black": 13, "hispanic":  3, "white":  3},
    "Conway":      {"black":  1, "hispanic": 15, "white":  5},
    "Gutierrez":   {"black":  5, "hispanic": 16, "white":  1},
    "Beale":       {"black":  7, "hispanic":  8, "white":  1},
    "Green":       {"black":  4, "hispanic":  0, "white":  2},
    "Villegas":    {"black":  3, "hispanic":  2, "white":  2},
}




def load_data() -> pd.DataFrame:
    scores = get_progressive_scores("chicago_mayor", scenario="generic")[
        ["JoinField", "score_pp"]
    ]
    scores["jf_upper"] = scores["JoinField"].str.upper()

    with sqlite3.connect(str(DB_PATH)) as conn:
        demo = pd.read_sql_query("""
            SELECT JoinField,
                   total,
                   total_hispanic_or_latino                                      AS hispanic,
                   total_not_hispanic_or_latino_black_or_african_american_alone  AS black,
                   total_not_hispanic_or_latino_white_alone                      AS white
            FROM precinct_demographics
            WHERE year = 2022 AND JoinField LIKE 'CITY OF CHICAGO:%'
        """, conn)

    t = demo["total"].replace(0, np.nan)
    demo["pct_black"]    = demo["black"]    / t
    demo["pct_hispanic"] = demo["hispanic"] / t
    demo["pct_white"]    = demo["white"]    / t
    demo["pct_other"]    = (1 - demo["pct_black"] - demo["pct_hispanic"] - demo["pct_white"]).clip(0)
    demo["jf_upper"]     = demo["JoinField"].str.upper()

    df = scores.merge(demo[["jf_upper","pct_black","pct_hispanic","pct_white","pct_other","total"]],
                      on="jf_upper", how="left")
    return df


def _scale_crosstabs(
    crosstabs: dict,
    old_topline: dict[str, float],
    new_topline: dict[str, float],
) -> dict:
    """
    Adjust crosstabs to be consistent with a new topline.
    For each group, the per-candidate delta (old_crosstab_pct - old_topline_pct)
    is preserved and applied to the new topline. Values are clipped to [0, 100]
    then renormalized so each group sums to 100%.
    """
    result = {}
    for group, shares in crosstabs.items():
        new_group: dict[str, float] = {}
        for cand in CANDIDATES:
            delta = (shares.get(cand) or 0.0) - old_topline.get(cand, 0.0)
            new_group[cand] = max(0.0, new_topline[cand] + delta)
        total = sum(new_group.values())
        if total > 1e-9:
            new_group = {c: v * 100.0 / total for c, v in new_group.items()}
        result[group] = new_group
    return result


def _build_demo_crosstabs(
    topline: dict[str, float],
    ideo: dict[str, list[float]],
    race_ct: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Convert pp-scale poll data to the fraction format compute_precinct_shares expects."""
    ct: dict[str, dict[str, float]] = {}
    for group in ["black", "hispanic", "white"]:
        ct[group] = {c: race_ct[c][group] / 100.0 for c in CANDIDATES}
    ideo_names = [
        "very_conservative", "somewhat_conservative", "moderate",
        "somewhat_liberal", "very_liberal",
    ]
    for idx, name in enumerate(ideo_names):
        ct[name] = {c: ideo[c][idx] / 100.0 for c in CANDIDATES}
    return ct


def predict(
    df: pd.DataFrame,
    topline: dict[str, float],
    demo_crosstabs: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Predict per-precinct vote shares using core.precinct_calibration."""
    baseline_f = {c: topline[c] / 100.0 for c in CANDIDATES}
    shares = compute_precinct_shares(
        df, demo_crosstabs, baseline_f, CANDIDATES, weight_col="total",
    )
    out = df[["JoinField", "score_pp"]].copy()
    for cand in CANDIDATES:
        out[cand] = shares[cand].values
    out["predicted_winner"] = shares["predicted_winner"].values
    out["total"] = df["total"].fillna(1).values.astype(int)
    return out


def compute_win_probs(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Run precinct Monte Carlo and return {precinct_id: {candidate: win_prob}}."""
    precincts_input = []
    for _, row in predictions.iterrows():
        baseline = {c: float(row[c]) / 100.0 for c in CANDIDATES}
        tw = int(row.get("total", 1) or 1)
        precincts_input.append({
            "id": row["JoinField"],
            "baseline": baseline,
            "turnout_weight": max(tw, 1),
        })

    result = simulator_runner.run_precinct_sim(
        n_simulations=N_SIM,
        candidates=CANDIDATES,
        moe_district=MOE,
        moe_precinct=MOE_PRECINCT,
        ideological_blocs=BLOCS,
        precincts=precincts_input,
    )

    return {p["id"]: p["win_probs"] for p in result["precincts"]}


def build_geojson(predictions: pd.DataFrame, win_probs: dict[str, dict[str, float]]) -> str:
    gdf = gpd.read_file(str(SHP_PATH)).to_crs(epsg=4326)
    gdf = gdf[gdf["JoinField"].str.startswith("CITY OF CHICAGO:", na=False)].copy()
    gdf["geometry"] = gdf["geometry"].simplify(SIMPLIFY, preserve_topology=True)
    gdf["jf_upper"] = gdf["JoinField"].str.upper()

    predictions["jf_upper"] = predictions["JoinField"].str.upper()
    gdf = gdf.merge(
        predictions.drop(columns=["JoinField"]),
        on="jf_upper", how="left"
    )

    features = []
    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        jf = row["JoinField"]
        wp = win_probs.get(jf, {})
        props = {
            "jf": jf,
            "winner": row.get("predicted_winner", ""),
            "prog_score": round(float(row["score_pp"]), 1) if pd.notna(row.get("score_pp")) else None,
        }
        for cand in CANDIDATES:
            v = row.get(cand)
            props[cand] = round(float(v), 1) if pd.notna(v) else None
            wv = wp.get(cand)
            props[f"wp_{cand}"] = round(float(wv) * 100, 1) if wv is not None else None

        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": props,
        })

    return json.dumps({"type": "FeatureCollection", "features": features})


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Chicago Mayor 2027 — Precinct Forecast</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #1a1a2e; color: #e0e0e0; display: flex; flex-direction: column; height: 100vh; }}
#header {{ padding: 10px 16px; background: #12122a; border-bottom: 1px solid #333;
          display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
#header h1 {{ font-size: 15px; font-weight: 600; color: #fff; white-space: nowrap; }}
#header label {{ font-size: 12px; color: #aaa; margin-right: 4px; }}
select, input[type=radio] {{ background: #252545; color: #e0e0e0; border: 1px solid #444;
                             border-radius: 4px; padding: 4px 8px; font-size: 12px; cursor: pointer; }}
.radio-group {{ display: flex; gap: 12px; align-items: center; font-size: 12px; }}
.radio-group label {{ color: #ccc; cursor: pointer; }}
#map {{ flex: 1; }}
#tooltip {{ position: fixed; background: #1e1e3a; border: 1px solid #444; border-radius: 6px;
           padding: 10px 12px; font-size: 12px; pointer-events: none; display: none;
           z-index: 9999; min-width: 220px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
#tooltip .tt-title {{ font-weight: 600; color: #fff; margin-bottom: 6px; font-size: 11px;
                      border-bottom: 1px solid #333; padding-bottom: 4px; }}
#tooltip .tt-header {{ display: flex; justify-content: space-between; gap: 16px;
                       color: #666; font-size: 10px; padding-bottom: 2px; }}
#tooltip .tt-row {{ display: flex; justify-content: space-between; gap: 16px;
                   padding: 1px 0; color: #ccc; }}
#tooltip .tt-row.winner {{ color: #fff; font-weight: 600; }}
#tooltip .tt-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block;
                   margin-right: 5px; flex-shrink: 0; }}
#tooltip .tt-vals {{ display: flex; gap: 10px; }}
#tooltip .tt-val {{ min-width: 38px; text-align: right; }}
#legend {{ position: absolute; bottom: 24px; right: 12px; background: #1e1e3a;
          border: 1px solid #444; border-radius: 6px; padding: 10px 12px; z-index: 999;
          font-size: 11px; min-width: 140px; }}
#legend.hidden {{ display: none; }}
#legend .leg-title {{ font-weight: 600; color: #fff; margin-bottom: 6px; }}
#legend .leg-row {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; color: #ccc; }}
#legend .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
#gradient-legend {{ position: absolute; bottom: 24px; right: 12px; background: #1e1e3a;
                   border: 1px solid #444; border-radius: 6px; padding: 10px 12px; z-index: 999;
                   font-size: 11px; min-width: 180px; }}
#gradient-legend.hidden {{ display: none; }}
#gradient-legend .gl-title {{ font-weight: 600; color: #fff; margin-bottom: 6px; }}
#gradient-legend .gl-bar {{ height: 12px; border-radius: 3px; margin: 4px 0; }}
#gradient-legend .gl-labels {{ display: flex; justify-content: space-between; color: #aaa; }}
#note {{ font-size: 10px; color: #666; padding: 4px 16px; background: #12122a; }}
</style>
</head>
<body>
<div id="header">
  <h1>Chicago Mayor 2027 — Precinct Forecast</h1>
  <div>
    <label>View</label>
    <div class="radio-group">
      <label><input type="radio" name="view" value="winner" checked> Predicted winner</label>
      <label><input type="radio" name="view" value="share"> Vote share</label>
      <label><input type="radio" name="view" value="winprob"> Win probability</label>
    </div>
  </div>
  <div id="cand-select-wrap" style="display:none">
    <label>Candidate</label>
    <select id="cand-select">{candidate_options}</select>
  </div>
</div>
<div id="map"></div>
<div id="note">
  Model: poll topline + race &amp; ideology crosstab calibration (2025 poll, n=697, MOE ±3.7pp).
  Win probabilities from Monte Carlo precinct sim (10k trials, ±8pp precinct uncertainty).
  Top-line numbers too early to be reliable — treat as illustrative spatial prior.
</div>
<div id="tooltip"></div>
<div id="legend">
  <div class="leg-title">Predicted winner</div>
  {legend_rows}
</div>
<div id="gradient-legend" class="hidden">
  <div class="gl-title" id="gl-title">Vote share: Vallas</div>
  <div class="gl-bar" id="gl-bar"></div>
  <div class="gl-labels"><span id="gl-lo">0%</span><span id="gl-mid"></span><span id="gl-hi"></span></div>
</div>

<script>
const GEOJSON = {geojson};
const CANDIDATES = {candidates_json};
const COLORS = {colors_json};
const TOPLINE = {topline_json};

const map = L.map('map', {{ zoomControl: true }}).setView([41.838, -87.680], 11);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19
}}).addTo(map);

let geoLayer = null;
let currentView = 'winner';
let currentCand = CANDIDATES[0];

const tooltip = document.getElementById('tooltip');

function hexToRgb(hex) {{
  return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
}}

function shareColor(pct, cand) {{
  const [r,g,b] = hexToRgb(COLORS[cand]);
  const alpha = Math.max(0.08, Math.min(0.95, pct / 45));
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

function winProbColor(prob, cand) {{
  const [r,g,b] = hexToRgb(COLORS[cand]);
  // prob is 0-100; scale so 100% = full opacity, 0% = nearly transparent
  const alpha = Math.max(0.05, Math.min(0.95, prob / 100));
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

function winnerColor(winner) {{
  return winner ? COLORS[winner] || '#444' : '#333';
}}

function styleFeature(feature) {{
  const p = feature.properties;
  if (currentView === 'winner') {{
    return {{ fillColor: winnerColor(p.winner), fillOpacity: 0.75,
              color: '#111', weight: 0.3 }};
  }} else if (currentView === 'share') {{
    const pct = p[currentCand] || 0;
    return {{ fillColor: shareColor(pct, currentCand), fillOpacity: 0.85,
              color: '#111', weight: 0.3 }};
  }} else {{
    const prob = p['wp_' + currentCand] || 0;
    return {{ fillColor: winProbColor(prob, currentCand), fillOpacity: 0.85,
              color: '#111', weight: 0.3 }};
  }}
}}

function buildTooltip(p) {{
  const sorted = [...CANDIDATES].sort((a,b) => (p[b]||0) - (p[a]||0));
  const hasWp = p['wp_' + CANDIDATES[0]] != null;
  const hdr = hasWp
    ? `<div class="tt-header"><span></span><div class="tt-vals"><span class="tt-val">Share</span><span class="tt-val">Win%</span></div></div>`
    : `<div class="tt-header"><span></span><div class="tt-vals"><span class="tt-val">Share</span></div></div>`;
  let rows = sorted.map(c => {{
    const share = p[c] != null ? p[c].toFixed(1)+'%' : '—';
    const wp    = hasWp && p['wp_'+c] != null ? p['wp_'+c].toFixed(0)+'%' : '—';
    const cls   = c === p.winner ? 'tt-row winner' : 'tt-row';
    const vals  = hasWp
      ? `<div class="tt-vals"><span class="tt-val">${{share}}</span><span class="tt-val">${{wp}}</span></div>`
      : `<div class="tt-vals"><span class="tt-val">${{share}}</span></div>`;
    return `<div class="${{cls}}">
      <span><span class="tt-dot" style="background:${{COLORS[c]}}"></span>${{c}}</span>
      ${{vals}}</div>`;
  }}).join('');
  const prog = p.prog_score != null
    ? `<div style="color:#888;margin-top:5px;font-size:10px">Prog. lean: ${{p.prog_score > 0 ? '+' : ''}}${{p.prog_score}}pp</div>`
    : '';
  return `<div class="tt-title">${{p.jf ? p.jf.replace('CITY OF CHICAGO:','') : ''}}</div>${{hdr}}${{rows}}${{prog}}`;
}}

function onEachFeature(feature, layer) {{
  layer.on({{
    mousemove: function(e) {{
      tooltip.innerHTML = buildTooltip(feature.properties);
      tooltip.style.display = 'block';
      tooltip.style.left = (e.originalEvent.pageX + 14) + 'px';
      tooltip.style.top  = (e.originalEvent.pageY - 10) + 'px';
    }},
    mouseout: function() {{ tooltip.style.display = 'none'; }},
  }});
}}

function redraw() {{
  if (geoLayer) geoLayer.setStyle(styleFeature);
  updateLegend();
}}

function updateLegend() {{
  const leg = document.getElementById('legend');
  const gl  = document.getElementById('gradient-legend');
  if (currentView === 'winner') {{
    leg.classList.remove('hidden');
    gl.classList.add('hidden');
  }} else {{
    leg.classList.add('hidden');
    gl.classList.remove('hidden');
    const [r,g,b] = hexToRgb(COLORS[currentCand]);
    if (currentView === 'share') {{
      document.getElementById('gl-title').textContent = 'Vote share: ' + currentCand;
      document.getElementById('gl-bar').style.background =
        `linear-gradient(to right, rgba(${{r}},${{g}},${{b}},0.08), rgba(${{r}},${{g}},${{b}},0.95))`;
      document.getElementById('gl-lo').textContent  = '0%';
      document.getElementById('gl-mid').textContent = TOPLINE[currentCand].toFixed(1) + '% avg';
      document.getElementById('gl-hi').textContent  = '45%';
    }} else {{
      document.getElementById('gl-title').textContent = 'Win probability: ' + currentCand;
      document.getElementById('gl-bar').style.background =
        `linear-gradient(to right, rgba(${{r}},${{g}},${{b}},0.05), rgba(${{r}},${{g}},${{b}},0.95))`;
      document.getElementById('gl-lo').textContent  = '0%';
      document.getElementById('gl-mid').textContent = '';
      document.getElementById('gl-hi').textContent  = '100%';
    }}
  }}
}}

geoLayer = L.geoJSON(GEOJSON, {{ style: styleFeature, onEachFeature }}).addTo(map);
updateLegend();

document.querySelectorAll('input[name=view]').forEach(r => {{
  r.addEventListener('change', function() {{
    currentView = this.value;
    document.getElementById('cand-select-wrap').style.display =
      (currentView === 'share' || currentView === 'winprob') ? 'block' : 'none';
    redraw();
  }});
}});

document.getElementById('cand-select').addEventListener('change', function() {{
  currentCand = this.value;
  redraw();
}});
</script>
</body>
</html>"""


def build_html(
    geojson_str: str,
    topline: dict[str, float],
    note: str,
    colors: dict[str, str] | None = None,
    legend_items: list[tuple[str, str]] | None = None,
) -> str:
    if colors is None:
        colors = COLORS
    if legend_items is None:
        legend_items = [(c, colors[c]) for c in CANDIDATES]

    opts = "\n".join(
        f'<option value="{c}">{c} ({topline[c]:.1f}%)</option>'
        for c in CANDIDATES
    )
    leg_rows = "\n".join(
        f'<div class="leg-row"><div class="leg-dot" style="background:{color}"></div>'
        f'<span>{label}</span></div>'
        for label, color in legend_items
    )
    template = HTML_TEMPLATE.replace(
        "Top-line numbers too early to be reliable — treat as illustrative spatial prior.",
        note,
    )
    return template.format(
        geojson=geojson_str,
        candidates_json=json.dumps(CANDIDATES),
        colors_json=json.dumps(colors),
        topline_json=json.dumps(topline),
        candidate_options=opts,
        legend_rows=leg_rows,
    )


def _build_scenario(
    name: str,
) -> tuple[dict[str, float], dict[str, dict[str, float]], str, str, dict[str, str], list[tuple[str, str]]]:
    """
    Returns (topline, demo_crosstabs, out_filename, map_note, colors, legend_items).
    Scenarios:
      "poll"     — as polled (default)
      "deadheat" — top 5 at 15% each, others scaled proportionally
      "blocs"    — conservative bloc vs progressive/center-left bloc, each scaled to 50%
    """
    if name == "blocs":
        # Two super-blocs derived from BLOCS:
        #   conservative  = Vallas, Wilson, Beale, Villegas          (poll: ~39%)
        #   left/center   = Giannoulias, Mendoza, Conway,            (poll: ~61%)
        #                   Johnson, Buckner, Green, Gutierrez
        # Each candidate's share is scaled proportionally within their bloc so both
        # blocs hit exactly 50%, preserving internal coalition ratios throughout.
        conservative = ["Vallas", "Wilson", "Beale", "Villegas"]
        left_center  = ["Giannoulias", "Mendoza", "Conway", "Johnson", "Buckner", "Green", "Gutierrez"]

        con_total  = sum(TOPLINE[c] for c in conservative)
        left_total = sum(TOPLINE[c] for c in left_center)

        topline = {}
        for c in conservative:
            topline[c] = TOPLINE[c] / con_total * 50.0
        for c in left_center:
            topline[c] = TOPLINE[c] / left_total * 50.0

        poll_ideo = {
            g: {c: IDEO[c][i] for c in CANDIDATES}
            for i, g in enumerate([
                "very_conservative", "somewhat_conservative", "moderate",
                "somewhat_liberal", "very_liberal",
            ])
        }
        poll_race = {
            g: {c: RACE_CT[c][g] for c in CANDIDATES}
            for g in ["black", "hispanic", "white"]
        }
        scaled_ideo = _scale_crosstabs(poll_ideo, TOPLINE, topline)
        scaled_race = _scale_crosstabs(poll_race, TOPLINE, topline)

        ideo_for_build = {
            c: [
                scaled_ideo[g][c]
                for g in ["very_conservative", "somewhat_conservative",
                          "moderate", "somewhat_liberal", "very_liberal"]
            ]
            for c in CANDIDATES
        }
        race_for_build = {
            c: {g: scaled_race[g][c] for g in ["black", "hispanic", "white"]}
            for c in CANDIDATES
        }
        demo_ct  = _build_demo_crosstabs(topline, ideo_for_build, race_for_build)
        filename = "chicago_mayor_2027_precinct_map_blocs.html"
        note     = (
            f"Hypothetical 50/50 bloc split: conservative bloc (Vallas {topline['Vallas']:.1f}%, "
            f"Wilson {topline['Wilson']:.1f}%, Beale {topline['Beale']:.1f}%, "
            f"Villegas {topline['Villegas']:.1f}%) vs. progressive/center-left bloc "
            f"(Giannoulias {topline['Giannoulias']:.1f}%, Mendoza {topline['Mendoza']:.1f}%, "
            f"Johnson {topline['Johnson']:.1f}%, others). "
            "Internal coalition ratios preserved from 2025 poll (n=697)."
        )
        con_color  = "#d32f2f"
        left_color = "#1565c0"
        colors = {c: con_color for c in conservative} | {c: left_color for c in left_center}
        legend_items = [
            ("Conservative (Vallas bloc)", con_color),
            ("Progressive / Center-left", left_color),
        ]
        return topline, demo_ct, filename, note, colors, legend_items

    if name == "deadheat":
        top5   = ["Vallas", "Giannoulias", "Mendoza", "Johnson", "Buckner"]
        others = [c for c in CANDIDATES if c not in top5]
        others_old_sum  = sum(TOPLINE[c] for c in others)
        others_remaining = 100.0 - 5 * 15.0
        topline = {c: 15.0 for c in top5}
        for c in others:
            topline[c] = TOPLINE[c] / others_old_sum * others_remaining

        poll_ideo = {
            g: {c: IDEO[c][i] for c in CANDIDATES}
            for i, g in enumerate([
                "very_conservative", "somewhat_conservative", "moderate",
                "somewhat_liberal", "very_liberal",
            ])
        }
        poll_race = {
            g: {c: RACE_CT[c][g] for c in CANDIDATES}
            for g in ["black", "hispanic", "white"]
        }
        scaled_ideo = _scale_crosstabs(poll_ideo, TOPLINE, topline)
        scaled_race = _scale_crosstabs(poll_race, TOPLINE, topline)

        ideo_for_build = {
            c: [
                scaled_ideo[g][c]
                for g in ["very_conservative", "somewhat_conservative",
                          "moderate", "somewhat_liberal", "very_liberal"]
            ]
            for c in CANDIDATES
        }
        race_for_build = {
            c: {g: scaled_race[g][c] for g in ["black", "hispanic", "white"]}
            for c in CANDIDATES
        }
        demo_ct  = _build_demo_crosstabs(topline, ideo_for_build, race_for_build)
        filename = "chicago_mayor_2027_precinct_map_deadheat.html"
        note     = ("Hypothetical dead heat: top 5 candidates each at 15%. "
                    "Crosstabs scaled via delta method from 2025 poll (n=697).")
    else:
        topline  = TOPLINE
        demo_ct  = _build_demo_crosstabs(TOPLINE, IDEO, RACE_CT)
        filename = "chicago_mayor_2027_precinct_map.html"
        note     = ("Model: poll topline + race &amp; ideology crosstab calibration "
                    "(2025 poll, n=697, MOE ±3.7pp). "
                    "Win probabilities from Monte Carlo precinct sim (10k trials, ±8pp precinct uncertainty). "
                    "Top-line numbers too early to be reliable — treat as illustrative spatial prior.")
    return topline, demo_ct, filename, note, COLORS, [(c, COLORS[c]) for c in CANDIDATES]


def main() -> None:
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "poll"
    if scenario not in ("poll", "deadheat", "blocs"):
        print(f"Unknown scenario '{scenario}'. Use: poll | deadheat | blocs")
        sys.exit(1)

    topline, demo_crosstabs, filename, note, colors, legend_items = _build_scenario(scenario)
    out_path = OUT_DIR / filename

    print(f"Scenario: {scenario}")
    print("Loading precinct data...")
    df = load_data()
    print(f"  {len(df)} precincts with progressive scores")

    print("Computing vote share predictions...")
    predictions = predict(df, topline, demo_crosstabs)

    # For the blocs scenario, the individual-plurality winner is misleading:
    # Vallas alone (~35%) would beat 7 split opponents even at 50/50 overall.
    # Instead, sum each bloc's total share per precinct and assign winner by bloc.
    # "Vallas" and "Giannoulias" are used as the winner labels so the bloc colors
    # in the JS color lookup resolve correctly.
    if scenario == "blocs":
        conservative = ["Vallas", "Wilson", "Beale", "Villegas"]
        left_center  = ["Giannoulias", "Mendoza", "Conway", "Johnson", "Buckner", "Green", "Gutierrez"]
        con_sum  = predictions[[c for c in conservative]].sum(axis=1)
        left_sum = predictions[[c for c in left_center]].sum(axis=1)
        predictions["predicted_winner"] = np.where(con_sum >= left_sum, "Vallas", "Giannoulias")

    winner_counts = predictions["predicted_winner"].value_counts()
    print("  Predicted winners:")
    for cand, n in winner_counts.items():
        print(f"    {cand:<14} {n:4d} precincts  "
              f"(avg share {predictions.loc[predictions.predicted_winner==cand, cand].mean():.1f}%)")

    print(f"Running precinct simulations ({N_SIM:,} trials × {len(predictions)} precincts)...")
    win_probs = compute_win_probs(predictions)
    print(f"  Done ({len(win_probs)} precincts)")

    print("Building GeoJSON...")
    geojson_str = build_geojson(predictions, win_probs)
    print(f"  {len(geojson_str)/1024:.0f} KB")

    print("Writing HTML...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(geojson_str, topline, note, colors, legend_items), encoding="utf-8")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
