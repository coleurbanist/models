"""
Pollster quality ratings from external sources.

Loads ratings from one or more external CSVs (Pollscore, Silver Bulletin,
Votehub, etc.), normalizes each source to [0.4, 1.0], and averages across
sources for pollsters that appear in multiple.

Outputs: pollster_ratings.json
  {
    pollster_id: {
      quality:                  float (0.4–1.0, higher = more accurate)
      house_effect_adjustment:  {candidate: pp, ...}  (positive = pollster overestimates)
      source:                   "external"
      sources_used:             [str, ...]
    }
  }

Usage
─────
python pollster_calibration.py \
    --external data/pollscore_ratings.csv data/silver_bulletin_ratings.csv \
    --output pollster_ratings.json

External ratings CSV schema (Pollscore, Silver Bulletin, Votehub, or similar):
    pollster_id     must match your pollster_id convention
    quality         float (any scale; normalized to [0.4, 1.0] per source)
    bias_{cand}     optional per-candidate bias columns (pp, positive=overestimate)
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
_OUTPUT_DEFAULT = _HERE / "pollster_ratings.json"

DEFAULT_QUALITY = 0.60  # fallback for pollsters with no external rating


def load_external_ratings(csv_paths: list[Path]) -> dict[str, dict]:
    """
    Load pollster quality scores from one or more external sources and average
    them together.

    Each CSV must have: pollster_id, quality
    Optional columns: bias_{CandidateName}  (pp, positive = overestimates that candidate)

    Quality scores from each source are normalized to [0.4, 1.0] before
    averaging so sources with different raw scales are treated equally.
    House effects are averaged across sources that provide them.
    """
    per_source: list[tuple[str, dict[str, dict]]] = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            warnings.warn(f"External ratings CSV not found: {csv_path} — skipping")
            continue

        df = pd.read_csv(str(csv_path))
        if "pollster_id" not in df.columns or "quality" not in df.columns:
            raise ValueError(f"{csv_path}: must have 'pollster_id' and 'quality' columns")

        scores = df["quality"].astype(float)
        lo, hi = scores.min(), scores.max()
        norm_scores = 0.4 + 0.6 * (scores - lo) / (hi - lo) if hi > lo else pd.Series([0.7] * len(scores))

        bias_cols = [c for c in df.columns if c.startswith("bias_")]
        source_ratings: dict[str, dict] = {}

        for i, row in df.iterrows():
            pid = str(row["pollster_id"])
            house_effect = {}
            for col in bias_cols:
                cand = col[len("bias_"):]
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    house_effect[cand] = round(float(val), 2)
            source_ratings[pid] = {
                "quality": float(norm_scores.iloc[i]),
                "house_effect_adjustment": house_effect,
            }

        per_source.append((csv_path.stem, source_ratings))

    if not per_source:
        return {}

    all_pids = set(pid for _, source in per_source for pid in source)
    ratings: dict[str, dict] = {}

    for pid in all_pids:
        entries = [(name, source[pid]) for name, source in per_source if pid in source]
        avg_quality = sum(e["quality"] for _, e in entries) / len(entries)

        all_he_keys = set(k for _, e in entries for k in e["house_effect_adjustment"])
        house_effect = {}
        for cand in all_he_keys:
            vals = [e["house_effect_adjustment"][cand] for _, e in entries if cand in e["house_effect_adjustment"]]
            house_effect[cand] = round(sum(vals) / len(vals), 2)

        ratings[pid] = {
            "quality": round(avg_quality, 4),
            "house_effect_adjustment": house_effect,
            "source": "external",
            "sources_used": [name for name, _ in entries],
        }

    return ratings


def main(
    external_csvs: list[Path],
    output_path: Path = _OUTPUT_DEFAULT,
) -> None:
    print(f"Loading external ratings from: {', '.join(str(p) for p in external_csvs)}")
    ratings = load_external_ratings(external_csvs)

    if not ratings:
        print("No ratings loaded — check that your CSV paths exist and have the right columns")
        return

    print(f"  {len(ratings)} pollsters rated")
    for pid, r in sorted(ratings.items(), key=lambda x: -x[1]["quality"]):
        sources = ", ".join(r["sources_used"])
        print(f"  {pid:22s}  quality={r['quality']:.3f}  [{sources}]")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)

    print(f"\nSaved {len(ratings)} pollster ratings to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pollster quality ratings from external sources")
    parser.add_argument(
        "--external", type=Path, nargs="+", required=True,
        help="One or more external ratings CSVs (pollster_id, quality, [bias_CandName...])",
    )
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT_DEFAULT,
        help="Output path for pollster_ratings.json",
    )
    args = parser.parse_args()
    main(external_csvs=args.external, output_path=args.output)
