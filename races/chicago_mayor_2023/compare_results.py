"""
Compare model predictions against actual 2023 Round 1 results.

Prints three tables:
  1. Citywide vote shares (model vs actual)
  2. Ward-level leader accuracy and mean absolute error
  3. Precinct-level mean absolute error per candidate

Run from repo root: python -m races.chicago_mayor_2023.compare_results
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from races.chicago_mayor_2023.race_config import CONFIG

DB_PATH          = Path("/home/cole/databases/illinois_elections.db")
REGIONAL_PATH    = CONFIG.output_dir / "regional_vote_forecast.json"
PRECINCT_CSV     = CONFIG.output_dir / f"{CONFIG.race_id}_precinct_probabilities.csv"

# Map DB candidate names to config candidate names
CAND_MAP = {
    'Paul Vallas':          'Paul Vallas',
    'Brandon Johnson':      'Brandon Johnson',
    'Lori Lightfoot':       'Lori Lightfoot',
    'Jesus "Chuy" Garcia':  'Chuy García',
    'Willie Wilson':        'Willie Wilson',
    'Kam Buckner':          'Kam Buckner',
    'Sophia King':          'Sophia King',
    "Ja'Mal Green":         'Jamal Green',
}

MODELED = [c for c in CONFIG.candidates if c not in ('Kam Buckner', 'Sophia King', 'Jamal Green')]


def load_actual_precinct() -> pd.DataFrame:
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

    # Pivot to wide: one row per precinct
    wide = df.pivot_table(index="JoinField", columns="candidate", values="votes", aggfunc="sum", fill_value=0)
    wide["actual_total"] = wide.sum(axis=1)
    for c in [col for col in wide.columns if col != "actual_total"]:
        wide[f"actual_pct_{c}"] = wide[c] / wide["actual_total"]
    wide = wide.reset_index()
    return wide


def ward_from_joinfield(jf: str) -> str:
    suffix = jf.split(":", 1)[-1].strip().upper()
    parts = suffix.split()
    if len(parts) >= 2 and parts[0] == "WARD":
        try:
            return f"Ward {int(parts[1])}"
        except ValueError:
            pass
    return "Unknown"


def main():
    import json

    actual = load_actual_precinct()
    model  = pd.read_csv(PRECINCT_CSV)

    # Normalize JoinField for merge
    actual["_jf"] = actual["JoinField"].str.strip().str.upper()
    model["_jf"]  = model["joinfield"].str.strip().str.upper()
    merged = model.merge(actual, on="_jf", how="inner")
    merged["ward"] = merged["joinfield"].apply(ward_from_joinfield)

    candidates = list(CAND_MAP.values())
    final_cols = {c: f"final_est_{c}" for c in candidates if f"final_est_{c}" in merged.columns}

    print(f"Matched {len(merged):,} of 1,291 precincts\n")

    # ── 1. Citywide ───────────────────────────────────────────────────────
    print("=" * 65)
    print("CITYWIDE VOTE SHARES")
    print("=" * 65)
    total_actual = merged["actual_total"].sum()
    total_model_w = merged["turnout_weight"].sum()

    print(f"  {'Candidate':<24} {'Model':>7} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*24} {'-'*7} {'-'*8} {'-'*8}")
    citywide_rows = []
    for cand, col in final_cols.items():
        actual_col = f"actual_pct_{cand}"
        if actual_col not in merged.columns:
            continue
        model_pct  = (merged[col] * merged["turnout_weight"]).sum() / total_model_w * 100
        actual_pct = (merged[cand].sum() / total_actual * 100) if cand in merged.columns else 0
        print(f"  {cand:<24} {model_pct:>6.1f}%  {actual_pct:>7.1f}%  {model_pct - actual_pct:>+7.1f}pp")
        citywide_rows.append({"candidate": cand, "model_pct": round(model_pct, 2),
                               "actual_pct": round(actual_pct, 2), "error_pp": round(model_pct - actual_pct, 2)})

    # ── 2. Ward-level ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("WARD-LEVEL LEADER ACCURACY & MEAN ABSOLUTE ERROR")
    print("=" * 65)

    with open(REGIONAL_PATH) as f:
        regional = json.load(f)

    ward_rows = []
    for ward, data in regional.items():
        ward_actual = merged[merged["ward"] == ward]
        if ward_actual.empty:
            continue
        ward_total = ward_actual["actual_total"].sum()
        if ward_total == 0:
            continue

        model_shares  = data["vote_shares"]
        model_leader  = max(model_shares, key=model_shares.get)

        actual_shares = {
            cand: ward_actual[cand].sum() / ward_total
            for cand in candidates
            if cand in ward_actual.columns
        }
        actual_leader = max(actual_shares, key=actual_shares.get)

        # MAE across modeled candidates
        errors = []
        for cand in final_cols:
            if cand in actual_shares:
                errors.append(abs(model_shares.get(cand, 0) - actual_shares[cand]) * 100)

        ward_rows.append({
            "ward":           ward,
            "model_leader":   model_leader,
            "actual_leader":  actual_leader,
            "correct":        model_leader == actual_leader,
            "mae_pp":         np.mean(errors) if errors else np.nan,
        })

    ward_df = pd.DataFrame(ward_rows).sort_values("ward", key=lambda s: s.map(lambda x: int(x.split()[-1])))
    correct = ward_df["correct"].sum()
    total   = len(ward_df)
    mean_mae = ward_df["mae_pp"].mean()

    print(f"  Leader correct: {correct}/{total} wards ({correct/total:.0%})")
    print(f"  Mean per-ward MAE: {mean_mae:.1f}pp\n")
    print(f"  {'Ward':<10} {'Model Leader':<22} {'Actual Leader':<22} {'MAE':>6} {'OK':>4}")
    print(f"  {'-'*10} {'-'*22} {'-'*22} {'-'*6} {'-'*4}")
    for _, row in ward_df.iterrows():
        ok = "✓" if row["correct"] else "✗"
        print(f"  {row['ward']:<10} {row['model_leader']:<22} {row['actual_leader']:<22} {row['mae_pp']:>5.1f}pp {ok:>4}")

    # ── 3. Precinct-level ─────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("PRECINCT-LEVEL MEAN ABSOLUTE ERROR PER CANDIDATE")
    print("=" * 65)
    print(f"  {'Candidate':<24} {'MAE':>7} {'Bias':>8}")
    print(f"  {'-'*24} {'-'*7} {'-'*8}")
    all_errors = []
    mae_rows = []
    for cand, col in final_cols.items():
        actual_col = f"actual_pct_{cand}"
        if actual_col not in merged.columns:
            continue
        errs = (merged[col] - merged[actual_col]) * 100
        mae  = errs.abs().mean()
        bias = errs.mean()
        all_errors.append(mae)
        print(f"  {cand:<24} {mae:>6.1f}pp  {bias:>+7.1f}pp")
        mae_rows.append({"candidate": cand, "mae_pp": round(mae, 2), "bias_pp": round(bias, 2)})
    print(f"  {'OVERALL':<24} {np.mean(all_errors):>6.1f}pp")

    # ── 4. Precinct-level leader accuracy ─────────────────────────────────
    print()
    print("=" * 65)
    print("PRECINCT-LEVEL LEADER ACCURACY")
    print("=" * 65)

    model_leader_cols = {c: col for c, col in final_cols.items() if col in merged.columns}
    actual_leader_cols = {c: f"actual_pct_{c}" for c in model_leader_cols if f"actual_pct_{c}" in merged.columns}
    common = [c for c in model_leader_cols if c in actual_leader_cols]

    if common:
        model_shares  = merged[[model_leader_cols[c]  for c in common]].rename(columns={model_leader_cols[c]:  c for c in common})
        actual_shares = merged[[actual_leader_cols[c] for c in common]].rename(columns={actual_leader_cols[c]: c for c in common})

        model_leader  = model_shares.idxmax(axis=1)
        actual_leader = actual_shares.idxmax(axis=1)

        correct = (model_leader == actual_leader).sum()
        total   = len(merged)
        print(f"  Correct: {correct:,} / {total:,} precincts ({correct / total:.1%})")
        print()

        # Per-candidate: how often is each candidate the actual leader and did we call it?
        print(f"  {'Candidate':<24} {'Actual wins':>12} {'Called':>8} {'Recall':>8}")
        print(f"  {'-'*24} {'-'*12} {'-'*8} {'-'*8}")
        for cand in sorted(common, key=lambda c: -(actual_leader == c).sum()):
            actual_wins = (actual_leader == cand).sum()
            if actual_wins == 0:
                continue
            called = ((actual_leader == cand) & (model_leader == cand)).sum()
            print(f"  {cand:<24} {actual_wins:>12,} {called:>8,} {called / actual_wins:>8.1%}")

        # ── Build per-precinct error DataFrame for miss analysis ──────────────
        analysis = merged.copy()
        analysis["model_leader"]  = model_leader.values
        analysis["actual_leader"] = actual_leader.values
        analysis["correct"]       = (model_leader == actual_leader).values

        err_cols_map: list[tuple[str, str]] = []
        for cand in common:
            ecol = f"err_{cand}"
            analysis[ecol] = (analysis[final_cols[cand]] - analysis[f"actual_pct_{cand}"]) * 100
            err_cols_map.append((cand, ecol))

        ecol_names = [e for _, e in err_cols_map]
        analysis["precinct_mae"] = analysis[ecol_names].abs().mean(axis=1)

        act_pct_mat = np.sort(analysis[[f"actual_pct_{c}" for c in common]].values, axis=1)[:, ::-1]
        analysis["actual_margin"] = (act_pct_mat[:, 0] - act_pct_mat[:, 1]) * 100

        miss = analysis[~analysis["correct"]].copy()

        # ── 5. Confusion matrix ───────────────────────────────────────────────
        print()
        print("=" * 70)
        print(f"LEADER MISS CONFUSION MATRIX  ({len(miss):,} wrong / {len(analysis):,} precincts)")
        print("=" * 70)
        conf         = miss.groupby(["model_leader", "actual_leader"]).size().unstack(fill_value=0)
        pred_cands   = [c for c in MODELED if c in conf.index]
        actual_cands = [c for c in MODELED if c in conf.columns]
        w = 11
        print(f"  {'Predicted ↓  /  Actual →':<28}" + "".join(f"{c.split()[-1]:>{w}}" for c in actual_cands))
        print("  " + "─" * (28 + w * len(actual_cands)))
        for pred in pred_cands:
            row_str = f"  {pred:<28}"
            for act in actual_cands:
                v = conf.loc[pred, act] if (pred in conf.index and act in conf.columns) else 0
                row_str += f"{'—' if v == 0 else v:>{w}}"
            print(row_str)
        miss_counts = miss.groupby(["model_leader", "actual_leader"]).size().sort_values(ascending=False)
        if len(miss_counts):
            top_pred, top_act = miss_counts.index[0]
            print(f"\n  Most common: predicted {top_pred} → actual {top_act}  ({miss_counts.iloc[0]:,} precincts)")

        # ── 6. Miss analysis by actual margin ─────────────────────────────────
        print()
        print("=" * 70)
        print("MISS ANALYSIS BY ACTUAL WINNER MARGIN")
        print("  Toss-up misses are expected noise; large-margin misses signal model failure")
        print("=" * 70)
        bins   = [0, 5, 10, 20, 100]
        labels = ["<5pp  (toss-up)", "5–10pp", "10–20pp", ">20pp (clear)"]
        miss["_mbin"] = pd.cut(miss["actual_margin"], bins=bins, labels=labels)
        mbin_counts   = miss.groupby("_mbin", observed=True).size()
        miss_mae      = miss.groupby("_mbin", observed=True)["precinct_mae"].mean()
        print(f"  {'Margin bucket':<20} {'Misses':>8}  {'% of misses':>12}  {'Mean MAE':>9}")
        print(f"  {'─'*20} {'─'*8}  {'─'*12}  {'─'*9}")
        for label in labels:
            n   = mbin_counts.get(label, 0)
            mae = miss_mae.get(label, float("nan"))
            mae_str = f"{mae:.1f}pp" if not pd.isna(mae) else "—"
            print(f"  {label:<20} {n:>8,}  {n/len(miss):>11.1%}  {mae_str:>9}")

        # ── 7. Demographic profile of misses ──────────────────────────────────
        print()
        print("=" * 70)
        print("DEMOGRAPHIC PROFILE: CORRECT vs. MISSED PRECINCTS")
        print("=" * 70)
        demo_spec = [
            ("pct_black",      "% Black",             True),
            ("pct_hispanic",   "% Hispanic",           True),
            ("pct_white",      "% White",              True),
            ("score_pp",       "Prog score (pp)",      False),
            ("actual_total",   "Votes cast",           False),
            ("actual_margin",  "Actual winner margin", False),
        ]
        avail_demo = [(c, l, p) for c, l, p in demo_spec if c in analysis.columns]
        if avail_demo:
            corr_g  = analysis[analysis["correct"]]
            miss_g  = analysis[~analysis["correct"]]
            print(f"  {'Dimension':<26} {'Correct':>9} {'Missed':>9} {'Diff':>8}")
            print(f"  {'─'*26} {'─'*9} {'─'*9} {'─'*8}")
            for col, label, is_pct in avail_demo:
                c_m = corr_g[col].mean()
                m_m = miss_g[col].mean()
                diff = m_m - c_m
                if is_pct:
                    flag = " ◄" if abs(diff) > 0.03 else ""
                    print(f"  {label:<26} {c_m*100:>8.1f}%  {m_m*100:>8.1f}%  {diff*100:>+7.1f}pp{flag}")
                else:
                    print(f"  {label:<26} {c_m:>9.1f}  {m_m:>9.1f}  {diff:>+8.1f}")

        # ── 8. Candidate bias by demographic quartile ─────────────────────────
        print()
        print("=" * 70)
        print("CANDIDATE BIAS BY DEMOGRAPHIC QUARTILE  (model − actual, pp)")
        print("  Positive = model overestimates   Negative = model underestimates")
        print("  Gradient across quartiles → signal the model is missing")
        print("=" * 70)
        quartile_dims = [
            ("pct_black",    "% Black"),
            ("pct_hispanic", "% Hispanic"),
            ("score_pp",     "Progressive score"),
        ]
        q_labels = ["Q1 (lo)", "Q2", "Q3", "Q4 (hi)"]
        for dim_col, dim_label in quartile_dims:
            if dim_col not in analysis.columns:
                continue
            analysis["_q"] = pd.qcut(analysis[dim_col].rank(method="first"), 4, labels=q_labels)
            print(f"\n  {dim_label}:")
            print(f"  {'Candidate':<22}" + "".join(f"{q:>11}" for q in q_labels))
            print(f"  {'─'*22}" + "─" * (11 * 4))
            for cand, ecol in err_cols_map:
                biases = analysis.groupby("_q", observed=True)[ecol].mean()
                row_str = f"  {cand:<22}"
                for q in q_labels:
                    b = biases.get(q, float("nan"))
                    row_str += f"  {b:>+8.1f}pp" if not pd.isna(b) else f"  {'—':>9}"
                print(row_str)

        # ── 9. Worst individual misses ────────────────────────────────────────
        print()
        print("=" * 70)
        print("WORST INDIVIDUAL MISSES  (wrong winner only, top 15 by MAE)")
        print("=" * 70)
        short = [c.split()[-1][:7] for c, _ in err_cols_map]
        print(f"  {'Ward':<10} {'Predicted':<16} {'Actual':<16} {'Margin':>7} {'MAE':>6}  " +
              "  ".join(f"{s:>7}" for s in short))
        print("  " + "─" * (10 + 16 + 16 + 8 + 7 + 2 + 9 * len(short)))
        for _, row in miss.nlargest(15, "precinct_mae").iterrows():
            errs = "  ".join(
                f"{row[ecol]:>+7.1f}" if not pd.isna(row[ecol]) else f"{'—':>7}"
                for _, ecol in err_cols_map
            )
            print(f"  {row['ward']:<10} {row['model_leader']:<16} {row['actual_leader']:<16} "
                  f"{row['actual_margin']:>6.1f}pp {row['precinct_mae']:>5.1f}pp  {errs}")

        # ── 10. Wards with highest miss rate ──────────────────────────────────
        print()
        print("=" * 70)
        print("WARDS WITH HIGHEST MISS RATE  (≥5 precincts)")
        print("=" * 70)
        ward_stats = (
            analysis.groupby("ward")
            .agg(n_total=("correct", "count"),
                 n_miss=("correct", lambda x: (~x).sum()),
                 mean_mae=("precinct_mae", "mean"))
            .assign(miss_rate=lambda d: d["n_miss"] / d["n_total"])
        )
        top_miss = (ward_stats[ward_stats["n_total"] >= 5]
                    .sort_values("miss_rate", ascending=False)
                    .head(10))
        print(f"  {'Ward':<12} {'Precincts':>10} {'Misses':>8} {'Miss Rate':>10} {'Mean MAE':>9}")
        print(f"  {'─'*12} {'─'*10} {'─'*8} {'─'*10} {'─'*9}")
        for ward, row in top_miss.iterrows():
            print(f"  {ward:<12} {row['n_total']:>10,} {row['n_miss']:>8,} "
                  f"{row['miss_rate']:>9.1%}  {row['mean_mae']:>8.1f}pp")
        print()

        # ── 11. Demographic × ideology 2D interaction ────────────────────────
        print()
        print("=" * 70)
        print("DEMOGRAPHIC × IDEOLOGY INTERACTION  (miss rate | candidate bias pp)")
        print("  Median splits: 'Hi' = above median for that dimension")
        print("  Positive bias = model overestimates   Negative = underestimates")
        print("=" * 70)

        key_cands = [c for c in ["Brandon Johnson", "Paul Vallas", "Chuy García", "Willie Wilson"]
                     if c in common]
        interact_dims = [("pct_black", "Black"), ("pct_hispanic", "Hispanic")]

        for demo_col, demo_word in interact_dims:
            if demo_col not in analysis.columns or "score_pp" not in analysis.columns:
                continue
            demo_med = analysis[demo_col].median()
            prog_med = analysis["score_pp"].median()
            demo_hi_mask = analysis[demo_col] >= demo_med
            prog_hi_mask = analysis["score_pp"] >= prog_med

            print(f"\n  % {demo_word} × Progressive Score")
            print(f"  (thresholds: ≥{demo_med*100:.1f}% {demo_word}  |  ≥{prog_med:.1f}pp prog score)")
            hdr = f"  {'Cell':<28} {'n':>5} {'Miss%':>7}"
            for c in key_cands:
                hdr += f"  {c.split()[-1][:8]:>9}"
            print(hdr)
            print("  " + "─" * (28 + 5 + 7 + 4 + 11 * len(key_cands)))

            cells = [
                (~demo_hi_mask & ~prog_hi_mask, f"Lo {demo_word}, Conservative"),
                (~demo_hi_mask &  prog_hi_mask, f"Lo {demo_word}, Progressive"),
                ( demo_hi_mask & ~prog_hi_mask, f"Hi {demo_word}, Conservative"),
                ( demo_hi_mask &  prog_hi_mask, f"Hi {demo_word}, Progressive"),
            ]
            # Reference row: all precincts
            for mask, label in [(pd.Series(True, index=analysis.index), "ALL precincts")] + cells:
                cell = analysis[mask]
                if cell.empty:
                    continue
                miss_pct = (~cell["correct"]).mean() * 100
                row = f"  {label:<28} {len(cell):>5} {miss_pct:>6.1f}%"
                for c in key_cands:
                    ecol = f"err_{c}"
                    if ecol in cell.columns:
                        row += f"  {cell[ecol].mean():>+8.1f}pp"
                    else:
                        row += f"  {'—':>9}"
                print(row)
        print()

        # ── Save CSVs ─────────────────────────────────────────────────────────
        out_dir = CONFIG.output_dir
        pd.DataFrame(citywide_rows).to_csv(out_dir / "compare_citywide.csv", index=False)
        ward_df.to_csv(out_dir / "compare_wards.csv", index=False)
        pd.DataFrame(mae_rows).to_csv(out_dir / "compare_candidate_mae.csv", index=False)
        save_cols = (
            ["joinfield", "ward", "model_leader", "actual_leader", "correct",
             "precinct_mae", "actual_margin"]
            + ecol_names
            + [c for c in ["pct_black", "pct_hispanic", "pct_white", "pct_asian", "score_pp"]
               if c in analysis.columns]
        )
        analysis[save_cols].to_csv(out_dir / "compare_precinct_detail.csv", index=False)
        print(f"CSVs saved → {out_dir}/compare_*.csv")


if __name__ == "__main__":
    main()
