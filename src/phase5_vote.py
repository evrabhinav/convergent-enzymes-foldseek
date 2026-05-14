"""
Parse Foldseek hits and evaluate as a structural kNN classifier.

For each test query, look at the top-k hits in the training set and vote on
the EC label, weighted by Foldseek bit-score. Try several voting strategies
and several k values.

Outputs:
  results/phase5_foldseek_predictions_k{K}_{vote}.csv
  results/phase5_foldseek_summary.csv
  results/phase5_summary.txt
  charts/phase5_all_methods_bar.png
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

HITS_PATH = ROOT / "foldseek_workdir" / "hits.tsv"  # written under WSL but accessible via WSL path
# Actual location: /mnt/c/...path... -> from Windows: ROOT/foldseek_workdir but we put it in /root inside WSL
# We will copy it out first
WSL_HITS = "/root/fs_work/hits.tsv"


def load_hits() -> pd.DataFrame:
    """Read the Foldseek hits TSV. We copied it from /root/fs_work/hits.tsv
    to ROOT/foldseek_workdir/hits.tsv via a helper bash call."""
    df = pd.read_csv(
        HITS_PATH, sep="\t",
        names=["query", "target", "bits", "evalue", "prob", "alntmscore",
               "fident", "lddt"],
    )
    return df


def vote(hits: pd.DataFrame, train_labels: dict, k: int,
         weight: str = "bits") -> pd.DataFrame:
    """Top-k weighted vote. `weight` in {'bits','prob','lddt','alntmscore','uniform'}."""
    out = []
    for q, grp in hits.groupby("query", sort=False):
        # Foldseek returns hits sorted best->worst, but be explicit
        grp = grp.sort_values("bits", ascending=False).head(k)
        score = defaultdict(float)
        for _, row in grp.iterrows():
            ec = train_labels.get(row["target"])
            if ec is None:
                continue
            w = 1.0 if weight == "uniform" else float(row[weight])
            score[ec] += w
        if score:
            best = max(score.items(), key=lambda x: x[1])
            out.append({"query": q, "pred": best[0],
                        "n_hits": len(grp), "top_score": best[1]})
        else:
            out.append({"query": q, "pred": None, "n_hits": 0, "top_score": 0.0})
    return pd.DataFrame(out)


def main() -> None:
    hits = load_hits()
    print(f"Loaded {len(hits)} hits across {hits['query'].nunique()} queries")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_labels = dict(zip(train_df["Entry"], train_df["Label"]))
    test_labels = dict(zip(test_df["Entry"], test_df["Label"]))

    rows = []
    for k in [1, 3, 5, 10, 20]:
        for w in ["bits", "prob", "lddt", "alntmscore", "uniform"]:
            preds = vote(hits, train_labels, k=k, weight=w)
            preds["true"] = preds["query"].map(test_labels)
            kept = preds.dropna(subset=["pred", "true"])
            if len(kept) == 0:
                continue
            # Score over ALL test queries (treat missing as wrong)
            full = preds.copy()
            full["pred"] = full["pred"].fillna("__none__")
            full["true"] = full["query"].map(test_labels)
            full = full.dropna(subset=["true"])
            f1 = f1_score(full["true"], full["pred"],
                          average="weighted", zero_division=0)
            cov = preds["pred"].notna().sum() / len(test_df)
            n_have_query = preds["query"].nunique()
            print(f"  k={k:2d} weight={w:11s}  F1={f1:.4f}  "
                  f"hit_coverage={cov:.3f}  n_queries={n_have_query}")
            rows.append({"k": k, "weight": w, "weighted_f1": f1,
                         "hit_coverage": cov, "n_queries": n_have_query})
            preds.to_csv(RESULTS_DIR / f"phase5_foldseek_pred_k{k}_{w}.csv",
                         index=False)

    summary = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    summary.to_csv(RESULTS_DIR / "phase5_foldseek_summary.csv", index=False)
    print(f"\nBest: {summary.iloc[0].to_dict()}")

    # -------- chart: Foldseek vs all baselines --------
    methods = {
        "Random (1/400)": 0.0025,
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        f"Foldseek-top{summary.iloc[0]['k']:.0f}-{summary.iloc[0]['weight']}":
            summary.iloc[0]["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#999", "#999", "#4c72b0", "#dd5555", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=colors)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes Classification — methods compared")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase5_all_methods_bar.png", dpi=140)
    plt.close()

    # -------- summary text --------
    txt = ["Convergent Enzymes Classification — final summary",
           "=" * 50, ""]
    for name, val in methods.items():
        txt.append(f"  {name:35s} F1 = {val:.4f}")
    txt.append("")
    txt.append("Foldseek top results (per k, weight):")
    for _, r in summary.head(10).iterrows():
        txt.append(f"  k={int(r['k']):2d} weight={r['weight']:11s} "
                   f"F1={r['weighted_f1']:.4f}  cov={r['hit_coverage']:.3f}")
    (RESULTS_DIR / "phase5_summary.txt").write_text("\n".join(txt))
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
