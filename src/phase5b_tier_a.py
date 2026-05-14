"""
Tier A combined evaluator:
  1. Load Tier-A Foldseek hits (TM-align rescored, iterative, looser e-value).
  2. For each test query, vote top-k weighted by score; we sweep over weight
     columns (bits, tmscore, prob, lddt) and several k values.
  3. For queries with no Foldseek hit, fall back to a Random Forest trained
     on the 100-D structural feature matrix (Phase 2 output).
  4. Optionally apply a hit-quality threshold (drop hits below min_prob).
  5. Report weighted F1 for each (k, weight, threshold, fallback?) combo.

Writes:
  results/phase5b_tier_a_summary.csv
  results/phase5b_tier_a_summary.txt
  charts/phase5b_all_methods.png
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"
HITS_V2 = ROOT / "foldseek_workdir" / "hits_v2.tsv"

COLS = ["query", "target", "bits", "evalue", "prob", "alntmscore",
        "fident", "lddt"]


def load_hits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=COLS)
    # foldseek can return hits sorted oddly under iterative mode; re-sort
    df = df.sort_values(["query", "bits"], ascending=[True, False])
    return df


def train_fallback_rf() -> tuple[RandomForestClassifier, pd.DataFrame, dict]:
    """Train RF on the same 'ALL minus aa_ss' feature subset that won Phase 3."""
    cols_df = pd.read_csv(FEAT_DIR / "feature_columns.csv")
    keep_idx = [i for i, g in enumerate(cols_df["group"]) if g != "aa_ss"]
    data = np.load(FEAT_DIR / "feature_matrix.npz", allow_pickle=True)
    X_train, y_train = data["X_train"][:, keep_idx], data["y_train"]
    X_test, y_test, entries_test = (data["X_test"][:, keep_idx],
                                    data["y_test"], data["entries_test"])
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0,
                                class_weight="balanced")
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    test_df = pd.DataFrame({"Entry": entries_test, "true": y_test, "rf_pred": pred})
    rf_pred_map = dict(zip(test_df["Entry"], test_df["rf_pred"]))
    return rf, test_df, rf_pred_map


def vote(hits: pd.DataFrame, train_labels: dict, k: int,
         weight: str, min_prob: float = 0.0) -> dict[str, str]:
    """Return {query: pred} from top-k weighted vote, filtered by min_prob."""
    out = {}
    use = hits if min_prob <= 0 else hits[hits["prob"] >= min_prob]
    for q, grp in use.groupby("query", sort=False):
        grp = grp.head(k)
        score = defaultdict(float)
        for _, row in grp.iterrows():
            ec = train_labels.get(row["target"])
            if ec is None:
                continue
            w = 1.0 if weight == "uniform" else float(row[weight])
            score[ec] += w
        if score:
            out[q] = max(score.items(), key=lambda x: x[1])[0]
    return out


def evaluate(fs_pred: dict, fallback_pred: dict, test_labels: dict,
             tag: str) -> dict:
    """Compose: prefer Foldseek where available, else fallback; score weighted F1
    over all test queries (missing-from-both -> labeled '__none__', counts as wrong)."""
    preds, trues = [], []
    n_fs = n_fb = n_none = 0
    for q, t in test_labels.items():
        if q in fs_pred:
            preds.append(fs_pred[q]); n_fs += 1
        elif q in fallback_pred:
            preds.append(fallback_pred[q]); n_fb += 1
        else:
            preds.append("__none__"); n_none += 1
        trues.append(t)
    f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    return {"tag": tag, "weighted_f1": f1, "n_foldseek": n_fs,
            "n_fallback": n_fb, "n_none": n_none}


def main():
    if not HITS_V2.exists():
        print("hits_v2.tsv missing - skipping Tier-A eval")
        sys.exit(0)
    hits = load_hits(HITS_V2)
    print(f"Loaded {len(hits)} hits across {hits['query'].nunique()} queries")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_labels = dict(zip(train_df["Entry"], train_df["Label"]))
    test_labels = dict(zip(test_df["Entry"], test_df["Label"]))

    # Train the RF fallback on the proven feature subset
    print("\nTraining RF fallback (n_est=500, class_weight=balanced) ...")
    _rf, _td, rf_pred_map = train_fallback_rf()

    print("\n--- Foldseek-only (no fallback) ---")
    rows = []
    for k in [1, 3, 5]:
        for w in ["bits", "alntmscore", "prob", "lddt", "uniform"]:
            for thr in [0.0, 0.3, 0.5, 0.7]:
                pred = vote(hits, train_labels, k, w, min_prob=thr)
                rec = evaluate(pred, {}, test_labels,
                               f"fs_k{k}_{w}_p{thr}")
                rec.update({"k": k, "weight": w, "min_prob": thr,
                            "fallback": "none"})
                rows.append(rec)
    print("\n--- Foldseek + RF fallback ---")
    for k in [1, 3, 5]:
        for w in ["bits", "alntmscore"]:
            for thr in [0.0, 0.3, 0.5]:
                pred = vote(hits, train_labels, k, w, min_prob=thr)
                rec = evaluate(pred, rf_pred_map, test_labels,
                               f"fs_k{k}_{w}_p{thr}_+rf")
                rec.update({"k": k, "weight": w, "min_prob": thr,
                            "fallback": "rf"})
                rows.append(rec)

    summary = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    summary.to_csv(RESULTS_DIR / "phase5b_tier_a_summary.csv", index=False)

    print("\n=== Top 10 configurations ===")
    print(summary.head(10).to_string(index=False))
    top = summary.iloc[0]
    print(f"\nBEST: {top.to_dict()}")

    # comparison chart
    methods = {
        "Random (1/400)": 0.0025,
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "Foldseek-top1 (Phase 5a)": 0.2446,
        f"Foldseek tier-A best\n({top['tag']})": top["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#bbb", "#bbb", "#88a", "#4c72b0", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=colors)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.265, ls="--", color="#444", alpha=0.4)
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes Classification — Tier A results")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase5b_all_methods.png", dpi=140)
    plt.close()

    txt = ["Tier A combined results", "=" * 50, ""]
    txt.append(f"BEST: {top['tag']}  F1 = {top['weighted_f1']:.4f}")
    txt.append(f"  k={int(top['k'])}, weight={top['weight']}, "
               f"min_prob={top['min_prob']}, fallback={top['fallback']}")
    txt.append(f"  coverage: foldseek={top['n_foldseek']}, "
               f"rf_fallback={top['n_fallback']}, none={top['n_none']}")
    txt.append("")
    for name, val in methods.items():
        n = name.replace("\n", " ")
        txt.append(f"  {n:45s} F1 = {val:.4f}")
    (RESULTS_DIR / "phase5b_tier_a_summary.txt").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
