"""
Layered Tier-A classifier:
  Priority 1: Phase 5a Foldseek top-1 (default scoring, more accurate)
  Priority 2: Tier-A Foldseek top-1 (TM-align/iter, broader recall)
  Priority 3: Random Forest trained on the proven structural-feature subset

Each priority is filled only for queries the prior priorities did not predict.

Also runs an ablation: each priority alone, and the layered combo.
"""
from __future__ import annotations

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


def top1_preds(hits_path: Path, train_labels: dict) -> dict[str, str]:
    df = pd.read_csv(hits_path, sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = df.sort_values("bits", ascending=False).groupby("q").head(1)
    return {q: train_labels[t] for q, t in zip(top["q"], top["target"])
            if t in train_labels}


def rf_preds() -> dict[str, str]:
    """RF on the 'all minus aa_ss' subset that won Phase 3."""
    cols_df = pd.read_csv(FEAT_DIR / "feature_columns.csv")
    keep = [i for i, g in enumerate(cols_df["group"]) if g != "aa_ss"]
    d = np.load(FEAT_DIR / "feature_matrix.npz", allow_pickle=True)
    X_tr, y_tr = d["X_train"][:, keep], d["y_train"]
    X_te, ent = d["X_test"][:, keep], d["entries_test"]
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0,
                                class_weight="balanced")
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    return {str(e): p for e, p in zip(ent, pred)}


def score(preds: dict[str, str], test_labels: dict) -> tuple[float, int]:
    """Weighted F1 over all 400 test queries; missing -> '__none__'."""
    trues, hats = [], []
    n_pred = 0
    for q, t in test_labels.items():
        trues.append(t)
        if q in preds:
            hats.append(preds[q]); n_pred += 1
        else:
            hats.append("__none__")
    return f1_score(trues, hats, average="weighted", zero_division=0), n_pred


def layer(*sources) -> dict[str, str]:
    """Earlier sources take precedence."""
    out = {}
    for src in sources:
        for q, p in src.items():
            if q not in out:
                out[q] = p
    return out


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    tl = dict(zip(train_df["Entry"], train_df["Label"]))
    test_labels = dict(zip(test_df["Entry"], test_df["Label"]))

    v1 = top1_preds(ROOT / "foldseek_workdir/hits.tsv", tl)        # Phase 5a default
    v2 = top1_preds(ROOT / "foldseek_workdir/hits_v2.tsv", tl)     # Tier A
    print("RF fallback training ...")
    rf = rf_preds()

    rows = []
    for tag, src in [
        ("foldseek_v1_only",            v1),
        ("foldseek_v2_only",            v2),
        ("rf_only",                     rf),
        ("v1 + v2_fallback",            layer(v1, v2)),
        ("v1 + rf_fallback",            layer(v1, rf)),
        ("v1 + v2 + rf_fallback",       layer(v1, v2, rf)),
        ("v2 + rf_fallback",            layer(v2, rf)),
    ]:
        f1, n = score(src, test_labels)
        rows.append({"method": tag, "weighted_f1": f1, "n_predicted": n})
        print(f"  {tag:30s} F1={f1:.4f}  predicted {n}/400")

    summary = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    summary.to_csv(RESULTS_DIR / "phase5c_layered_summary.csv", index=False)
    best = summary.iloc[0]
    print(f"\nBEST: {best['method']}  F1={best['weighted_f1']:.4f}")

    # chart
    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "Foldseek v1 (Phase 5a)": 0.2446,
        f"BEST layered\n({best['method']})": best["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = ["#bbb", "#88a", "#4c72b0", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.5,
               label="ESM2-3B reference")
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes — Tier A layered classifier")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase5c_layered.png", dpi=140)
    plt.close()

    (RESULTS_DIR / "phase5c_layered.txt").write_text(
        "Tier A layered classifier\n" +
        "=" * 40 + "\n" +
        summary.to_string(index=False) + "\n\n" +
        f"BEST: {best['method']}  F1 = {best['weighted_f1']:.4f}\n"
    )


if __name__ == "__main__":
    main()
