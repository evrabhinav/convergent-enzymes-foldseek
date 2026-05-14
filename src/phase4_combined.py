"""
Phase 4: Combine sequence (424) + structural (100) features and re-evaluate.

Loads both feature matrices, aligns rows by Entry ID, concatenates columns,
runs the same SVM/RF/LR/kNN suite, and produces a comparison chart against
the three pure-feature settings (structural alone, sequence alone, combined).

Outputs:
  results/phase4_combined_results.csv
  charts/phase4_comparison.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from phase3_train_eval import evaluate, filter_test_to_known_labels, BASELINES

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)


def align_and_concat(matrix_path_a: Path, matrix_path_b: Path):
    """Load two .npz matrices and inner-join their rows on Entry ID per split.

    Returns (X_train, y_train, X_test, y_test, n_features_a).
    The same Label is required between the two matrices; if mismatched the
    function will raise (sanity check).
    """
    a = np.load(matrix_path_a, allow_pickle=True)
    b = np.load(matrix_path_b, allow_pickle=True)

    def _join(split: str):
        Xa, ya, ea = a[f"X_{split}"], a[f"y_{split}"], a[f"entries_{split}"]
        Xb, yb, eb = b[f"X_{split}"], b[f"y_{split}"], b[f"entries_{split}"]
        a_idx = {str(e): i for i, e in enumerate(ea)}
        b_idx = {str(e): i for i, e in enumerate(eb)}
        common = [e for e in a_idx if e in b_idx]
        ia = [a_idx[e] for e in common]
        ib = [b_idx[e] for e in common]
        ya2, yb2 = ya[ia], yb[ib]
        if not np.array_equal(ya2, yb2):
            raise RuntimeError(f"label mismatch on {split} after join")
        X = np.hstack([Xa[ia], Xb[ib]])
        return X, ya2, np.array(common), Xa.shape[1]

    Xtr, ytr, _, na = _join("train")
    Xte, yte, _, _ = _join("test")
    return Xtr, ytr, Xte, yte, na


def main():
    struct_path = FEATURES_DIR / "feature_matrix.npz"
    seq_path = FEATURES_DIR / "sequence_feature_matrix.npz"
    if not struct_path.exists() or not seq_path.exists():
        raise SystemExit("Run phase2_features.py and sequence_features.py first.")

    Xtr, ytr, Xte, yte, n_struct = align_and_concat(struct_path, seq_path)
    print(f"Combined: train {Xtr.shape}, test {Xte.shape}  (struct cols 0..{n_struct-1}, seq cols {n_struct}..)")

    Xte_f, yte_f, _, dropped = filter_test_to_known_labels(ytr, Xte, yte, np.arange(len(yte)))
    print(f"Test rows kept: {len(yte_f)}  (dropped {dropped})")

    # 1) Combined
    print("\n--- combined (structural + sequence) ---")
    combined = evaluate(Xtr, ytr, Xte_f, yte_f)
    for m, f in combined.items():
        print(f"  {m:20s} F1={f:.4f}")

    # 2) Structural only
    print("\n--- structural only ---")
    s_only = evaluate(Xtr[:, :n_struct], ytr, Xte_f[:, :n_struct], yte_f)
    for m, f in s_only.items():
        print(f"  {m:20s} F1={f:.4f}")

    # 3) Sequence only
    print("\n--- sequence only ---")
    q_only = evaluate(Xtr[:, n_struct:], ytr, Xte_f[:, n_struct:], yte_f)
    for m, f in q_only.items():
        print(f"  {m:20s} F1={f:.4f}")

    rows = []
    for tag, res in [("combined", combined), ("structural", s_only), ("sequence", q_only)]:
        for m, f in res.items():
            rows.append({"feature_set": tag, "model": m, "weighted_f1": f})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase4_combined_results.csv", index=False)

    # Comparison chart
    pivot = df.pivot(index="model", columns="feature_set", values="weighted_f1")
    pivot = pivot[["sequence", "structural", "combined"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot.index))
    w = 0.25
    for i, col in enumerate(pivot.columns):
        bars = ax.bar(x + (i - 1) * w, pivot[col].values, w, label=col)
        for b, v in zip(bars, pivot[col].values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
    for name, val in BASELINES.items():
        ax.axhline(val, ls="--", color="gray", alpha=0.6)
        ax.text(len(pivot.index) - 1, val, f" {name} = {val:.3f}",
                fontsize=8, va="center", ha="right", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes: sequence vs structural vs combined")
    ax.legend(loc="upper left")
    ymax = max(pivot.values[~np.isnan(pivot.values)].max() * 1.2, 0.3)
    ax.set_ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase4_comparison.png", dpi=140)
    plt.close()
    print(f"\nSaved {CHARTS_DIR / 'phase4_comparison.png'}")


if __name__ == "__main__":
    main()
