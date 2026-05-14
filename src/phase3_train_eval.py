"""
Phase 3: Train classifiers on structural features and evaluate on test set.

Trains SVM (RBF, C=1.0), Random Forest (100 trees), Logistic Regression
(C=1.0, max_iter=2000), and kNN (k=1,3,5). Also runs feature ablation by
feature group. Outputs:

  results/phase3_results.csv          all (model, feature_group) F1 scores
  results/phase3_summary.txt          human-readable summary + baselines
  charts/phase3_overall_bar.png       all features: per-model F1 bars
  charts/phase3_ablation_heatmap.png  group x model F1 heatmap

Baselines printed for comparison:
  sequence_424   = 0.016   (user's previous baseline)
  esm2_3b        = 0.265   (DGEB paper)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

BASELINES = {"sequence_424 (yours)": 0.016, "ESM2-3B (DGEB)": 0.265}


def make_models() -> dict:
    return {
        "SVM-rbf-C1": SVC(kernel="rbf", C=1.0),
        "RandomForest-100": RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
        "LogReg-C1": LogisticRegression(C=1.0, max_iter=2000, n_jobs=-1),
        "kNN-1": KNeighborsClassifier(n_neighbors=1),
        "kNN-3": KNeighborsClassifier(n_neighbors=3),
        "kNN-5": KNeighborsClassifier(n_neighbors=5),
    }


def evaluate(X_train, y_train, X_test, y_test, scale: bool = True) -> dict[str, float]:
    """Train & evaluate every model. Returns {model_name: weighted_f1}."""
    if scale:
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test
    out = {}
    for name, model in make_models().items():
        # tree models don't need scaling
        Xtr = X_train if "Forest" in name else X_train_s
        Xte = X_test if "Forest" in name else X_test_s
        try:
            model.fit(Xtr, y_train)
            pred = model.predict(Xte)
            out[name] = f1_score(y_test, pred, average="weighted", zero_division=0)
        except Exception as e:
            print(f"[{name}] failed: {e}")
            out[name] = float("nan")
    return out


def filter_test_to_known_labels(y_train, X_test, y_test, entries_test):
    """If a test EC class never appears in train, drop those rows. Reports drop count."""
    known = set(y_train.tolist())
    mask = np.array([y in known for y in y_test])
    dropped = int((~mask).sum())
    return X_test[mask], y_test[mask], np.array(entries_test)[mask], dropped


def main(matrix_path: Path = None, tag: str = "structural"):
    matrix_path = matrix_path or (FEATURES_DIR / "feature_matrix.npz")
    data = np.load(matrix_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test, entries_test = data["X_test"], data["y_test"], data["entries_test"]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train classes: {len(set(y_train))}, Test classes: {len(set(y_test))}")

    X_test, y_test, entries_test, dropped = filter_test_to_known_labels(
        y_train, X_test, y_test, entries_test)
    print(f"  test rows kept: {len(y_test)} (dropped {dropped} unseen-class rows)")

    cols_df = pd.read_csv(FEATURES_DIR / "feature_columns.csv")
    groups = cols_df["group"].tolist()
    group_names = sorted(set(groups))
    print(f"Feature groups: {group_names}")

    rows = []
    # Overall (all features)
    print("\n--- all features ---")
    res = evaluate(X_train, y_train, X_test, y_test)
    for m, f in res.items():
        rows.append({"feature_group": "ALL", "model": m, "weighted_f1": f})
        print(f"  {m:20s} F1={f:.4f}")

    # Ablation: each group alone
    for g in group_names:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        Xtr, Xte = X_train[:, idx], X_test[:, idx]
        print(f"\n--- group: {g}  ({len(idx)} features) ---")
        res = evaluate(Xtr, y_train, Xte, y_test)
        for m, f in res.items():
            rows.append({"feature_group": g, "model": m, "weighted_f1": f})
            print(f"  {m:20s} F1={f:.4f}")

    # Leave-one-group-out (drop each group, train on remainder)
    for g in group_names:
        idx = [i for i, gg in enumerate(groups) if gg != g]
        Xtr, Xte = X_train[:, idx], X_test[:, idx]
        print(f"\n--- ALL minus {g} ({len(idx)} features) ---")
        res = evaluate(Xtr, y_train, Xte, y_test)
        for m, f in res.items():
            rows.append({"feature_group": f"ALL_minus_{g}", "model": m, "weighted_f1": f})
            print(f"  {m:20s} F1={f:.4f}")

    df = pd.DataFrame(rows)
    df["tag"] = tag
    out_csv = RESULTS_DIR / f"phase3_results_{tag}.csv"
    df.to_csv(out_csv, index=False)

    # === Charts ===
    # 1. Bar chart: all-features F1 per model + baselines
    overall = df[df["feature_group"] == "ALL"].set_index("model")["weighted_f1"]
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(overall.index) + list(BASELINES.keys())
    vals = list(overall.values) + list(BASELINES.values())
    colors = ["#4c72b0"] * len(overall) + ["#999999", "#cc4444"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("weighted F1")
    ax.set_title(f"Convergent Enzymes Classification — {tag} features (all)")
    ax.set_ylim(0, max(max(vals) * 1.2, 0.3))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"phase3_overall_bar_{tag}.png", dpi=140)
    plt.close()

    # 2. Heatmap: group x model
    pivot = (df[df["feature_group"].isin(["ALL"] + group_names)]
             .pivot(index="feature_group", columns="model", values="weighted_f1"))
    pivot = pivot.reindex(["ALL"] + group_names)
    fig, ax = plt.subplots(figsize=(10, 1 + 0.5 * len(pivot)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < pivot.values[~np.isnan(pivot.values)].mean() else "black",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label="weighted F1")
    ax.set_title(f"Ablation: feature group x model ({tag})")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"phase3_ablation_heatmap_{tag}.png", dpi=140)
    plt.close()

    # === Summary text ===
    best_model = overall.idxmax()
    summary = [
        f"Convergent Enzymes Classification - {tag} features",
        f"==================================================",
        f"Train: {X_train.shape}  Test: {X_test.shape}  (dropped {dropped} unseen-class rows)",
        f"Best model on all features: {best_model} = {overall[best_model]:.4f}",
        f"",
        f"Baselines:",
    ]
    for name, val in BASELINES.items():
        summary.append(f"  {name:25s} F1 = {val:.4f}")
    summary.append("")
    summary.append("All-features per-model:")
    for m, v in overall.items():
        summary.append(f"  {m:25s} F1 = {v:.4f}")
    summary.append("")
    summary.append("Best per group (single-group ablation):")
    for g in group_names:
        sub = df[df["feature_group"] == g].set_index("model")["weighted_f1"]
        m = sub.idxmax()
        summary.append(f"  {g:10s} best = {m:25s} F1 = {sub[m]:.4f}")
    txt = "\n".join(summary)
    (RESULTS_DIR / f"phase3_summary_{tag}.txt").write_text(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
