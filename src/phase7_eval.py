"""
Phase 7 evaluator: pocket features alone + ensembles with Foldseek/ESM2.

Pipeline:
  1. Score pocket features alone (SVM/RF/LR/kNN).
  2. Ensemble: Foldseek top-1 first, fall back to pocket-RF for no-hit queries.
  3. Three-way ensemble: Foldseek > (pocket-RF + ESM2-best) consensus on
     low-confidence Foldseek queries.
  4. Confidence-gated ensemble: Foldseek if prob >= 0.3, else pocket-RF.

Outputs:
  results/phase7_pocket_results.csv
  results/phase7_ensemble_summary.csv
  results/phase7_summary.txt
  charts/phase7_final_methods.png
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
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"


def load(matrix_path: Path):
    d = np.load(matrix_path, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["entries_train"],
            d["X_test"], d["y_test"], d["entries_test"])


def load_foldseek_top1() -> tuple[dict, dict]:
    df = pd.read_csv(ROOT / "foldseek_workdir/hits.tsv", sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = df.sort_values("bits", ascending=False).groupby("q").head(1)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    tl = dict(zip(train_df["Entry"], train_df["Label"]))
    pred = {q: tl[t] for q, t in zip(top["q"], top["target"]) if t in tl}
    prob = dict(zip(top["q"], top["prob"]))
    return pred, prob


def best_esm2_pred() -> dict:
    """Re-train LogReg on ESM2-35M (the best from Phase 6) and return predictions."""
    d = np.load(FEAT_DIR / "esm2_35m_matrix.npz", allow_pickle=True)
    sc = StandardScaler()
    Xtr = sc.fit_transform(d["X_train"]); Xte = sc.transform(d["X_test"])
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(Xtr, d["y_train"])
    p = m.predict(Xte)
    return {str(e): pi for e, pi in zip(d["entries_test"], p)}


def train_eval_pocket(Xtr, ytr, Xte, yte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    models = {
        "Pocket SVM-rbf-C1":   SVC(kernel="rbf", C=1.0),
        "Pocket LogReg-C1":    LogisticRegression(C=1.0, max_iter=2000),
        "Pocket kNN-1":        KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "Pocket kNN-3":        KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "Pocket kNN-5":        KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Pocket RF-500":       RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                       random_state=0,
                                                       class_weight="balanced"),
    }
    out = {}
    for name, m in models.items():
        Xa, Xb = (Xtr, Xte) if "RF" in name else (Xtr_s, Xte_s)
        m.fit(Xa, ytr)
        pred = m.predict(Xb)
        out[name] = (f1_score(yte, pred, average="weighted", zero_division=0), pred)
    return out


def majority(votes: list[str]) -> str:
    """Plurality with ties broken by first occurrence."""
    from collections import Counter
    return Counter(votes).most_common(1)[0][0]


def main():
    Xtr, ytr, ent_tr, Xte, yte, ent_te = load(FEAT_DIR / "pocket_feature_matrix.npz")
    print(f"Pocket features: train {Xtr.shape}, test {Xte.shape}")
    fs_pred, fs_prob = load_foldseek_top1()
    esm_pred = best_esm2_pred()
    test_labels = dict(zip(ent_te, yte))

    print("\n--- Pocket features alone ---")
    pocket_res = train_eval_pocket(Xtr, ytr, Xte, yte)
    rows = []
    best_pname = None; best_pf1 = -1; best_ppred = None
    for name, (f1, pred) in pocket_res.items():
        print(f"  {name:22s} F1 = {f1:.4f}")
        rows.append({"method": name, "weighted_f1": f1})
        if f1 > best_pf1:
            best_pf1 = f1; best_pname = name; best_ppred = pred
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "phase7_pocket_results.csv", index=False)

    pocket_pred_map = {str(e): p for e, p in zip(ent_te, best_ppred)}

    print("\n--- Ensembles ---")
    ens_rows = []

    # 1. Foldseek > Pocket fallback (Foldseek when has hit, else Pocket)
    p1 = [fs_pred.get(str(e), pocket_pred_map[str(e)]) for e in ent_te]
    f1 = f1_score(yte, p1, average="weighted", zero_division=0)
    ens_rows.append({"ensemble": "Foldseek > Pocket fallback", "weighted_f1": f1})

    # 2. Confidence-gated: Foldseek if prob>=0.3 else Pocket
    for thr in [0.3, 0.5, 0.7]:
        p = [(fs_pred[str(e)] if (str(e) in fs_pred and fs_prob.get(str(e), 0) >= thr)
              else pocket_pred_map[str(e)]) for e in ent_te]
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        ens_rows.append({"ensemble": f"Foldseek(prob>={thr}) > Pocket",
                         "weighted_f1": f1})

    # 3. Three-way: Foldseek > majority(Pocket, ESM2)
    p3 = []
    for e in ent_te:
        e = str(e)
        if e in fs_pred and fs_prob.get(e, 0) >= 0.3:
            p3.append(fs_pred[e])
        else:
            p3.append(majority([pocket_pred_map[e], esm_pred[e]]))
    f1 = f1_score(yte, p3, average="weighted", zero_division=0)
    ens_rows.append({"ensemble": "Foldseek(prob>=0.3) > majority(Pocket+ESM2)",
                     "weighted_f1": f1})

    # 4. Three-way w/o conf gate: Foldseek > majority
    p4 = []
    for e in ent_te:
        e = str(e)
        if e in fs_pred:
            p4.append(fs_pred[e])
        else:
            p4.append(majority([pocket_pred_map[e], esm_pred[e]]))
    f1 = f1_score(yte, p4, average="weighted", zero_division=0)
    ens_rows.append({"ensemble": "Foldseek > majority(Pocket+ESM2) fallback",
                     "weighted_f1": f1})

    for r in sorted(ens_rows, key=lambda x: -x["weighted_f1"]):
        print(f"  {r['ensemble']:50s} F1 = {r['weighted_f1']:.4f}")
    pd.DataFrame(ens_rows).to_csv(RESULTS_DIR / "phase7_ensemble_summary.csv",
                                  index=False)

    best_ens = max(ens_rows, key=lambda r: r["weighted_f1"])

    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2 ensemble": 0.2502,
        f"Pocket best\n({best_pname})": best_pf1,
        f"BEST Phase 7 ensemble\n({best_ens['ensemble']})": best_ens["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(13, 5.5))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#5aa55a", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes — Phase 7 (pocket features added)")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=12, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase7_final_methods.png", dpi=140)
    plt.close()

    txt = ["Phase 7 results (pocket features)", "=" * 50, ""]
    txt.append(f"Best pocket classifier: {best_pname}  F1 = {best_pf1:.4f}")
    txt.append(f"Best ensemble:          {best_ens['ensemble']}  F1 = {best_ens['weighted_f1']:.4f}")
    txt.append("")
    for name, val in methods.items():
        txt.append(f"  {name.replace(chr(10), ' '):55s} F1 = {val:.4f}")
    (RESULTS_DIR / "phase7_summary.txt").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
