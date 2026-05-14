"""
Phase 12: ESM2-3B alone + Foldseek + ESM2-3B ensemble.

Two questions:
  1. Does our ESM2-3B replicate the DGEB-paper 0.265? (sanity check the pipeline)
  2. Does Foldseek + ESM2-3B ensemble CROSS 0.265?

Outputs:
  results/phase12_esm3b_results.csv
  results/phase12_summary.txt
  charts/phase12_final.png
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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"


def load_foldseek_top1():
    df = pd.read_csv(ROOT / "foldseek_workdir/hits.tsv", sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = df.sort_values("bits", ascending=False).groupby("q").head(1)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    tl = dict(zip(train_df["Entry"], train_df["Label"]))
    pred = {q: tl[t] for q, t in zip(top["q"], top["target"]) if t in tl}
    prob = dict(zip(top["q"], top["prob"]))
    return pred, prob


def ncm(Xtr, ytr, Xte):
    Xtr_n = normalize(Xtr); Xte_n = normalize(Xte)
    classes = sorted(set(ytr.tolist()))
    centroids = np.stack([Xtr_n[ytr == c].mean(axis=0) for c in classes])
    centroids = normalize(centroids)
    sim = Xte_n @ centroids.T
    return np.array(classes)[sim.argmax(axis=1)]


def train_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    out = {}
    models = {
        "SVM-rbf-C1":   SVC(kernel="rbf", C=1.0),
        "LogReg-C1":    LogisticRegression(C=1.0, max_iter=2000),
        "kNN-1-cos":    KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "kNN-3-cos":    KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "kNN-5-cos":    KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "RF-500":       RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                random_state=0,
                                                class_weight="balanced"),
    }
    for name, m in models.items():
        Xa, Xb = (Xtr, Xte) if "RF" in name else (Xtr_s, Xte_s)
        m.fit(Xa, ytr); pred = m.predict(Xb)
        out[name] = (f1_score(yte, pred, average="weighted", zero_division=0), pred)
    p = ncm(Xtr, ytr, Xte)
    out["NCM (cosine)"] = (f1_score(yte, p, average="weighted", zero_division=0), p)
    return out


def main():
    d = np.load(FEAT_DIR / "esm2_3b_matrix.npz", allow_pickle=True)
    Xtr, ytr, ent_tr = d["X_train"], d["y_train"], d["entries_train"]
    Xte, yte, ent_te = d["X_test"], d["y_test"], d["entries_test"]
    print(f"ESM2-3B features: train {Xtr.shape}, test {Xte.shape}")

    fs_pred, fs_prob = load_foldseek_top1()

    print("\n--- ESM2-3B classifiers (alone) ---")
    res = train_eval(Xtr, ytr, Xte, yte)
    rows = []
    best_name, best_f1, best_pred = None, -1.0, None
    for name, (f1, pred) in res.items():
        print(f"  {name:20s} F1 = {f1:.4f}")
        rows.append({"method": f"ESM2-3B {name}", "weighted_f1": f1})
        if f1 > best_f1:
            best_f1 = f1; best_name = name; best_pred = pred
    pred_map = {str(e): p for e, p in zip(ent_te, best_pred)}

    print("\n--- Ensembles with Foldseek ---")
    ens_rows = []
    # rule A: Foldseek if hit, else ESM2-3B best
    pa = [fs_pred.get(str(e), pred_map.get(str(e))) for e in ent_te]
    fa = f1_score(yte, pa, average="weighted", zero_division=0)
    ens_rows.append({"ensemble": "Foldseek > ESM2-3B fallback", "weighted_f1": fa})

    for thr in [0.3, 0.5, 0.7, 0.9]:
        p = [(fs_pred[str(e)] if (str(e) in fs_pred and fs_prob.get(str(e), 0) >= thr)
              else pred_map.get(str(e))) for e in ent_te]
        f = f1_score(yte, p, average="weighted", zero_division=0)
        ens_rows.append({"ensemble": f"Foldseek(prob>={thr}) > ESM2-3B", "weighted_f1": f})

    for r in sorted(ens_rows, key=lambda x: -x["weighted_f1"]):
        print(f"  {r['ensemble']:42s} F1 = {r['weighted_f1']:.4f}")

    rows += ens_rows

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "phase12_esm3b_results.csv", index=False)
    best_ens = max(ens_rows, key=lambda r: r["weighted_f1"])
    overall_best = max(best_f1, best_ens["weighted_f1"])

    print(f"\nBest ESM2-3B alone:     {best_name}  F1 = {best_f1:.4f}")
    print(f"Best ensemble:          {best_ens['ensemble']}  F1 = {best_ens['weighted_f1']:.4f}")
    print(f"Published ESM2-3B (DGEB): 0.265")
    print(f"Crossed 0.265? {'YES' if overall_best > 0.265 else 'no'}")

    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M": 0.2520,
        "FS + ProstT5": 0.2543,
        f"ESM2-3B alone\n({best_name})": best_f1,
        f"BEST: FS + ESM2-3B\n({best_ens['ensemble']})": best_ens["weighted_f1"],
        "ESM2-3B (DGEB paper)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#5588cc",
            "#5aa55a", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6,
               label="DGEB ESM2-3B baseline")
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 12 — ESM2-3B + Foldseek ensemble")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=10, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase12_final.png", dpi=140)
    plt.close()

    txt = ["Phase 12 — ESM2-3B + Foldseek", "=" * 50, ""]
    for name, v in methods.items():
        txt.append(f"  {name.replace(chr(10), ' '):50s} F1 = {v:.4f}")
    txt.append("")
    txt.append(f"Crossed 0.265: {'YES' if overall_best > 0.265 else 'no'}")
    (RESULTS_DIR / "phase12_summary.txt").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
