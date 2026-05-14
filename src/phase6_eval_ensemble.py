"""
Phase 6 evaluator + Foldseek/ESM2 ensemble.

Stages:
  1. Train SVM/RF/LR/kNN/centroid classifier on ESM2-35M embeddings.
     Print weighted F1 per model (over all 391 structure-bearing test rows,
     no missing penalty since ESM2 is sequence-only and covers every row).
  2. Build an ensemble with Foldseek top-1:
       - if Foldseek and ESM2-best agree, use that EC
       - if they disagree, prefer Foldseek (higher accuracy on shared queries)
         unless Foldseek has no hit, in which case use ESM2
     Plus an alternative: confidence-gated — use Foldseek if top-1 prob > thr,
     else fall back to ESM2.
  3. Also try the L2-normalized centroid (NCM) classifier on ESM2: for each
     EC class compute the mean of train embeddings, classify test by cosine.
     NCM is the natural fit for 5-shot 400-class settings.
  4. Output a final big chart of all methods.

Outputs:
  results/phase6_esm2_results.csv
  results/phase6_ensemble_summary.csv
  results/phase6_summary.txt
  charts/phase6_final_methods.png
"""
from __future__ import annotations

import argparse
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


def load_esm2(matrix_path: Path):
    d = np.load(matrix_path, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["entries_train"],
            d["X_test"], d["y_test"], d["entries_test"])


def load_foldseek_top1() -> tuple[dict, dict]:
    """Return (top1_pred_by_entry, top1_score_by_entry) from Phase 5a hits."""
    df = pd.read_csv(ROOT / "foldseek_workdir/hits.tsv", sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = df.sort_values("bits", ascending=False).groupby("q").head(1)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    tl = dict(zip(train_df["Entry"], train_df["Label"]))
    pred = {q: tl[t] for q, t in zip(top["q"], top["target"]) if t in tl}
    score = dict(zip(top["q"], top["prob"]))
    return pred, score


def ncm(Xtr, ytr, Xte) -> np.ndarray:
    """Nearest class-mean (centroid) on L2-normalized features (cosine)."""
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
        "kNN-1":        KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        "kNN-3":        KNeighborsClassifier(n_neighbors=3, metric="cosine"),
        "kNN-5":        KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "RF-500":       RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                random_state=0,
                                                class_weight="balanced"),
    }
    for name, m in models.items():
        Xa, Xb = (Xtr, Xte) if "RF" in name else (Xtr_s, Xte_s)
        m.fit(Xa, ytr)
        pred = m.predict(Xb)
        out[name] = (f1_score(yte, pred, average="weighted", zero_division=0), pred)
    # nearest class-mean on raw (cosine)
    p = ncm(Xtr, ytr, Xte)
    out["ESM2 NCM (cosine)"] = (f1_score(yte, p, average="weighted", zero_division=0), p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(FEAT_DIR / "esm2_35m_matrix.npz"),
                    help="path to embedding matrix .npz")
    ap.add_argument("--tag", default=None,
                    help="tag for output filenames (defaults from matrix path)")
    args = ap.parse_args()
    matrix_path = Path(args.matrix)
    tag = args.tag or matrix_path.stem.replace("_matrix", "")

    Xtr, ytr, ent_tr, Xte, yte, ent_te = load_esm2(matrix_path)
    print(f"{tag} features: train {Xtr.shape}, test {Xte.shape}")

    fs_pred, fs_prob = load_foldseek_top1()

    print(f"\n--- {tag} classifiers on full 400-row test set ---")
    res = train_eval(Xtr, ytr, Xte, yte)
    rows = []
    best_esm_name = None; best_esm_f1 = -1; best_esm_pred = None
    for name, (f1, pred) in res.items():
        print(f"  {name:22s} F1 = {f1:.4f}")
        rows.append({"method": name, "weighted_f1": f1})
        if f1 > best_esm_f1:
            best_esm_f1 = f1; best_esm_name = name; best_esm_pred = pred
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"phase6_{tag}_results.csv", index=False)

    # ensemble
    test_labels = dict(zip(ent_te, yte))
    esm_pred_map = {str(e): p for e, p in zip(ent_te, best_esm_pred)}

    ensemble_rows = []
    # rule A: Foldseek if it has a hit, else ESM2
    pred_A = {q: fs_pred.get(q, esm_pred_map.get(q)) for q in ent_te}
    f1_A = f1_score(yte, [pred_A[str(e)] for e in ent_te],
                    average="weighted", zero_division=0)
    ensemble_rows.append({"ensemble": "Foldseek > ESM2(best) fallback",
                          "weighted_f1": f1_A})

    # rule B: confidence-gated by Foldseek prob threshold
    for thr in [0.3, 0.5, 0.7, 0.9]:
        pred_B = {q: (fs_pred[q] if (q in fs_pred and fs_prob.get(q, 0) >= thr)
                      else esm_pred_map.get(q)) for q in ent_te}
        f1_B = f1_score(yte, [pred_B[str(e)] for e in ent_te],
                        average="weighted", zero_division=0)
        ensemble_rows.append({"ensemble": f"Foldseek(prob>={thr}) > ESM2",
                              "weighted_f1": f1_B})

    # rule C: ESM2 if it agrees with Foldseek, else use Foldseek
    pred_C = {}
    for q in ent_te:
        fs = fs_pred.get(q); es = esm_pred_map.get(q)
        if fs is None:
            pred_C[q] = es
        elif fs == es:
            pred_C[q] = fs
        else:
            pred_C[q] = fs   # prefer foldseek on disagreement (matches Tier-A finding)
    f1_C = f1_score(yte, [pred_C[str(e)] for e in ent_te],
                    average="weighted", zero_division=0)
    ensemble_rows.append({"ensemble": "Agreement: prefer Foldseek on disagree",
                          "weighted_f1": f1_C})

    # rule D: prefer ESM2 on disagreement
    pred_D = {}
    for q in ent_te:
        fs = fs_pred.get(q); es = esm_pred_map.get(q)
        if fs is None:
            pred_D[q] = es
        elif fs == es:
            pred_D[q] = fs
        else:
            pred_D[q] = es
    f1_D = f1_score(yte, [pred_D[str(e)] for e in ent_te],
                    average="weighted", zero_division=0)
    ensemble_rows.append({"ensemble": "Agreement: prefer ESM2 on disagree",
                          "weighted_f1": f1_D})

    print("\n--- Ensembles ---")
    for r in ensemble_rows:
        print(f"  {r['ensemble']:48s} F1={r['weighted_f1']:.4f}")
    pd.DataFrame(ensemble_rows).to_csv(RESULTS_DIR / f"phase6_{tag}_ensemble_summary.csv",
                                       index=False)

    # final chart
    best_ensemble = max(ensemble_rows, key=lambda r: r["weighted_f1"])
    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "Foldseek top-1 (Phase 5a)": 0.2383,
        f"{tag} best\n({best_esm_name})": best_esm_f1,
        f"BEST ensemble\n({best_ensemble['ensemble']})": best_ensemble["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cols = ["#bbb", "#88a", "#4c72b0", "#5aa55a", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes — final methods comparison")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"phase6_{tag}_final_methods.png", dpi=140)
    plt.close()

    txt = ["Final comparison",
           "=" * 50,
           f"Best ESM2 classifier: {best_esm_name} F1 = {best_esm_f1:.4f}",
           f"Best ensemble:        {best_ensemble['ensemble']} F1 = {best_ensemble['weighted_f1']:.4f}",
           "",
           "All methods:"]
    for name, v in methods.items():
        txt.append(f"  {name.replace(chr(10),' '):50s} F1 = {v:.4f}")
    (RESULTS_DIR / f"phase6_{tag}_summary.txt").write_text("\n".join(txt))
    print("\n" + "\n".join(txt))


if __name__ == "__main__":
    main()
