"""
Phase 11: multi-model ensemble + concatenation experiments.

We have three sequence-based embeddings on disk:
  ESM2-35M    (480-D)
  ESM2-150M   (640-D)
  ProstT5     (1024-D)

And Foldseek top-1 predictions with per-query 'prob' (alignment probability).

Experiments:
  A. Each LM alone (LogReg)
  B. Concatenate all three -> single LogReg/kNN/SVM
  C. Confidence-gated ensembles using each fallback model
  D. Multi-model majority fallback (when Foldseek prob < thr)
  E. Concatenated-feature classifier as fallback (rule C with the best
     single classifier on the concatenation)

Outputs:
  results/phase11_multimodel_results.csv
  results/phase11_summary.txt
  charts/phase11_final.png
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def load_emb(name: str):
    d = np.load(FEAT_DIR / name, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["entries_train"],
            d["X_test"], d["y_test"], d["entries_test"])


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


def lr_predict(Xtr, ytr, Xte):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(Xtr_s, ytr)
    return m.predict(Xte_s)


def main():
    Xa, ytr, ent_tr, Ta, yte, ent_te = load_emb("esm2_35m_matrix.npz")
    Xb, _, _, Tb, _, _ = load_emb("esm2_t30_150M_matrix.npz")
    Xc, _, _, Tc, _, _ = load_emb("prostT5_aa_matrix.npz")
    print(f"ESM2-35M: {Xa.shape}  ESM2-150M: {Xb.shape}  ProstT5: {Xc.shape}")

    # row-align by entry id
    def align(entries_a, entries_b, Xb_full):
        idx = {str(e): i for i, e in enumerate(entries_b)}
        order = [idx[str(e)] for e in entries_a]
        return Xb_full[order]
    Xb_tr = align(ent_tr, np.load(FEAT_DIR/"esm2_t30_150M_matrix.npz", allow_pickle=True)["entries_train"], Xb)
    Xc_tr = align(ent_tr, np.load(FEAT_DIR/"prostT5_aa_matrix.npz", allow_pickle=True)["entries_train"], Xc)
    Xb_te = align(ent_te, np.load(FEAT_DIR/"esm2_t30_150M_matrix.npz", allow_pickle=True)["entries_test"], Tb)
    Xc_te = align(ent_te, np.load(FEAT_DIR/"prostT5_aa_matrix.npz", allow_pickle=True)["entries_test"], Tc)

    # L2-normalize each block, then concatenate (keeps scales comparable)
    Xa_n, Xb_n, Xc_n = normalize(Xa), normalize(Xb_tr), normalize(Xc_tr)
    Ta_n, Tb_n, Tc_n = normalize(Ta), normalize(Xb_te), normalize(Xc_te)
    X_cat_tr = np.hstack([Xa_n, Xb_n, Xc_n])
    X_cat_te = np.hstack([Ta_n, Tb_n, Tc_n])
    print(f"concatenated: train {X_cat_tr.shape}  test {X_cat_te.shape}")

    fs_pred, fs_prob = load_foldseek_top1()

    # === A. each LM alone (LogReg) ===
    print("\n=== A. each LM alone (LogReg) ===")
    rows = []
    preds_alone = {}
    for name, Xtr_e, Xte_e in [("ESM2-35M",  Xa,   Ta),
                                ("ESM2-150M", Xb_tr, Xb_te),
                                ("ProstT5",   Xc_tr, Xc_te),
                                ("CAT-2144D", X_cat_tr, X_cat_te)]:
        p = lr_predict(Xtr_e, ytr, Xte_e)
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        rows.append({"method": f"{name} LR", "weighted_f1": f1})
        preds_alone[name] = {str(e): pi for e, pi in zip(ent_te, p)}
        print(f"  {name:10s} LR  F1 = {f1:.4f}")

    # === D. Multi-model majority fallback ===
    print("\n=== D. Foldseek + multi-model majority fallback ===")
    fb_keys = ["ESM2-35M", "ESM2-150M", "ProstT5"]
    def majority(votes):
        return Counter(votes).most_common(1)[0][0]
    for thr in [0.3, 0.5, 0.7]:
        preds = []
        for e in ent_te:
            e = str(e)
            if e in fs_pred and fs_prob.get(e, 0) >= thr:
                preds.append(fs_pred[e])
            else:
                votes = [preds_alone[k][e] for k in fb_keys if e in preds_alone[k]]
                preds.append(majority(votes) if votes else "__none__")
        f1 = f1_score(yte, preds, average="weighted", zero_division=0)
        rows.append({"method": f"FS(prob>={thr}) > MAJ(35M+150M+ProstT5)", "weighted_f1": f1})
        print(f"  FS(prob>={thr}) > MAJ            F1 = {f1:.4f}")

    # === E. Concatenated-feature classifier as fallback ===
    print("\n=== E. Foldseek + concat-LR fallback ===")
    cat_pred_map = preds_alone["CAT-2144D"]
    for thr in [0.3, 0.5, 0.7]:
        preds = []
        for e in ent_te:
            e = str(e)
            if e in fs_pred and fs_prob.get(e, 0) >= thr:
                preds.append(fs_pred[e])
            else:
                preds.append(cat_pred_map.get(e, "__none__"))
        f1 = f1_score(yte, preds, average="weighted", zero_division=0)
        rows.append({"method": f"FS(prob>={thr}) > CAT-LR", "weighted_f1": f1})
        print(f"  FS(prob>={thr}) > CAT-LR         F1 = {f1:.4f}")

    # === Compare to single-model baselines (re-run for fairness) ===
    print("\n=== Reference single-model fallbacks ===")
    for k in fb_keys:
        for thr in [0.5]:
            preds = []
            for e in ent_te:
                e = str(e)
                if e in fs_pred and fs_prob.get(e, 0) >= thr:
                    preds.append(fs_pred[e])
                else:
                    preds.append(preds_alone[k].get(e, "__none__"))
            f1 = f1_score(yte, preds, average="weighted", zero_division=0)
            rows.append({"method": f"FS(prob>={thr}) > {k}", "weighted_f1": f1})
            print(f"  FS(prob>=0.5) > {k:9s}     F1 = {f1:.4f}")

    df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "phase11_multimodel_results.csv", index=False)
    best = df.iloc[0]
    print(f"\nBEST: {best['method']}  F1 = {best['weighted_f1']:.4f}")

    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M": 0.2520,
        "FS + ProstT5": 0.2543,
        f"BEST Phase 11\n({best['method']})": best["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(14, 5.5))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#5aa55a",
            "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 11 — multi-model ensembles")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=10, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase11_final.png", dpi=140)
    plt.close()

    (RESULTS_DIR / "phase11_summary.txt").write_text(
        "Phase 11 — multi-model\n" + "=" * 40 + "\n" +
        df.to_string(index=False) + "\n"
    )


if __name__ == "__main__":
    main()
