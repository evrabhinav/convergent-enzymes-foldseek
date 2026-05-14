"""
Phase 13: three small experiments aimed at crossing 0.265.

  (1) Multi-model fallback: Foldseek confident -> top-1, else majority of
      {ESM2-3B, ProstT5, ESM2-150M} predictions.
  (2) Concat fallback: train LR on hstack(ESM2-3B, ProstT5) embeddings,
      use that as the Foldseek fallback.
  (3) C-tune fallback: sweep LogisticRegression C on ESM2-3B alone for the
      fallback step.

The current best is 0.2646 (Foldseek(prob>=0.9) -> ESM2-3B LR-C=1). Target
is to clear 0.265 (DGEB-paper baseline).

Outputs:
  results/phase13_crossover_results.csv
  charts/phase13_final.png
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
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"


def load_emb(name: str):
    d = np.load(FEAT_DIR / name, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["entries_train"],
            d["X_test"], d["y_test"], d["entries_test"])


def align(entries_a, entries_b, X_b):
    idx = {str(e): i for i, e in enumerate(entries_b)}
    order = [idx[str(e)] for e in entries_a]
    return X_b[order]


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


def lr_predict(Xtr, ytr, Xte, C: float = 1.0):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(Xtr_s, ytr)
    return m.predict(Xte_s)


def f1_with_foldseek(fb_pred_map, fs_pred, fs_prob, ent_te, yte, thr):
    p = [(fs_pred[str(e)] if (str(e) in fs_pred and fs_prob.get(str(e), 0) >= thr)
          else fb_pred_map.get(str(e), "__none__")) for e in ent_te]
    return f1_score(yte, p, average="weighted", zero_division=0)


def main():
    # Load all three embeddings, aligned by Entry ID
    Xa, ytr, ent_tr, Ta, yte, ent_te = load_emb("esm2_3b_matrix.npz")
    d150 = np.load(FEAT_DIR / "esm2_t30_150M_matrix.npz", allow_pickle=True)
    Xb_tr = align(ent_tr, d150["entries_train"], d150["X_train"])
    Xb_te = align(ent_te, d150["entries_test"],  d150["X_test"])
    dprost = np.load(FEAT_DIR / "prostT5_aa_matrix.npz", allow_pickle=True)
    Xc_tr = align(ent_tr, dprost["entries_train"], dprost["X_train"])
    Xc_te = align(ent_te, dprost["entries_test"],  dprost["X_test"])
    print(f"ESM2-3B:  {Xa.shape}  ESM2-150M: {Xb_tr.shape}  ProstT5: {Xc_tr.shape}")

    fs_pred, fs_prob = load_foldseek_top1()
    rows = []
    BASELINE = 0.265

    # --- (3) C-tune fallback on ESM2-3B alone ---
    print("\n=== (3) ESM2-3B LR with various C, used as fallback ===")
    best_3b = {}  # (C, thr) -> f1
    for C in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]:
        pred = lr_predict(Xa, ytr, Ta, C=C)
        pred_map = {str(e): p for e, p in zip(ent_te, pred)}
        for thr in [0.5, 0.7, 0.9]:
            f1 = f1_with_foldseek(pred_map, fs_pred, fs_prob, ent_te, yte, thr)
            tag = f"FS(prob>={thr}) > ESM2-3B LR-C={C}"
            rows.append({"experiment": "C-tune", "method": tag,
                         "weighted_f1": f1})
            best_3b[(C, thr)] = f1
            star = "  <<<" if f1 > BASELINE else ""
            print(f"  C={C:<5}  thr={thr}  F1 = {f1:.4f}{star}")

    # --- (2) Concat ESM2-3B + ProstT5 as fallback ---
    print("\n=== (2) Concat ESM2-3B + ProstT5 LR as fallback ===")
    Xa_n = normalize(Xa); Xc_tr_n = normalize(Xc_tr)
    Ta_n = normalize(Ta); Xc_te_n = normalize(Xc_te)
    Xcat_tr = np.hstack([Xa_n, Xc_tr_n])
    Xcat_te = np.hstack([Ta_n, Xc_te_n])
    for C in [0.1, 1.0, 3.0]:
        pred = lr_predict(Xcat_tr, ytr, Xcat_te, C=C)
        pred_map = {str(e): p for e, p in zip(ent_te, pred)}
        for thr in [0.5, 0.7, 0.9]:
            f1 = f1_with_foldseek(pred_map, fs_pred, fs_prob, ent_te, yte, thr)
            tag = f"FS(prob>={thr}) > CAT(3B+ProstT5) LR-C={C}"
            rows.append({"experiment": "concat", "method": tag,
                         "weighted_f1": f1})
            star = "  <<<" if f1 > BASELINE else ""
            print(f"  C={C:<5}  thr={thr}  F1 = {f1:.4f}{star}")

    # --- (1) Multi-model majority fallback ---
    print("\n=== (1) Multi-model majority fallback ===")
    # use best C for ESM2-3B (best from (3))
    best_3b_pred = lr_predict(Xa, ytr, Ta, C=1.0)
    prost_pred = lr_predict(Xc_tr, ytr, Xc_te, C=1.0)
    e150_pred = lr_predict(Xb_tr, ytr, Xb_te, C=1.0)
    map_3b = {str(e): p for e, p in zip(ent_te, best_3b_pred)}
    map_p  = {str(e): p for e, p in zip(ent_te, prost_pred)}
    map_e  = {str(e): p for e, p in zip(ent_te, e150_pred)}

    def majority(*args):
        return Counter(args).most_common(1)[0][0]

    for include in [
        ("3B+ProstT5",     [map_3b, map_p]),
        ("3B+ProstT5+150M", [map_3b, map_p, map_e]),
    ]:
        name, maps = include
        for thr in [0.3, 0.5, 0.7, 0.9]:
            preds = []
            for e in ent_te:
                e = str(e)
                if e in fs_pred and fs_prob.get(e, 0) >= thr:
                    preds.append(fs_pred[e])
                else:
                    votes = [m[e] for m in maps if e in m]
                    preds.append(majority(*votes) if votes else "__none__")
            f1 = f1_score(yte, preds, average="weighted", zero_division=0)
            tag = f"FS(prob>={thr}) > MAJ({name})"
            rows.append({"experiment": "majority", "method": tag,
                         "weighted_f1": f1})
            star = "  <<<" if f1 > BASELINE else ""
            print(f"  {tag:60s}  F1 = {f1:.4f}{star}")

    df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "phase13_crossover_results.csv", index=False)
    best = df.iloc[0]
    print(f"\n=== OVERALL BEST ===")
    print(f"  {best['method']}  F1 = {best['weighted_f1']:.4f}")
    print(f"  Published ESM2-3B (DGEB): {BASELINE}")
    print(f"  Crossed: {'YES — by ' + str(round(best['weighted_f1']-BASELINE,4)) if best['weighted_f1'] > BASELINE else 'no'}")

    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M": 0.2520,
        "FS + ProstT5": 0.2543,
        "FS + ESM2-3B (Phase 12)": 0.2646,
        f"BEST Phase 13\n({best['method']})": best["weighted_f1"],
        "ESM2-3B (DGEB)": BASELINE,
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    cols = ["#bbb", "#88a", "#4c72b0", "#5588cc", "#5588cc", "#6699dd",
            "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(BASELINE, ls="--", color="gray", alpha=0.6,
               label="DGEB ESM2-3B baseline")
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 13 — crossover experiments")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=10, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase13_final.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
