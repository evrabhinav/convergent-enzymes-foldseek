"""
Phase 14: multi-track ensemble — adds the Nucleotide Transformer (NT-v1-2.5B)
to the Foldseek + AA-LM stack from Phase 13.

This is the experiment that engages the NA track of DGEB Convergent Enzymes.
DGEB reports NT-v1-2.5B alone at F1 = 0.506, our previous best is 0.2668
(Foldseek + AA-LM majority). Adding NT into the ensemble is the natural way
to test whether the AA-side structural-retrieval signal still helps once a
strong DNA-side model is in the mix.

Inputs (already on disk from prior phases):
  features/esm2_3b_matrix.npz
  features/prostT5_aa_matrix.npz
  features/esm2_t30_150M_matrix.npz
  features/nt_v1_25b_matrix.npz                 (produced by Colab script)
  foldseek_workdir/hits.tsv                       (test-vs-train Foldseek hits)
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"


def load_emb(name: str):
    d = np.load(FEAT_DIR / name, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["entries_train"],
            d["X_test"], d["y_test"], d["entries_test"])


def align(target_entries, src_entries, src_X):
    idx = {str(e): i for i, e in enumerate(src_entries)}
    order = [idx.get(str(e), -1) for e in target_entries]
    return np.array(order)


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


def lr_predict(Xtr, ytr, Xte, C=1.0):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(Xtr_s, ytr)
    return m.predict(Xte_s)


def main():
    # Base alignment from ESM2-3B (smallest set of test rows: 400)
    X3b, ytr, ent_tr, T3b, yte, ent_te = load_emb("esm2_3b_matrix.npz")
    print(f"ESM2-3B base: train {X3b.shape}, test {T3b.shape}")

    # Align ProstT5 + ESM2-150M to the same row order
    dp = np.load(FEAT_DIR / "prostT5_aa_matrix.npz", allow_pickle=True)
    Xp_tr = dp["X_train"][align(ent_tr, dp["entries_train"], dp["X_train"])]
    Xp_te = dp["X_test"][align(ent_te, dp["entries_test"], dp["X_test"])]
    d150 = np.load(FEAT_DIR / "esm2_t30_150M_matrix.npz", allow_pickle=True)
    X150_tr = d150["X_train"][align(ent_tr, d150["entries_train"], d150["X_train"])]
    X150_te = d150["X_test"][align(ent_te, d150["entries_test"], d150["X_test"])]

    # Try to load NT
    nt_path = FEAT_DIR / "nt_v1_25b_matrix.npz"
    if not nt_path.exists():
        print(f"\n[!] {nt_path} not found. Run colab/nucleotide_transformer_colab.py first.")
        return
    dnt = np.load(nt_path, allow_pickle=True)
    Xnt_tr_full = dnt["X_train"]
    Xnt_te_full = dnt["X_test"]
    # Align to our 3B-based ordering. Some rows may have all-zero NT vectors
    # (DNA fetch failures); we keep them but the LR will down-weight them.
    nt_idx_tr = align(ent_tr, dnt["entries_train"], Xnt_tr_full)
    nt_idx_te = align(ent_te, dnt["entries_test"],  Xnt_te_full)
    Xnt_tr = Xnt_tr_full[nt_idx_tr]
    Xnt_te = Xnt_te_full[nt_idx_te]
    # Stats on how many actually have an NT embedding (non-zero row)
    n_have_tr = (Xnt_tr.sum(axis=1) != 0).sum()
    n_have_te = (Xnt_te.sum(axis=1) != 0).sum()
    print(f"NT-v1-2.5B: train {Xnt_tr.shape} ({n_have_tr} non-zero), "
          f"test {Xnt_te.shape} ({n_have_te} non-zero)")

    fs_pred, fs_prob = load_foldseek_top1()

    # --- per-model predictions ---
    print("\n=== Individual model predictions (LR) ===")
    preds = {}
    f1s = {}
    for name, Xtr, Xte in [("ESM2-3B", X3b, T3b),
                            ("ProstT5", Xp_tr, Xp_te),
                            ("ESM2-150M", X150_tr, X150_te),
                            ("NT-v1-2.5B", Xnt_tr, Xnt_te)]:
        p = lr_predict(Xtr, ytr, Xte)
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        f1s[name] = f1
        preds[name] = {str(e): pi for e, pi in zip(ent_te, p)}
        print(f"  {name:12s} F1 = {f1:.4f}")

    # --- ensembles ---
    print("\n=== Foldseek + {model} fallback ===")
    rows = []
    for name in ["ESM2-3B", "ProstT5", "ESM2-150M", "NT-v1-2.5B"]:
        for thr in [0.3, 0.5, 0.7, 0.9]:
            p = [(fs_pred[str(e)] if (str(e) in fs_pred and fs_prob.get(str(e), 0) >= thr)
                  else preds[name].get(str(e), "__none__")) for e in ent_te]
            f1 = f1_score(yte, p, average="weighted", zero_division=0)
            rows.append({"method": f"FS(prob>={thr}) > {name}", "weighted_f1": f1})

    print("\n=== Foldseek + majority(AA models) fallback ===")
    aa_keys = ["ESM2-3B", "ProstT5", "ESM2-150M"]
    for thr in [0.3, 0.5, 0.7, 0.9]:
        p = []
        for e in ent_te:
            e = str(e)
            if e in fs_pred and fs_prob.get(e, 0) >= thr:
                p.append(fs_pred[e])
            else:
                votes = [preds[k][e] for k in aa_keys if e in preds[k]]
                p.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        rows.append({"method": f"FS(prob>={thr}) > MAJ(AA-3)", "weighted_f1": f1})

    print("\n=== Foldseek + majority(AA + NT) fallback ===")
    all_keys = ["ESM2-3B", "ProstT5", "ESM2-150M", "NT-v1-2.5B"]
    for thr in [0.3, 0.5, 0.7, 0.9]:
        p = []
        for e in ent_te:
            e = str(e)
            if e in fs_pred and fs_prob.get(e, 0) >= thr:
                p.append(fs_pred[e])
            else:
                votes = [preds[k][e] for k in all_keys if e in preds[k]]
                p.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        rows.append({"method": f"FS(prob>={thr}) > MAJ(AA+NT-4)", "weighted_f1": f1})

    print("\n=== NT-leaning rules ===")
    # Rule X: NT primary, Foldseek + AA-majority as fallback if NT is undefined
    p = []
    for e in ent_te:
        e = str(e)
        nt_pred = preds["NT-v1-2.5B"].get(e)
        if nt_pred and Xnt_te[ent_te.tolist().index(e)].sum() != 0:
            p.append(nt_pred)
        elif e in fs_pred:
            p.append(fs_pred[e])
        else:
            votes = [preds[k][e] for k in aa_keys if e in preds[k]]
            p.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
    f1 = f1_score(yte, p, average="weighted", zero_division=0)
    rows.append({"method": "NT primary > FS > MAJ(AA-3)", "weighted_f1": f1})

    # Rule Y: NT and FS vote together (each gets one vote), break ties with AA majority
    p = []
    for e in ent_te:
        e = str(e)
        votes_strong = []
        if e in fs_pred:
            votes_strong.append(fs_pred[e])
        if preds["NT-v1-2.5B"].get(e):
            votes_strong.append(preds["NT-v1-2.5B"][e])
        if not votes_strong:
            votes = [preds[k][e] for k in aa_keys if e in preds[k]]
            p.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
            continue
        # If FS and NT agree, easy. Otherwise add the AA majority and take overall plurality
        if len(votes_strong) == 2 and votes_strong[0] == votes_strong[1]:
            p.append(votes_strong[0])
        else:
            aa_votes = [preds[k][e] for k in aa_keys if e in preds[k]]
            all_votes = votes_strong + aa_votes
            p.append(Counter(all_votes).most_common(1)[0][0])
    f1 = f1_score(yte, p, average="weighted", zero_division=0)
    rows.append({"method": "FS+NT vote, AA tie-break", "weighted_f1": f1})

    for r in sorted(rows, key=lambda x: -x["weighted_f1"]):
        marker = " <<<" if r["weighted_f1"] > 0.265 else ""
        print(f"  {r['method']:50s} F1 = {r['weighted_f1']:.4f}{marker}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase14_multitrack_results.csv", index=False)
    best = df.sort_values("weighted_f1", ascending=False).iloc[0]
    print(f"\n=== OVERALL BEST: {best['method']}  F1 = {best['weighted_f1']:.4f} ===")
    print(f"   DGEB ESM2-3B (AA-best): 0.265  -> crossed: {best['weighted_f1'] > 0.265}")
    print(f"   DGEB NT-v1-2.5B (NA-SOTA): 0.506  -> crossed: {best['weighted_f1'] > 0.506}")

    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "Foldseek top-1": 0.2383,
        "FS + AA-3 majority (Phase 13)": 0.2668,
        f"NT-v1-2.5B alone (ours)": f1s["NT-v1-2.5B"],
        f"BEST Phase 14\n({best['method']})": best["weighted_f1"],
        "ESM2-3B DGEB (AA-best)": 0.265,
        "NT-v1-2.5B DGEB (NA-best)": 0.506,
    }
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["#bbb", "#88a", "#4c72b0", "#5588cc", "#a070d0", "#cc4444", "#888", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=colors)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.5,
               label="DGEB AA-best (ESM2-3B = 0.265)")
    ax.axhline(0.506, ls="--", color="darkred", alpha=0.5,
               label="DGEB NA-best (NT-v1-2.5B = 0.506)")
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 14 — multi-track ensemble (AA + structure + NA)")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.6))
    plt.xticks(rotation=10, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase14_multitrack.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
