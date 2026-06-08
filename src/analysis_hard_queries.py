"""
Analysis: what makes a Convergent Enzymes test query hard?

We do not have NT-v2 per-query predictions to directly contrast against
ESM2-3B (the DGEB authors did not release them publicly). So we do the
proxy: characterize which test queries our Foldseek + AA-LM ensemble gets
right vs wrong, then look for patterns in the hard queries.

Properties measured per test query:
  - protein sequence length
  - CDS DNA length (from our fetched data/dna_sequences.csv)
  - CDS GC content
  - codon usage entropy (proxy for codon bias)
  - whether Foldseek had a hit at all
  - top-1 Foldseek prob (model confidence)
  - whether the top-1 train neighbour's EC matches the true EC
  - per-class size (# train examples — should be 5 for everyone, sanity)

Then: split the 400 test queries into "ensemble-correct" vs
"ensemble-incorrect", and compare property distributions.

Outputs:
  results/analysis_hard_queries.csv
  charts/analysis_hard_vs_easy.png
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


def gc_content(seq: str) -> float:
    seq = seq.upper()
    n = sum(1 for c in seq if c in "ACGT")
    if n == 0:
        return float("nan")
    return sum(1 for c in seq if c in "GC") / n


def codon_entropy(seq: str) -> float:
    """Shannon entropy of codon frequencies (proxy for codon bias). 0 = highly biased, max ~5.95 bits for uniform 61 codons."""
    seq = "".join(c for c in seq.upper() if c in "ACGT")
    if len(seq) < 6:
        return float("nan")
    n = len(seq) // 3
    codons = [seq[i*3:i*3+3] for i in range(n) if seq[i*3:i*3+3] not in ("TAA", "TAG", "TGA")]
    if not codons:
        return float("nan")
    c = Counter(codons)
    total = sum(c.values())
    return float(-sum((v/total) * np.log2(v/total) for v in c.values()))


def run_ensemble_predictions():
    """Reproduce Phase 13 best ensemble on Convergent Enzymes, return per-query predictions."""
    d3b = np.load(FEAT_DIR / "esm2_3b_matrix.npz", allow_pickle=True)
    ent_te = d3b["entries_test"]
    ytr, yte = d3b["y_train"], d3b["y_test"]
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    tl = dict(zip(train_df["Entry"], train_df["Label"]))

    # Foldseek top-1
    fs = pd.read_csv(ROOT / "foldseek_workdir/hits.tsv", sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = fs.sort_values("bits", ascending=False).groupby("q").head(1)
    fs_pred = {q: tl[t] for q, t in zip(top["q"], top["target"]) if t in tl}
    fs_prob = dict(zip(top["q"], top["prob"]))

    # Per-LM LR predictions (aligned by entry_train order)
    def align(target_e, src_e, X):
        idx = {str(e): i for i, e in enumerate(src_e)}
        return X[[idx[str(e)] for e in target_e]]

    def lr_predict(Xtr, ytr, Xte):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        m = LogisticRegression(C=1.0, max_iter=2000); m.fit(Xtr_s, ytr)
        return m.predict(Xte_s)

    X3b_tr, X3b_te = d3b["X_train"], d3b["X_test"]
    dp = np.load(FEAT_DIR / "prostT5_aa_matrix.npz", allow_pickle=True)
    Xp_tr = align(d3b["entries_train"], dp["entries_train"], dp["X_train"])
    Xp_te = align(ent_te, dp["entries_test"], dp["X_test"])
    d150 = np.load(FEAT_DIR / "esm2_t30_150M_matrix.npz", allow_pickle=True)
    X150_tr = align(d3b["entries_train"], d150["entries_train"], d150["X_train"])
    X150_te = align(ent_te, d150["entries_test"], d150["X_test"])

    preds_3b = lr_predict(X3b_tr, ytr, X3b_te)
    preds_p = lr_predict(Xp_tr, ytr, Xp_te)
    preds_150 = lr_predict(X150_tr, ytr, X150_te)

    # Ensemble: FS(prob>=0.9) -> top-1, else majority(3B, ProstT5, 150M)
    from collections import Counter as C
    ens_pred = []
    for i, e in enumerate(ent_te):
        e = str(e)
        if e in fs_pred and fs_prob.get(e, 0) >= 0.9:
            ens_pred.append(fs_pred[e])
        else:
            votes = [preds_3b[i], preds_p[i], preds_150[i]]
            ens_pred.append(C(votes).most_common(1)[0][0])
    return ent_te, yte, np.array(ens_pred), fs_pred, fs_prob, preds_3b, preds_p, preds_150


def main():
    ent_te, yte, ens_pred, fs_pred, fs_prob, p3b, pp, p150 = run_ensemble_predictions()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    dna = pd.read_csv(DATA_DIR / "dna_sequences.csv")
    dna = dna.set_index("Entry")

    rows = []
    test_seqs = dict(zip(test_df["Entry"], test_df["Sequence"]))
    for i, e in enumerate(ent_te):
        e = str(e)
        true_ec = yte[i]
        ens_correct = (ens_pred[i] == true_ec)
        fs_correct = (fs_pred.get(e) == true_ec) if e in fs_pred else False
        esm_correct = (p3b[i] == true_ec)
        prot_seq = test_seqs.get(e, "")
        dna_seq = ""
        if e in dna.index:
            row = dna.loc[e]
            if row["status"] == "ok":
                dna_seq = str(row["cds_dna"])
        rows.append({
            "Entry": e,
            "true_ec": true_ec,
            "ensemble_correct": ens_correct,
            "foldseek_correct": fs_correct,
            "esm2_3b_correct": esm_correct,
            "fs_prob": float(fs_prob.get(e, 0.0)),
            "has_fs_hit": e in fs_pred,
            "protein_len": len(prot_seq),
            "cds_len": len(dna_seq),
            "gc_content": gc_content(dna_seq),
            "codon_entropy": codon_entropy(dna_seq),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "analysis_hard_queries.csv", index=False)

    print(f"=== Convergent Enzymes hard-vs-easy analysis ===")
    print(f"n_test: {len(df)}")
    print(f"ensemble correct: {df['ensemble_correct'].sum()} ({100*df['ensemble_correct'].mean():.1f}%)")
    print(f"foldseek correct: {df['foldseek_correct'].sum()}")
    print(f"esm2-3b correct : {df['esm2_3b_correct'].sum()}")
    print()
    print("Mean of properties by ensemble correctness:")
    by_corr = df.groupby("ensemble_correct")[
        ["fs_prob", "protein_len", "cds_len", "gc_content", "codon_entropy"]
    ].mean()
    print(by_corr.round(3))
    print()

    # Mann-Whitney U test for each property
    from scipy.stats import mannwhitneyu
    print("Mann-Whitney U test (hard vs easy):")
    for prop in ["fs_prob", "protein_len", "cds_len", "gc_content", "codon_entropy"]:
        easy = df[df["ensemble_correct"]][prop].dropna()
        hard = df[~df["ensemble_correct"]][prop].dropna()
        if len(easy) > 5 and len(hard) > 5:
            u, p = mannwhitneyu(easy, hard, alternative="two-sided")
            print(f"  {prop:18s} easy_mean={easy.mean():.3f} hard_mean={hard.mean():.3f} p={p:.4g}")

    # Plot: distribution of properties by correct/incorrect
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, prop in zip(axes, ["fs_prob", "protein_len", "gc_content", "codon_entropy"]):
        easy = df[df["ensemble_correct"]][prop].dropna()
        hard = df[~df["ensemble_correct"]][prop].dropna()
        ax.boxplot([easy, hard], labels=["correct", "incorrect"])
        ax.set_title(prop)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Convergent Enzymes: ensemble-correct vs ensemble-incorrect test queries")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "analysis_hard_vs_easy.png", dpi=140)
    plt.close()
    print(f"\nsaved {CHARTS_DIR / 'analysis_hard_vs_easy.png'}")


if __name__ == "__main__":
    main()
