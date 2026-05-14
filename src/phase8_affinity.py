"""
Phase 8: Foldseek-affinity vector classifier.

Idea: each protein gets a 1969-D vector of Foldseek bit-scores against every
training protein. A classifier learns *which patterns* of training-set
similarity imply which EC class — strictly more information than top-K voting.

Two affinity matrices are loaded:
  foldseek_workdir/hits_train_train.tsv  (built by easy-search train trainDB)
  foldseek_workdir/hits.tsv              (test vs train, from Phase 5)

We mask the train self-similarity diagonal to zero so the classifier can't
trivially memorize "I match myself perfectly."

We try several classifiers:
  - L2 logistic regression on raw bit-score vectors
  - L2 logistic regression on log1p(bit-score) (more linear)
  - Random Forest
  - SVM with linear kernel
  - All of the above with ESM2-35M embeddings concatenated

Outputs:
  features/affinity_matrix.npz
  results/phase8_affinity_results.csv
  results/phase8_summary.txt
  charts/phase8_final.png
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
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
FS_DIR = ROOT / "foldseek_workdir"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"


def load_hits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    # filter contaminating fpocket _out entries that leaked into queries
    df = df[~df["q"].str.contains("_out", na=False)]
    df = df[~df["target"].str.contains("_out", na=False)]
    return df


def build_affinity(hits: pd.DataFrame, query_ids: list[str],
                   target_ids: list[str], col: str = "bits") -> np.ndarray:
    """Dense (n_query, n_target) matrix of `col` values; zero where no hit."""
    q_idx = {q: i for i, q in enumerate(query_ids)}
    t_idx = {t: i for i, t in enumerate(target_ids)}
    M = np.zeros((len(query_ids), len(target_ids)), dtype=np.float32)
    for q, t, v in zip(hits["q"], hits["target"], hits[col]):
        if q in q_idx and t in t_idx:
            M[q_idx[q], t_idx[t]] = float(v)
    return M


def evaluate(name, Xtr, ytr, Xte, yte, model, scale=False):
    if scale:
        sc = StandardScaler(with_mean=False)
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    f1 = f1_score(yte, pred, average="weighted", zero_division=0)
    print(f"  {name:55s} F1 = {f1:.4f}")
    return f1


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    train_ids = train_df["Entry"].tolist()
    test_ids = test_df["Entry"].tolist()

    hits_tr = load_hits(FS_DIR / "hits_train_train.tsv")
    hits_te = load_hits(FS_DIR / "hits.tsv")
    print(f"hits_train_train: {len(hits_tr)} rows | hits_test_train: {len(hits_te)} rows")

    # Build affinity matrices over the full 1969 train index
    # (proteins with structures are a subset of train_df)
    A_tr = build_affinity(hits_tr, train_ids, train_ids)
    A_te = build_affinity(hits_te, test_ids, train_ids)
    print(f"A_tr {A_tr.shape}  nonzero={np.count_nonzero(A_tr)}")
    print(f"A_te {A_te.shape}  nonzero={np.count_nonzero(A_te)}")

    # Mask train self-similarity to 0 (already 0 unless self-hit reported)
    np.fill_diagonal(A_tr, 0.0)

    y_tr = train_df["Label"].to_numpy()
    y_te = test_df["Label"].to_numpy()

    # Drop train rows with no structure (all-zero row in A_tr and all-zero column)
    # Actually proteins without structure won't appear in hits_tr at all → all-zero row.
    # Remove them so the classifier doesn't see degenerate examples.
    has_any_hit = A_tr.sum(axis=1) > 0
    print(f"Train rows with at least 1 Foldseek hit: {has_any_hit.sum()}/{len(train_ids)}")
    Atr = A_tr[has_any_hit]
    ytr = y_tr[has_any_hit]

    # Use log1p of bit-scores as a default transform
    Atr_log = np.log1p(Atr)
    Ate_log = np.log1p(A_te)

    # Load ESM2-35M embeddings for concat experiments
    esm = np.load(FEAT_DIR / "esm2_35m_matrix.npz", allow_pickle=True)
    train_id_to_idx = {e: i for i, e in enumerate(esm["entries_train"])}
    test_id_to_idx = {e: i for i, e in enumerate(esm["entries_test"])}
    E_tr_full = esm["X_train"]
    E_te = esm["X_test"]
    # Align ESM features to our train order (skip proteins removed above)
    train_keep_idx = [train_id_to_idx[train_ids[i]] for i in range(len(train_ids)) if has_any_hit[i]]
    E_tr = E_tr_full[train_keep_idx]

    rows = []
    print("\n=== Affinity vector only ===")
    rows.append({"method": "LR-l2 (raw bits)", "F1":
        evaluate("LR-l2 (raw bits)", Atr, ytr, A_te, y_te,
                 LogisticRegression(C=1.0, max_iter=3000))})
    rows.append({"method": "LR-l2 (log1p bits)", "F1":
        evaluate("LR-l2 (log1p bits)", Atr_log, ytr, Ate_log, y_te,
                 LogisticRegression(C=1.0, max_iter=3000))})
    rows.append({"method": "LinearSVC (log1p)", "F1":
        evaluate("LinearSVC (log1p)", Atr_log, ytr, Ate_log, y_te,
                 LinearSVC(C=1.0, max_iter=5000, dual="auto"))})
    rows.append({"method": "RF-500 (raw bits)", "F1":
        evaluate("RF-500 (raw bits)", Atr, ytr, A_te, y_te,
                 RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                        random_state=0, class_weight="balanced"))})

    # Top-1 argmax as a "Foldseek-only-classifier" sanity check
    pred_top1 = np.array(train_ids)[A_te.argmax(axis=1)]
    train_label_map = dict(zip(train_ids, y_tr))
    pred_label = np.array([train_label_map[e] for e in pred_top1])
    f1_top1 = f1_score(y_te, pred_label, average="weighted", zero_division=0)
    print(f"  Sanity argmax-on-affinity:                              F1 = {f1_top1:.4f}")
    rows.append({"method": "argmax-on-affinity (sanity)", "F1": f1_top1})

    print("\n=== Affinity + ESM2-35M concat ===")
    # L2-normalize the ESM2 part to keep scales comparable
    E_tr_n = normalize(E_tr)
    E_te_n = normalize(E_te)
    X_tr_cat = np.hstack([Atr_log, E_tr_n])
    X_te_cat = np.hstack([Ate_log, E_te_n])
    rows.append({"method": "LR-l2 (log1p bits + ESM2-35M)", "F1":
        evaluate("LR-l2 (log1p bits + ESM2-35M)", X_tr_cat, ytr, X_te_cat, y_te,
                 LogisticRegression(C=1.0, max_iter=3000))})
    rows.append({"method": "LinearSVC (log1p bits + ESM2-35M)", "F1":
        evaluate("LinearSVC (log1p bits + ESM2-35M)", X_tr_cat, ytr, X_te_cat, y_te,
                 LinearSVC(C=1.0, max_iter=5000, dual="auto"))})

    # Save matrices for downstream experiments
    np.savez(FEAT_DIR / "affinity_matrix.npz",
             A_train=Atr, y_train=ytr, train_ids=np.array(train_ids)[has_any_hit],
             A_test=A_te,  y_test=y_te,  test_ids=np.array(test_ids),
             train_ids_all=np.array(train_ids))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "phase8_affinity_results.csv", index=False)
    best = df.sort_values("F1", ascending=False).iloc[0]
    print(f"\nBest: {best['method']}  F1 = {best['F1']:.4f}")

    # Final chart
    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M ensemble": 0.2520,
        f"Affinity best\n({best['method']})": best["F1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Convergent Enzymes — Phase 8 (Foldseek-affinity classifier)")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=12, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase8_final.png", dpi=140)
    plt.close()

    txt = ["Phase 8 — Foldseek-affinity classifier", "=" * 50, ""]
    for r in df.sort_values("F1", ascending=False).iterrows():
        txt.append(f"  {r[1]['method']:55s} F1 = {r[1]['F1']:.4f}")
    txt.append("")
    txt.append(f"Best: {best['method']}  F1 = {best['F1']:.4f}")
    for name, v in methods.items():
        txt.append(f"  {name.replace(chr(10),' '):55s} F1 = {v:.4f}")
    (RESULTS_DIR / "phase8_summary.txt").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
