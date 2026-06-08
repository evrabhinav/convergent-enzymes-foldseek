"""
Generic crossover ensemble — runs the Phase 13 recipe on any task.

For task <task>:
  features/<task>/esm2_t12_35M_matrix.npz   (optional)
  features/<task>/esm2_t30_150M_matrix.npz
  features/<task>/esm2_t36_3B_matrix.npz
  features/<task>/prostT5_aa_matrix.npz
  foldseek_workdir/<task>/hits.tsv

Outputs:
  results/<task>/crossover_results.csv
  results/<task>/summary.txt
  charts/<task>_crossover.png
"""
from __future__ import annotations

import argparse
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
FS_DIR = ROOT / "foldseek_workdir"


def load_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    return (d["X_train"], d["y_train"], d["entries_train"],
            d["X_test"], d["y_test"], d["entries_test"])


def align(target_e, src_e, X):
    idx = {str(e): i for i, e in enumerate(src_e)}
    return X[[idx[str(e)] for e in target_e]]


def load_foldseek_top1(task: str):
    df = pd.read_csv(FS_DIR / task / "hits.tsv", sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = df.sort_values("bits", ascending=False).groupby("q").head(1)
    train_df = pd.read_csv(DATA_DIR / task / "train.csv")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--baseline-f1", type=float, default=None,
                    help="DGEB AA-best reference F1 for this task")
    args = ap.parse_args()

    task = args.task
    feat = FEAT_DIR / task
    (RESULTS_DIR / task).mkdir(parents=True, exist_ok=True)

    # Base alignment from ESM2-3B if present, otherwise from ESM2-35M
    if (feat / "esm2_t36_3B_matrix.npz").exists():
        base_name = "esm2_t36_3B_matrix.npz"
    else:
        base_name = "esm2_t12_35M_matrix.npz"
    X3b, ytr, ent_tr, T3b, yte, ent_te = load_npz(feat / base_name)
    print(f"[{task}] base ({base_name}): train {X3b.shape}, test {T3b.shape}")

    models = {base_name.replace("_matrix.npz", ""): (X3b, T3b)}
    for f in ["esm2_t12_35M_matrix.npz", "esm2_t30_150M_matrix.npz",
              "prostT5_aa_matrix.npz"]:
        p = feat / f
        if p.exists() and f != base_name:
            d = np.load(p, allow_pickle=True)
            Xtr = align(ent_tr, d["entries_train"], d["X_train"])
            Xte = align(ent_te, d["entries_test"], d["X_test"])
            models[f.replace("_matrix.npz", "")] = (Xtr, Xte)
            print(f"[{task}] loaded {f}: train {Xtr.shape}")

    fs_pred, fs_prob = load_foldseek_top1(task)
    rows = []

    print(f"\n=== {task}: per-model F1 (LogReg) ===")
    preds_per_model = {}
    for name, (Xtr, Xte) in models.items():
        p = lr_predict(Xtr, ytr, Xte)
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        preds_per_model[name] = {str(e): pp for e, pp in zip(ent_te, p)}
        print(f"  {name:25s} F1 = {f1:.4f}")
        rows.append({"method": f"{name} alone (LR)", "weighted_f1": f1})

    print(f"\n=== {task}: Foldseek + single-model fallback ===")
    for name in models:
        for thr in [0.3, 0.5, 0.7, 0.9]:
            p = [(fs_pred[str(e)] if (str(e) in fs_pred and fs_prob.get(str(e), 0) >= thr)
                  else preds_per_model[name].get(str(e), "__none__")) for e in ent_te]
            f1 = f1_score(yte, p, average="weighted", zero_division=0)
            rows.append({"method": f"FS(prob>={thr}) > {name}", "weighted_f1": f1})

    print(f"\n=== {task}: Foldseek + majority(all LMs) fallback ===")
    keys = [k for k in models]
    for thr in [0.3, 0.5, 0.7, 0.9]:
        p = []
        for e in ent_te:
            e = str(e)
            if e in fs_pred and fs_prob.get(e, 0) >= thr:
                p.append(fs_pred[e])
            else:
                votes = [preds_per_model[k][e] for k in keys if e in preds_per_model[k]]
                p.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
        f1 = f1_score(yte, p, average="weighted", zero_division=0)
        rows.append({"method": f"FS(prob>={thr}) > MAJ({len(keys)} LMs)", "weighted_f1": f1})

    # Foldseek top-1 alone
    p_fs = [fs_pred.get(str(e), "__none__") for e in ent_te]
    f1_fs = f1_score(yte, p_fs, average="weighted", zero_division=0)
    rows.insert(0, {"method": "Foldseek top-1 alone", "weighted_f1": f1_fs})

    df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    df.to_csv(RESULTS_DIR / task / "crossover_results.csv", index=False)
    best = df.iloc[0]
    print(f"\n=== OVERALL BEST [{task}]: {best['method']}  F1 = {best['weighted_f1']:.4f} ===")
    if args.baseline_f1 is not None:
        crossed = best["weighted_f1"] > args.baseline_f1
        diff = best["weighted_f1"] - args.baseline_f1
        print(f"   DGEB baseline: {args.baseline_f1}  crossed: {crossed}  (delta {diff:+.4f})")

    # chart
    fig, ax = plt.subplots(figsize=(12, max(4, 0.3 * len(df))))
    df_sorted = df.sort_values("weighted_f1")
    bars = ax.barh(df_sorted["method"], df_sorted["weighted_f1"], color="#4c72b0")
    for b, v in zip(bars, df_sorted["weighted_f1"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    if args.baseline_f1:
        ax.axvline(args.baseline_f1, ls="--", color="darkred", alpha=0.7,
                   label=f"DGEB baseline = {args.baseline_f1}")
        ax.legend()
    ax.set_xlabel("weighted F1")
    ax.set_title(f"{task} — crossover ensemble")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{task}_crossover.png", dpi=140)
    plt.close()

    # summary text
    txt = [f"{task} crossover summary", "=" * 50, ""]
    if args.baseline_f1:
        txt.append(f"DGEB AA-best reference: F1 = {args.baseline_f1}")
    txt.append(f"BEST: {best['method']}  F1 = {best['weighted_f1']:.4f}")
    txt.append("")
    txt.append("Top 5:")
    for _, r in df.head(5).iterrows():
        txt.append(f"  {r['method']:55s} F1 = {r['weighted_f1']:.4f}")
    (RESULTS_DIR / task / "summary.txt").write_text("\n".join(txt))
    print(f"\nsaved {CHARTS_DIR / f'{task}_crossover.png'}")


if __name__ == "__main__":
    main()
