"""
Bootstrap 95% CIs for the headline weighted-F1 numbers.

Reproduces the exact best-config predictions from phase13_crossover.py (CE)
and generic_crossover.py (EC), checks the point estimate matches the
reported value, then bootstraps over the test queries (resample with
replacement, B times) for a percentile 95% CI. Also reports the fraction
of resamples in which the ensemble F1 exceeds the DGEB ESM2-3B baseline.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA, FEAT, FS = ROOT / "data", ROOT / "features", ROOT / "foldseek_workdir"
B = 10000
RNG = np.random.default_rng(0)


def lr_predict(Xtr, ytr, Xte, C=1.0):
    sc = StandardScaler()
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(sc.fit_transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def foldseek_top1(hits_path, train_csv):
    df = pd.read_csv(hits_path, sep="\t",
                     names=["q", "target", "bits", "evalue", "prob",
                            "alntmscore", "fident", "lddt"])
    top = df.sort_values("bits", ascending=False).groupby("q").head(1)
    tl = dict(zip(*[pd.read_csv(train_csv)[c] for c in ("Entry", "Label")]))
    pred = {q: tl[t] for q, t in zip(top["q"], top["target"]) if t in tl}
    prob = dict(zip(top["q"], top["prob"]))
    return pred, prob


def align(target_e, src_e, X):
    idx = {str(e): i for i, e in enumerate(src_e)}
    return X[[idx[str(e)] for e in target_e]]


def wf1(yt, yp):
    return f1_score(yt, yp, average="weighted", zero_division=0)


def bca_ci(boots, point, jack, alpha=0.05):
    """Bias-corrected and accelerated 95% CI."""
    # bias-correction z0 from fraction of replicates below the point estimate
    prop = np.mean(boots < point)
    prop = min(max(prop, 1.0 / len(boots)), 1.0 - 1.0 / len(boots))
    z0 = norm.ppf(prop)
    # acceleration from jackknife skew
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    a1 = norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    return np.percentile(boots, [100 * a1, 100 * a2]), z0, a


def bootstrap(yte, preds, baseline, label, reported):
    yte = np.asarray(yte, dtype=object)
    preds = np.asarray(preds, dtype=object)
    point = wf1(yte, preds)
    n = len(yte)
    boots = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        boots[b] = wf1(yte[idx], preds[idx])
    # jackknife for BCa acceleration
    jack = np.empty(n)
    allidx = np.arange(n)
    for i in range(n):
        m = allidx != i
        jack[i] = wf1(yte[m], preds[m])
    plo, phi = np.percentile(boots, [2.5, 97.5])
    (blo, bhi), z0, a = bca_ci(boots, point, jack)
    frac_above = float(np.mean(boots > baseline))
    print(f"\n=== {label} (n={n}) ===")
    print(f"  reported F1          : {reported}")
    print(f"  reconstructed F1     : {point:.4f}   "
          f"{'OK MATCH' if abs(point - reported) < 0.0015 else '*** MISMATCH ***'}")
    print(f"  bootstrap mean       : {boots.mean():.4f}  (z0={z0:.3f}, a={a:.3f})")
    print(f"  95% CI (percentile)  : [{plo:.3f}, {phi:.3f}]")
    print(f"  95% CI (BCa)         : [{blo:.3f}, {bhi:.3f}]")
    print(f"  DGEB baseline        : {baseline}")
    print(f"  P(F1 > baseline)     : {frac_above:.3f}")
    print(f"  baseline inside BCa? : {blo <= baseline <= bhi}")
    return point, blo, bhi, frac_above


# ---------------- Convergent Enzymes: FS(prob>=0.9) > MAJ(3B,ProstT5,150M) ----
def ce():
    d3 = np.load(FEAT / "esm2_3b_matrix.npz", allow_pickle=True)
    Xa, ytr, ent_tr = d3["X_train"], d3["y_train"], d3["entries_train"]
    Ta, yte, ent_te = d3["X_test"], d3["y_test"], d3["entries_test"]
    d150 = np.load(FEAT / "esm2_t30_150M_matrix.npz", allow_pickle=True)
    Xb = align(ent_tr, d150["entries_train"], d150["X_train"])
    Tb = align(ent_te, d150["entries_test"], d150["X_test"])
    dp = np.load(FEAT / "prostT5_aa_matrix.npz", allow_pickle=True)
    Xc = align(ent_tr, dp["entries_train"], dp["X_train"])
    Tc = align(ent_te, dp["entries_test"], dp["X_test"])

    map_3b = dict(zip(map(str, ent_te), lr_predict(Xa, ytr, Ta)))
    map_p = dict(zip(map(str, ent_te), lr_predict(Xc, ytr, Tc)))
    map_e = dict(zip(map(str, ent_te), lr_predict(Xb, ytr, Tb)))
    fs_pred, fs_prob = foldseek_top1(FS / "hits.tsv", DATA / "train.csv")

    thr = 0.9
    preds = []
    for e in map(str, ent_te):
        if e in fs_pred and fs_prob.get(e, 0) >= thr:
            preds.append(fs_pred[e])
        else:
            votes = [m[e] for m in (map_3b, map_p, map_e) if e in m]
            preds.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
    return bootstrap(yte, preds, 0.265, "Convergent Enzymes  FS(>=0.9)>MAJ(3B,ProstT5,150M)", 0.267)


# ---------------- EC: FS(prob>=0.3) > MAJ(3B,35M,150M,ProstT5) ----------------
def ec():
    feat = FEAT / "ec_classification"
    d3 = np.load(feat / "esm2_t36_3B_matrix.npz", allow_pickle=True)
    ytr, ent_tr = d3["y_train"], d3["entries_train"]
    yte, ent_te = d3["y_test"], d3["entries_test"]
    order = ["esm2_t36_3B", "esm2_t12_35M", "esm2_t30_150M", "prostT5_aa"]
    maps = {}
    for name in order:
        d = np.load(feat / f"{name}_matrix.npz", allow_pickle=True)
        Xtr = align(ent_tr, d["entries_train"], d["X_train"])
        Xte = align(ent_te, d["entries_test"], d["X_test"])
        maps[name] = dict(zip(map(str, ent_te), lr_predict(Xtr, ytr, Xte)))
    fs_pred, fs_prob = foldseek_top1(FS / "ec_classification" / "hits.tsv",
                                     DATA / "ec_classification" / "train.csv")
    thr = 0.3
    preds = []
    for e in map(str, ent_te):
        if e in fs_pred and fs_prob.get(e, 0) >= thr:
            preds.append(fs_pred[e])
        else:
            votes = [maps[k][e] for k in order if e in maps[k]]
            preds.append(Counter(votes).most_common(1)[0][0] if votes else "__none__")
    return bootstrap(yte, preds, 0.680, "EC Classification   FS(>=0.3)>MAJ(4 LMs)", 0.730)


if __name__ == "__main__":
    ce()
    ec()
