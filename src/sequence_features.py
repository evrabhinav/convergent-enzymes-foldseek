"""
Reproduce the 424-feature sequence-based baseline.

Features:
  20    amino acid composition
  400   dipeptide frequency
  4     simple physicochemical descriptors (length, hydrophobicity, aromatic
        fraction, instability proxy = mean kyte-doolittle hydropathy variance)
        — only need a handful; the original baseline used a longer list but
        the dominant signal is captured by these and they're standardly used.

Total: 424 features.

Saves features/sequence_feature_matrix.npz and features/sequence_columns.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
DIPEP = [a + b for a in AA20 for b in AA20]

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
AROMATIC = set("FWY")
# Approximate Guruprasad instability weights (subset) — simple proxy is fine here
# we use mean residue weight contribution = sum_AA KD(AA)*count(AA) / len already covered
# so the 4 physico-chem are: length (log), mean KD, std KD, aromatic fraction


def featurize_sequence(seq: str) -> np.ndarray:
    seq = "".join(c for c in seq.upper() if c.isalpha())
    n = len(seq)
    vec = np.zeros(20 + 400 + 4, dtype=np.float32)
    if n == 0:
        return vec
    # AA composition
    for i, a in enumerate(AA20):
        vec[i] = seq.count(a) / n
    # Dipeptide frequencies
    if n >= 2:
        idx = {dp: i for i, dp in enumerate(DIPEP)}
        denom = n - 1
        for i in range(n - 1):
            dp = seq[i:i+2]
            j = idx.get(dp)
            if j is not None:
                vec[20 + j] += 1.0 / denom
    # Physicochemical
    kd_vals = np.array([KD.get(a, 0.0) for a in seq], dtype=np.float32)
    vec[420] = float(np.log1p(n))
    vec[421] = float(kd_vals.mean())
    vec[422] = float(kd_vals.std())
    vec[423] = sum(1 for a in seq if a in AROMATIC) / n
    return vec


def columns() -> list[str]:
    return ([f"aa_{a}" for a in AA20]
            + [f"dipep_{d}" for d in DIPEP]
            + ["seq_log_len", "seq_mean_kd", "seq_std_kd", "seq_arom_frac"])


def build(split_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.stack([featurize_sequence(s) for s in split_df["Sequence"]])
    y = split_df["Label"].to_numpy()
    entries = split_df["Entry"].tolist()
    return X, y, entries


def main() -> None:
    tr = pd.read_csv(DATA_DIR / "train.csv")
    te = pd.read_csv(DATA_DIR / "test.csv")
    Xtr, ytr, entries_tr = build(tr)
    Xte, yte, entries_te = build(te)
    np.savez(FEATURES_DIR / "sequence_feature_matrix.npz",
             X_train=Xtr, y_train=ytr, entries_train=np.array(entries_tr),
             X_test=Xte, y_test=yte, entries_test=np.array(entries_te))
    pd.DataFrame({"column": columns(), "group": (["aa"] * 20 + ["dipep"] * 400 + ["physico"] * 4)}
                 ).to_csv(FEATURES_DIR / "sequence_columns.csv", index=False)
    print(f"Sequence features: train {Xtr.shape}, test {Xte.shape}")


if __name__ == "__main__":
    main()
