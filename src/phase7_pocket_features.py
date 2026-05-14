"""
Phase 7: Pocket-based features via fpocket.

fpocket was run inside WSL on every train/test PDB. Each run produces a
`<name>_out/` directory next to the input PDB containing:
  - <name>_info.txt           summary of each detected pocket (numeric props)
  - pockets/pocket<N>_atm.pdb  atoms lining pocket N

This script:
  1. For each protein, parses up to TOP_K pockets from `_info.txt`.
  2. Extracts 19 numeric descriptors per pocket.
  3. For the top-1 pocket, also counts the amino-acid composition of the
     pocket-lining residues (20-D).
  4. Aggregates into a flat per-protein vector:
       - 3 pockets x 19 = 57 numeric pocket descriptors
       - 4 aggregate (n_pockets, best_score, best_volume, mean_score)
       - 20 AA composition of top-1 pocket residues
     Total ≈ 81 features per protein.

Outputs:
  features/pocket_feature_matrix.npz
  features/pocket_feature_columns.csv
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
FEAT_DIR.mkdir(exist_ok=True)
# WSL-side path containing the fpocket *_out/ directories
WSL_FP_TRAIN = "/root/fs_work/train"
WSL_FP_TEST = "/root/fs_work/test"
# These are accessed via the Windows mount; the train/test folders are real
# directories on the host
HOST_FP_TRAIN = ROOT / "structures" / "train"
HOST_FP_TEST = ROOT / "structures" / "test"

TOP_K = 3
NUMERIC_FIELDS = [
    "Score", "Druggability Score", "Number of Alpha Spheres",
    "Total SASA", "Polar SASA", "Apolar SASA",
    "Volume", "Mean local hydrophobic density",
    "Mean alpha sphere radius", "Mean alp. sph. solvent access",
    "Apolar alpha sphere proportion", "Hydrophobicity score",
    "Volume score", "Polarity score", "Charge score",
    "Proportion of polar atoms", "Alpha sphere density",
    "Cent. of mass - Alpha Sphere max dist", "Flexibility",
]
AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_info(info_path: Path) -> list[dict]:
    """Parse a `<name>_info.txt`. Returns list of dicts (one per pocket)."""
    text = info_path.read_text()
    pockets = []
    # Each pocket block: "Pocket N :\n\t<field>: \t<value>\n..."
    for block in re.split(r"\nPocket \d+ :\n", text):
        if not block.strip():
            continue
        d = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.strip().partition(":")
            k = k.strip(); v = v.strip()
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if d:
            pockets.append(d)
    return pockets


def aa_composition_of_pocket(atm_pdb: Path) -> np.ndarray:
    """Count distinct residues lining the pocket, return 20-D AA fractions."""
    seen = set()
    if not atm_pdb.exists():
        return np.zeros(20, dtype=np.float32)
    for line in atm_pdb.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip()
        chain = line[21:22].strip()
        resnum = line[22:26].strip()
        if resname in AA3_TO_1:
            seen.add((chain, resnum, AA3_TO_1[resname]))
    if not seen:
        return np.zeros(20, dtype=np.float32)
    counts = Counter(aa for _, _, aa in seen)
    total = sum(counts.values())
    return np.array([counts.get(a, 0) / total for a in AA20], dtype=np.float32)


def feature_vector(out_dir: Path) -> np.ndarray:
    """Build the per-protein pocket feature vector from a `*_out` directory."""
    base = out_dir.name.replace("_out", "")
    info_path = out_dir / f"{base}_info.txt"
    n_num = len(NUMERIC_FIELDS)
    vec_pockets = np.zeros((TOP_K, n_num), dtype=np.float32)
    n_pockets = 0
    best_score = 0.0
    best_volume = 0.0
    mean_score = 0.0
    aa_comp = np.zeros(20, dtype=np.float32)
    if info_path.exists():
        pockets = parse_info(info_path)
        n_pockets = len(pockets)
        if pockets:
            scores = [p.get("Score", 0.0) for p in pockets]
            best_score = max(scores)
            mean_score = float(np.mean(scores))
            best_volume = max(p.get("Volume", 0.0) for p in pockets)
            for i, p in enumerate(pockets[:TOP_K]):
                for j, f in enumerate(NUMERIC_FIELDS):
                    vec_pockets[i, j] = p.get(f, 0.0)
            atm_pdb = out_dir / "pockets" / "pocket0_atm.pdb"
            aa_comp = aa_composition_of_pocket(atm_pdb)
    return np.concatenate([
        vec_pockets.flatten(),
        np.array([n_pockets, best_score, best_volume, mean_score], dtype=np.float32),
        aa_comp,
    ])


def columns() -> list[tuple[str, str]]:
    cols = []
    for i in range(TOP_K):
        for f in NUMERIC_FIELDS:
            slug = re.sub(r"[^a-z0-9]+", "_", f.lower()).strip("_")
            cols.append((f"pk{i+1}_{slug}", "pocket_num"))
    cols += [("n_pockets", "pocket_agg"), ("best_score", "pocket_agg"),
             ("best_volume", "pocket_agg"), ("mean_score", "pocket_agg")]
    for a in AA20:
        cols.append((f"pk1_aa_{a}_frac", "pocket_aa"))
    return cols


def build_matrix(entries: list[str], base_dir: Path) -> tuple[np.ndarray, list[int]]:
    cols = columns()
    rows = np.zeros((len(entries), len(cols)), dtype=np.float32)
    n_missing = 0
    for i, e in enumerate(tqdm(entries)):
        out_dir = base_dir / f"{e}_out"
        if not out_dir.exists():
            n_missing += 1
            continue
        rows[i] = feature_vector(out_dir)
    return rows, n_missing


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("=== train pocket features ===")
    Xtr, miss_tr = build_matrix(train_df["Entry"].tolist(), HOST_FP_TRAIN)
    print(f"  shape: {Xtr.shape}  missing: {miss_tr}")
    print("=== test pocket features ===")
    Xte, miss_te = build_matrix(test_df["Entry"].tolist(), HOST_FP_TEST)
    print(f"  shape: {Xte.shape}  missing: {miss_te}")

    cols = columns()
    pd.DataFrame(cols, columns=["column", "group"]).to_csv(
        FEAT_DIR / "pocket_feature_columns.csv", index=False)
    np.savez(FEAT_DIR / "pocket_feature_matrix.npz",
             X_train=Xtr, y_train=train_df["Label"].to_numpy(),
             entries_train=train_df["Entry"].to_numpy(),
             X_test=Xte, y_test=test_df["Label"].to_numpy(),
             entries_test=test_df["Entry"].to_numpy())
    print(f"saved {FEAT_DIR / 'pocket_feature_matrix.npz'}")


if __name__ == "__main__":
    main()
