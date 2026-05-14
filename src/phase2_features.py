"""
Phase 2: Extract structural features from AlphaFold PDB files.

Feature groups (and indices in the final vector):
  ss      (3)   : secondary structure percentages (H, E, L) — pydssp
  geom    (5)   : radius of gyration, mean B-factor (pLDDT), n_residues,
                  mean SASA per residue, total SASA — BioPython
  aa_ss   (60)  : amino acid composition stratified by SS element
                  (20 amino acids x 3 SS types, fraction of total residues)
  contact (12)  : long-range contact count, contact density,
                  short/medium/long sequence-separation bin fractions,
                  mean / median contact distance, etc.
  aa      (20)  : overall amino acid composition (used for ablation comparison)

A single protein returns a flat numpy vector. The pipeline writes:
  features/feature_matrix.npz   X_train, y_train, X_test, y_test, entries
  features/feature_columns.csv  column name + group label per feature

`pydssp` is a pure-Python DSSP-equivalent that does not require the mkdssp
binary, so this runs on Windows without conda or WSL.
"""
from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm

import pydssp

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STRUCT_DIR = ROOT / "structures"
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
SS_LABELS = ["H", "E", "L"]  # helix, strand, loop/coil (we normalize pydssp's '-' to 'L')

CONTACT_DIST = 8.0    # Å; CA-CA within this distance is a contact
LONG_RANGE_SEP = 12   # |i-j| >= this is "long-range"


def _ss_columns() -> list[str]:
    return [f"ss_{x}_frac" for x in SS_LABELS]

def _geom_columns() -> list[str]:
    return ["geom_radius_of_gyration", "geom_mean_bfactor",
            "geom_n_residues", "geom_mean_sasa", "geom_total_sasa"]

def _aa_ss_columns() -> list[str]:
    return [f"aass_{aa}_{ss}" for aa in AA20 for ss in SS_LABELS]

def _contact_columns() -> list[str]:
    return [
        "contact_density",            # contacts / n_residues
        "contact_lr_per_res",         # long-range contacts / n_residues
        "contact_mean_dist",
        "contact_median_dist",
        "contact_short_frac",         # 1 < |i-j| < 6
        "contact_medium_frac",        # 6 <= |i-j| < 12
        "contact_long_frac",          # |i-j| >= 12
        "contact_mean_seq_sep",
        "contact_max_seq_sep_per_n",
        "contact_cys_pairs_per_res",  # cys-cys CA <7Å pairs per residue
        "contact_clustering",         # avg per-res neighbour count / n_residues
        "contact_radius_per_n",       # max CA-CA dist / sqrt(n_residues)
    ]

def _aa_columns() -> list[str]:
    return [f"aa_{a}_frac" for a in AA20]


def all_feature_columns() -> list[tuple[str, str]]:
    """List of (column_name, group) pairs in the order they appear in the vector."""
    pairs = []
    pairs += [(c, "ss") for c in _ss_columns()]
    pairs += [(c, "geom") for c in _geom_columns()]
    pairs += [(c, "aa_ss") for c in _aa_ss_columns()]
    pairs += [(c, "contact") for c in _contact_columns()]
    pairs += [(c, "aa") for c in _aa_columns()]
    return pairs


def parse_structure(pdb_path: Path):
    """Return (residues, sequence_str, ca_coords (N,3), backbone_coords (N,4,3),
    bfactors (N,)). Skips non-standard residues and residues missing backbone atoms."""
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    residues, seq_chars, ca_coords, bb_coords, bfacs = [], [], [], [], []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            if resname not in THREE_TO_ONE:
                continue
            # need N, CA, C, O for pydssp
            needed = ["N", "CA", "C", "O"]
            if not all(a in res for a in needed):
                continue
            atoms = [res[a].get_coord() for a in needed]
            residues.append(res)
            seq_chars.append(THREE_TO_ONE[resname])
            ca_coords.append(res["CA"].get_coord())
            bb_coords.append(atoms)
            bfacs.append(res["CA"].get_bfactor())
    return (residues,
            "".join(seq_chars),
            np.array(ca_coords, dtype=np.float32) if ca_coords else np.zeros((0, 3), np.float32),
            np.array(bb_coords, dtype=np.float32) if bb_coords else np.zeros((0, 4, 3), np.float32),
            np.array(bfacs, dtype=np.float32) if bfacs else np.zeros(0, np.float32),
            structure)


def secondary_structure(backbone: np.ndarray) -> np.ndarray:
    """Return a numpy array of 'H','E','L' chars for each residue."""
    if backbone.shape[0] < 4:
        return np.array(["L"] * backbone.shape[0])
    try:
        ss = pydssp.assign(backbone.astype(np.float32), out_type="c3")
    except Exception:
        return np.array(["L"] * backbone.shape[0])
    # pydssp emits '-' for loop/coil; normalize to 'L' for our bookkeeping
    ss = np.where(ss == "-", "L", ss)
    return ss


def compute_sasa(structure) -> dict:
    """Return mean and total per-residue SASA over the first model."""
    try:
        sr = ShrakeRupley()
        sr.compute(structure, level="R")
    except Exception:
        return {"mean": 0.0, "total": 0.0}
    sasa_vals = []
    for res in structure.get_residues():
        if res.get_resname() not in THREE_TO_ONE:
            continue
        if hasattr(res, "sasa") and res.sasa is not None:
            sasa_vals.append(float(res.sasa))
    if not sasa_vals:
        return {"mean": 0.0, "total": 0.0}
    return {"mean": float(np.mean(sasa_vals)), "total": float(np.sum(sasa_vals))}


def feature_vector(pdb_path: Path) -> tuple[np.ndarray, dict]:
    """Compute the structural feature vector for one PDB file.

    Returns (vector, meta) where meta has diagnostic counts (n_residues, ss_counts).
    """
    cols = all_feature_columns()
    out = np.zeros(len(cols), dtype=np.float32)

    residues, seq, ca, bb, bfacs, structure = parse_structure(pdb_path)
    n = len(residues)
    meta = {"n_residues": n}
    if n == 0:
        return out, meta

    # --- 1. SS percentages
    ss = secondary_structure(bb)
    counts = Counter(ss.tolist())
    for i, lab in enumerate(SS_LABELS):
        out[i] = counts.get(lab, 0) / n

    # --- 2. Geometry
    centroid = ca.mean(axis=0)
    rg = float(np.sqrt(((ca - centroid) ** 2).sum(axis=1).mean()))
    mean_b = float(bfacs.mean())
    sasa = compute_sasa(structure)
    g0 = len(_ss_columns())
    out[g0 + 0] = rg
    out[g0 + 1] = mean_b
    out[g0 + 2] = n
    out[g0 + 3] = sasa["mean"]
    out[g0 + 4] = sasa["total"]

    # --- 3. AA composition by SS
    g1 = g0 + len(_geom_columns())
    aa_ss_counts = np.zeros((20, 3), dtype=np.float32)
    aa_idx = {a: i for i, a in enumerate(AA20)}
    ss_idx = {s: i for i, s in enumerate(SS_LABELS)}
    for a, s in zip(seq, ss):
        if a in aa_idx and s in ss_idx:
            aa_ss_counts[aa_idx[a], ss_idx[s]] += 1
    aa_ss_counts /= max(n, 1)
    out[g1 : g1 + 60] = aa_ss_counts.flatten()

    # --- 4. Contact features (CA-CA)
    g2 = g1 + len(_aa_ss_columns())
    dists = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    contact_mask = (dists < CONTACT_DIST) & (dists > 0)
    # iu = upper triangle indices for i<j
    iu = np.triu_indices(n, k=2)  # |i-j|>=2 to skip neighbors
    sep = iu[1] - iu[0]
    is_contact = contact_mask[iu]
    n_contact = int(is_contact.sum())
    long_range = int((is_contact & (sep >= LONG_RANGE_SEP)).sum())
    short = int((is_contact & (sep >= 2) & (sep < 6)).sum())
    medium = int((is_contact & (sep >= 6) & (sep < LONG_RANGE_SEP)).sum())
    long_ = int((is_contact & (sep >= LONG_RANGE_SEP)).sum())
    if n_contact > 0:
        contact_dists = dists[iu][is_contact]
        contact_seps = sep[is_contact]
        mean_dist = float(contact_dists.mean())
        median_dist = float(np.median(contact_dists))
        mean_seq_sep = float(contact_seps.mean())
        max_seq_sep = float(contact_seps.max())
    else:
        mean_dist = median_dist = mean_seq_sep = max_seq_sep = 0.0
    # cys-cys
    cys_idx = [i for i, a in enumerate(seq) if a == "C"]
    cys_pairs = 0
    if len(cys_idx) >= 2:
        for i in range(len(cys_idx)):
            for j in range(i + 1, len(cys_idx)):
                if dists[cys_idx[i], cys_idx[j]] < 7.0:
                    cys_pairs += 1
    # clustering: average neighbours per residue (excl self)
    neigh = int(contact_mask.sum())  # counts (i,j) and (j,i), excludes diag
    clustering = neigh / (n * max(n - 1, 1))
    max_dist = float(dists.max()) if n > 1 else 0.0
    sqrtn = math.sqrt(max(n, 1))

    out[g2 + 0] = n_contact / n
    out[g2 + 1] = long_range / n
    out[g2 + 2] = mean_dist
    out[g2 + 3] = median_dist
    out[g2 + 4] = short / max(n_contact, 1)
    out[g2 + 5] = medium / max(n_contact, 1)
    out[g2 + 6] = long_ / max(n_contact, 1)
    out[g2 + 7] = mean_seq_sep
    out[g2 + 8] = max_seq_sep / sqrtn
    out[g2 + 9] = cys_pairs / n
    out[g2 + 10] = clustering
    out[g2 + 11] = max_dist / sqrtn

    # --- 5. Overall AA composition (ablation)
    g3 = g2 + len(_contact_columns())
    aa_counter = Counter(seq)
    for i, a in enumerate(AA20):
        out[g3 + i] = aa_counter.get(a, 0) / n

    meta["ss_counts"] = dict(counts)
    return out, meta


def build_feature_matrix(split: str, ids_labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """ids_labels has columns Entry, Label. Returns X, y, kept_entries, metas."""
    cols = all_feature_columns()
    rows, ys, entries, metas = [], [], [], []
    for _, row in tqdm(ids_labels.iterrows(), total=len(ids_labels),
                       desc=f"features [{split}]"):
        uid = row["Entry"]
        path = STRUCT_DIR / f"{uid}.pdb"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            vec, meta = feature_vector(path)
        except Exception as e:
            meta = {"error": str(e)}
            print(f"[warn] {uid}: {e}")
            continue
        rows.append(vec)
        ys.append(row["Label"])
        entries.append(uid)
        metas.append({"Entry": uid, **meta})
    X = np.stack(rows) if rows else np.zeros((0, len(cols)), np.float32)
    return X, np.array(ys), entries, metas


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    Xtr, ytr, entries_tr, meta_tr = build_feature_matrix("train", train_df[["Entry", "Label"]])
    Xte, yte, entries_te, meta_te = build_feature_matrix("test", test_df[["Entry", "Label"]])

    cols = all_feature_columns()
    col_df = pd.DataFrame(cols, columns=["column", "group"])
    col_df.to_csv(FEATURES_DIR / "feature_columns.csv", index=False)

    np.savez(FEATURES_DIR / "feature_matrix.npz",
             X_train=Xtr, y_train=ytr, entries_train=np.array(entries_tr),
             X_test=Xte, y_test=yte, entries_test=np.array(entries_te))
    pd.DataFrame(meta_tr).to_csv(FEATURES_DIR / "meta_train.csv", index=False)
    pd.DataFrame(meta_te).to_csv(FEATURES_DIR / "meta_test.csv", index=False)

    print(f"\nTrain feature matrix: {Xtr.shape}  ({len(set(ytr))} EC classes)")
    print(f"Test  feature matrix: {Xte.shape}  ({len(set(yte))} EC classes)")
    print(f"  feature groups: {sorted({g for _, g in cols})}")
    print(f"  saved to {FEATURES_DIR / 'feature_matrix.npz'}")


if __name__ == "__main__":
    main()
