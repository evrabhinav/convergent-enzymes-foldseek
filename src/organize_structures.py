"""
Organize downloaded AlphaFold PDBs into per-split subdirectories.

Phase 1 downloads every structure into structures/<UniProtID>.pdb. Foldseek
and fpocket both want the train and test sets as separate directories, so
this script hard-links (falling back to copy) each PDB into
structures/train/ and structures/test/ according to data/train.csv and
data/test.csv.

Hard links are used so this costs ~no extra disk. Run this once after
phase1_load_and_download.py and before the WSL Foldseek / fpocket steps.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STRUCT_DIR = ROOT / "structures"


def link_split(df: pd.DataFrame, sub: str) -> int:
    out_dir = STRUCT_DIR / sub
    out_dir.mkdir(exist_ok=True)
    n = 0
    for entry in df["Entry"]:
        src = STRUCT_DIR / f"{entry}.pdb"
        dst = out_dir / f"{entry}.pdb"
        if src.exists() and not dst.exists():
            try:
                os.link(src, dst)          # hard link on NTFS — no extra disk
            except OSError:
                shutil.copy(src, dst)      # fallback if hard-linking fails
            n += 1
    return n


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    n_train = link_split(train_df, "train")
    n_test = link_split(test_df, "test")
    have_train = len(list((STRUCT_DIR / "train").glob("*.pdb")))
    have_test = len(list((STRUCT_DIR / "test").glob("*.pdb")))
    print(f"linked {n_train} new train PDBs, {n_test} new test PDBs")
    print(f"structures/train: {have_train} files")
    print(f"structures/test : {have_test} files")


if __name__ == "__main__":
    main()
