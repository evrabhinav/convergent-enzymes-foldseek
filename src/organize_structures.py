"""
Organize downloaded AlphaFold PDBs into per-split subdirectories.

Phase 1 downloads every structure into structures/<task>/<UniProtID>.pdb.
Foldseek and fpocket both want the train and test sets as separate
directories, so this script hard-links (falling back to copy) each PDB into
structures/<task>/train/ and structures/<task>/test/ according to
data/<task>/train.csv and data/<task>/test.csv.

Hard links are used so this costs ~no extra disk. Run this once after the
structure downloader and before the WSL Foldseek / fpocket steps.

Usage:
  python src/organize_structures.py                           # convergent_enzymes (default)
  python src/organize_structures.py --task ec_classification
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def link_split(df: pd.DataFrame, sub: str, struct_dir: Path) -> int:
    out_dir = struct_dir / sub
    out_dir.mkdir(exist_ok=True)
    n = 0
    for entry in df["Entry"]:
        src = struct_dir / f"{entry}.pdb"
        dst = out_dir / f"{entry}.pdb"
        if src.exists() and not dst.exists():
            try:
                os.link(src, dst)          # hard link on NTFS - no extra disk
            except OSError:
                shutil.copy(src, dst)      # fallback if hard-linking fails
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    help="task subfolder. If omitted, uses the original "
                         "convergent-enzymes layout (data/, structures/).")
    args = ap.parse_args()

    if args.task:
        data_dir = ROOT / "data" / args.task
        struct_dir = ROOT / "structures" / args.task
    else:
        data_dir = ROOT / "data"
        struct_dir = ROOT / "structures"
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    n_train = link_split(train_df, "train", struct_dir)
    n_test = link_split(test_df, "test", struct_dir)
    have_train = len(list((struct_dir / "train").glob("*.pdb")))
    have_test = len(list((struct_dir / "test").glob("*.pdb")))
    print(f"linked {n_train} new train PDBs, {n_test} new test PDBs")
    print(f"{struct_dir}/train: {have_train} files")
    print(f"{struct_dir}/test : {have_test} files")


if __name__ == "__main__":
    main()
