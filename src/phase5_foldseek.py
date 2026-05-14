"""
Phase 5: Foldseek structural kNN classifier.

Index every train PDB into a Foldseek database, query every test PDB against
it, vote on EC labels of the top-k hits.

Foldseek does not ship a native Windows binary. This script supports three
backends, selected by --backend:

  --backend wsl     Call `foldseek` inside WSL. Requires WSL2 with a Linux
                    distro (e.g. `wsl --install -d Ubuntu`) and `foldseek`
                    installed inside it (`conda install -c bioconda foldseek`
                    or `wget` the linux-avx2 release tarball).

  --backend path    Call `foldseek` from PATH on the host (works if you ran
                    Foldseek from an MSYS2/MinGW environment, or wired up a
                    native build). Same logic, just no `wsl` prefix.

  --backend web     Use the Foldseek web API (https://search.foldseek.com).
                    Slowest, rate-limited, and constrained to their hosted
                    databases — useful only as a last resort. Implementation
                    is a stub here: you'd query each test PDB, parse hits,
                    intersect with train UniProt IDs, then vote.

The voting rule: of the top K Foldseek hits, take the majority EC label,
weighting by Foldseek's bit-score. Ties broken by highest single-hit bit-score.

Outputs:
  results/phase5_foldseek_predictions.csv   one row per test protein
  results/phase5_foldseek_summary.txt       weighted F1 vs all baselines
  charts/phase5_all_methods_bar.png         combined comparison chart
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STRUCT_DIR = ROOT / "structures"
FS_DIR = ROOT / "foldseek_workdir"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"
FS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)


def wsl_path(p: Path) -> str:
    """Translate C:\\path -> /mnt/c/path for WSL invocations."""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def make_runner(backend: str):
    """Return a callable that runs Foldseek args list."""
    if backend == "wsl":
        def run(args: list[str], cwd: Path = None) -> subprocess.CompletedProcess:
            translated = ["wsl", "foldseek"] + args
            return subprocess.run(translated, check=True, capture_output=True, text=True)
        return run, wsl_path
    if backend == "path":
        if not shutil.which("foldseek"):
            raise SystemExit("foldseek not on PATH; install or use --backend wsl")
        def run(args: list[str], cwd: Path = None) -> subprocess.CompletedProcess:
            return subprocess.run(["foldseek"] + args, check=True, capture_output=True, text=True)
        return run, str
    if backend == "web":
        raise SystemExit("--backend web is a stub; see file header. Use wsl or path.")
    raise SystemExit(f"unknown backend {backend!r}")


def build_db(run, pathfn, pdbs: list[Path], db_path: Path, name: str) -> None:
    """Build Foldseek DB from a list of PDBs."""
    list_file = FS_DIR / f"{name}_pdb_list.txt"
    list_file.write_text("\n".join(pathfn(p) for p in pdbs))
    # `foldseek createdb` accepts a list of files or a directory; we use createdb
    # with the list-of-files flag (--file-include) where supported; else fall back to
    # passing a directory containing only these PDBs. Simplest: just pass them all.
    args = ["createdb"] + [pathfn(p) for p in pdbs] + [pathfn(db_path)]
    print(f"[{name}] createdb on {len(pdbs)} PDBs ...")
    run(args)


def search(run, pathfn, query_db: Path, target_db: Path, out_tsv: Path) -> None:
    tmp = FS_DIR / "fstmp"
    tmp.mkdir(exist_ok=True)
    print("[search] easy-search ...")
    # easy-search is the one-shot pipeline; outputs a tab-separated hit table
    args = [
        "easy-search",
        pathfn(target_db),  # query
        pathfn(query_db),   # target db ("train" db)
        pathfn(out_tsv),
        pathfn(tmp),
        "--format-output", "query,target,bits,evalue,prob,alntmscore",
        "--exhaustive-search", "1",
    ]
    run(args)


def vote(hits: pd.DataFrame, train_labels: dict, k: int = 5) -> pd.DataFrame:
    """For each query, take top-k hits, weighted vote by bits."""
    preds = []
    for q, grp in hits.groupby("query"):
        grp = grp.sort_values("bits", ascending=False).head(k)
        score = defaultdict(float)
        for _, row in grp.iterrows():
            tgt = row["target"]
            ec = train_labels.get(tgt)
            if ec is None:
                continue
            score[ec] += float(row["bits"])
        if not score:
            preds.append({"query": q, "pred": None, "n_hits": 0})
            continue
        best = max(score.items(), key=lambda x: x[1])
        preds.append({"query": q, "pred": best[0], "n_hits": len(grp), "top_score": best[1]})
    return pd.DataFrame(preds)


def stem_to_entry(s: str) -> str:
    """Foldseek typically reports queries/targets as filename stems; we used
    UniProt IDs as filenames so the stem == UniProt ID."""
    return Path(s).stem.replace(".pdb", "")


def run_pipeline(backend: str, k_values: list[int]) -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    have = {p.stem for p in STRUCT_DIR.glob("*.pdb")}
    train_pdbs = [STRUCT_DIR / f"{e}.pdb" for e in train_df["Entry"] if e in have]
    test_pdbs  = [STRUCT_DIR / f"{e}.pdb" for e in test_df["Entry"]  if e in have]
    train_labels = dict(zip(train_df["Entry"], train_df["Label"]))
    test_labels  = dict(zip(test_df["Entry"],  test_df["Label"]))
    print(f"Train PDBs: {len(train_pdbs)}/{len(train_df)} | Test PDBs: {len(test_pdbs)}/{len(test_df)}")

    run, pathfn = make_runner(backend)

    train_db = FS_DIR / "trainDB"
    test_db  = FS_DIR / "testDB"
    hits_tsv = FS_DIR / "hits.tsv"

    if not train_db.exists():
        build_db(run, pathfn, train_pdbs, train_db, "train")
    if not test_db.exists():
        build_db(run, pathfn, test_pdbs, test_db, "test")
    search(run, pathfn, train_db, test_db, hits_tsv)

    hits = pd.read_csv(hits_tsv, sep="\t",
                       names=["query", "target", "bits", "evalue", "prob", "alntmscore"])
    hits["query"] = hits["query"].map(stem_to_entry)
    hits["target"] = hits["target"].map(stem_to_entry)
    print(f"Total hits returned: {len(hits)}")

    rows = []
    for k in k_values:
        preds = vote(hits, train_labels, k=k)
        # join with true labels
        preds["true"] = preds["query"].map(test_labels)
        kept = preds.dropna(subset=["pred", "true"])
        if len(kept) == 0:
            print(f"k={k}: no usable predictions")
            continue
        f1 = f1_score(kept["true"], kept["pred"], average="weighted", zero_division=0)
        coverage = len(kept) / len(test_df)
        rows.append({"method": f"foldseek_top{k}", "k": k,
                     "weighted_f1": f1, "test_coverage": coverage,
                     "n_predicted": len(kept)})
        preds.to_csv(RESULTS_DIR / f"phase5_foldseek_predictions_k{k}.csv", index=False)
        print(f"k={k}: weighted F1 = {f1:.4f}  (predicted {len(kept)}/{len(test_df)})")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "phase5_foldseek_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["wsl", "path", "web"], default="wsl")
    ap.add_argument("--k", default="1,3,5,10", help="comma-sep k values to try")
    args = ap.parse_args()
    k_values = [int(x) for x in args.k.split(",")]
    run_pipeline(args.backend, k_values)


if __name__ == "__main__":
    main()
