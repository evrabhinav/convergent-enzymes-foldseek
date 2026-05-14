"""
Phase 1: Load dataset and download AlphaFold structures.

Downloads AlphaFold predicted structures for every UniProt ID in the
tattabio/convergent_enzymes dataset, saves them as PDB files, and reports
coverage. Idempotent: re-running skips files already on disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STRUCT_DIR = ROOT / "structures"
DATA_DIR.mkdir(exist_ok=True)
STRUCT_DIR.mkdir(exist_ok=True)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN = 0.05  # courtesy delay


def load_dataset_to_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load HuggingFace dataset, write train/test CSVs, return DataFrames."""
    from datasets import load_dataset

    ds = load_dataset("tattabio/convergent_enzymes")
    print(f"Splits: {list(ds.keys())}")
    for split in ds.keys():
        print(f"  {split}: {len(ds[split])} rows, columns={ds[split].column_names}")

    # The dataset commonly uses split names "train" and "test"
    train_df = ds["train"].to_pandas() if "train" in ds else None
    test_df = ds["test"].to_pandas() if "test" in ds else None
    if train_df is None or test_df is None:
        # fall back: take first split as train and second as test
        keys = list(ds.keys())
        train_df = ds[keys[0]].to_pandas()
        test_df = ds[keys[1]].to_pandas()

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"Train EC classes: {train_df['Label'].nunique()}, Test EC classes: {test_df['Label'].nunique()}")
    return train_df, test_df


def fetch_alphafold_pdb(uniprot_id: str, out_path: Path) -> str:
    """Download AlphaFold predicted PDB for a UniProt ID.

    Returns: "ok" | "exists" | "no_entry" | "no_pdb_url" | f"error:{msg}"
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return "exists"
    try:
        r = requests.get(ALPHAFOLD_API.format(uniprot_id=uniprot_id), timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return f"error:request:{e.__class__.__name__}"
    if r.status_code == 404:
        return "no_entry"
    if r.status_code != 200:
        return f"error:http_{r.status_code}"
    try:
        meta = r.json()
    except json.JSONDecodeError:
        return "error:bad_json"
    if not meta or not isinstance(meta, list):
        return "no_entry"
    pdb_url = meta[0].get("pdbUrl")
    if not pdb_url:
        return "no_pdb_url"
    try:
        pdb_resp = requests.get(pdb_url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return f"error:pdb_request:{e.__class__.__name__}"
    if pdb_resp.status_code != 200:
        return f"error:pdb_http_{pdb_resp.status_code}"
    out_path.write_bytes(pdb_resp.content)
    return "ok"


def download_all(ids: list[str], split_label: str, workers: int = 16) -> pd.DataFrame:
    """Download PDBs concurrently for every uniprot id; return a status DataFrame."""
    def _one(uid: str) -> dict:
        out_path = STRUCT_DIR / f"{uid}.pdb"
        status = fetch_alphafold_pdb(uid, out_path)
        return {"Entry": uid, "split": split_label, "status": status,
                "path": str(out_path) if status in ("ok", "exists") else ""}

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, uid): uid for uid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"AF download [{split_label}]"):
            rows.append(fut.result())
    return pd.DataFrame(rows)


def summarize(status_df: pd.DataFrame) -> None:
    print("\n=== AlphaFold coverage ===")
    for split, sub in status_df.groupby("split"):
        total = len(sub)
        have = (sub["status"].isin(["ok", "exists"])).sum()
        print(f"  {split}: {have}/{total} structures available ({100*have/total:.1f}%)")
        print(sub["status"].value_counts().to_string())


def main(limit: int | None = None) -> None:
    train_df, test_df = load_dataset_to_csv()
    train_ids = train_df["Entry"].tolist()
    test_ids = test_df["Entry"].tolist()
    if limit:
        train_ids = train_ids[:limit]
        test_ids = test_ids[:limit]
        print(f"[LIMIT MODE] only first {limit} of each split")

    status_train = download_all(train_ids, "train")
    status_test = download_all(test_ids, "test")
    status = pd.concat([status_train, status_test], ignore_index=True)
    status.to_csv(DATA_DIR / "structure_status.csv", index=False)
    summarize(status)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
