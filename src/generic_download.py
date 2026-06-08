"""
Generic AlphaFold structure downloader for any DGEB protein-classification
dataset (or any HuggingFace dataset that exposes an `Entry` column of
UniProt IDs).

Same logic as src/phase1_load_and_download.py, but the dataset name and the
output paths are configurable so we can reuse the pipeline on EC Classification,
MIBIG Classification, etc. without copy-pasting code.

Outputs:
  data/<task>/train.csv, test.csv
  data/<task>/structure_status.csv
  structures/<task>/<UniProtID>.pdb
"""
from __future__ import annotations

import argparse
import json
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

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
REQUEST_TIMEOUT = 30


def load_hf(dataset_name: str, task_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset
    task_data_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_name)
    keys = list(ds.keys())
    print(f"Splits: {keys}")
    train_key = "train" if "train" in ds else keys[0]
    test_key = "test" if "test" in ds else keys[1]
    train_df = ds[train_key].to_pandas()
    test_df = ds[test_key].to_pandas()
    # If the dataset doesn't have a "Label" column but has something else,
    # normalize it for downstream code (we always read 'Label').
    if "Label" not in train_df.columns:
        for candidate in ("label", "class", "simple_class", "category"):
            if candidate in train_df.columns:
                train_df = train_df.rename(columns={candidate: "Label"})
                test_df = test_df.rename(columns={candidate: "Label"})
                break
    train_df.to_csv(task_data_dir / "train.csv", index=False)
    test_df.to_csv(task_data_dir / "test.csv", index=False)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    if "Label" in train_df.columns:
        print(f"Train classes: {train_df['Label'].nunique()}, "
              f"Test classes: {test_df['Label'].nunique()}")
    return train_df, test_df


def fetch_alphafold_pdb(uniprot_id: str, out_path: Path) -> str:
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


def download_all(ids: list[str], split_label: str, struct_dir: Path,
                 workers: int = 16) -> pd.DataFrame:
    def _one(uid: str) -> dict:
        out_path = struct_dir / f"{uid}.pdb"
        status = fetch_alphafold_pdb(uid, out_path)
        return {"Entry": uid, "split": split_label, "status": status,
                "path": str(out_path) if status in ("ok", "exists") else ""}
    rows = []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="HF dataset id, e.g. tattabio/ec_classification")
    ap.add_argument("--task", required=True,
                    help="short task name used as the subfolder name")
    ap.add_argument("--limit", type=int, default=0,
                    help="optional smoke-test limit on rows per split")
    args = ap.parse_args()

    task_data = DATA_DIR / args.task
    task_struct = STRUCT_DIR / args.task
    task_struct.mkdir(parents=True, exist_ok=True)
    print(f"[task={args.task}] data -> {task_data}, structures -> {task_struct}")

    train_df, test_df = load_hf(args.dataset, task_data)
    train_ids = train_df["Entry"].tolist()
    test_ids = test_df["Entry"].tolist()
    if args.limit:
        train_ids = train_ids[: args.limit]
        test_ids = test_ids[: args.limit]
        print(f"[LIMIT] only first {args.limit} of each split")

    status_train = download_all(train_ids, "train", task_struct)
    status_test = download_all(test_ids, "test", task_struct)
    status = pd.concat([status_train, status_test], ignore_index=True)
    status.to_csv(task_data / "structure_status.csv", index=False)
    summarize(status)


if __name__ == "__main__":
    main()
