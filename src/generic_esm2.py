"""
Generic ESM2 embedder for any DGEB protein-classification task.

Reads:
  data/<task>/train.csv, test.csv          (cols: Entry, Label, Sequence)
Writes:
  features/<task>/<model_slug>_matrix.npz  (X_train, y_train, X_test, ...)

Same featurization logic as src/phase6_esm2.py but scoped to a task subfolder
so we can run the same pipeline against EC Classification, MIBIG, etc. without
overwriting Convergent Enzymes outputs.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"

MAX_LEN = 1024
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


def slugify(model_name: str) -> str:
    return model_name.split("/")[-1].replace("_UR50D", "")


def featurize(seqs: list[str], tokenizer, model, desc: str = "embed") -> np.ndarray:
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(tqdm(seqs, desc=desc)):
            s = "".join(c for c in str(s).upper() if c.isalpha())[:MAX_LEN]
            if not s:
                continue
            enc = tokenizer(s, return_tensors="pt", truncation=True,
                            max_length=MAX_LEN, add_special_tokens=True)
            o = model(**enc)
            h = o.last_hidden_state.squeeze(0)
            if h.shape[0] > 2:
                h = h[1:-1]
            out[i] = h.mean(dim=0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="task name; reads data/<task>/{train,test}.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id (e.g. facebook/esm2_t30_150M_UR50D)")
    args = ap.parse_args()

    task_data = DATA_DIR / args.task
    task_feat = FEAT_DIR / args.task
    task_feat.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(8)
    print(f"loading {args.model} (CPU, fp32) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(torch.float32)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ready in {time.time()-t0:.1f}s | hidden={model.config.hidden_size} "
          f"| params={n_params:.1f}M")

    train_df = pd.read_csv(task_data / "train.csv")
    test_df = pd.read_csv(task_data / "test.csv")
    print(f"  train: {len(train_df)} rows, test: {len(test_df)} rows")

    Xtr = featurize(train_df["Sequence"].tolist(), tok, model,
                    desc=f"{slugify(args.model)} train")
    Xte = featurize(test_df["Sequence"].tolist(), tok, model,
                    desc=f"{slugify(args.model)} test")
    out_path = task_feat / f"{slugify(args.model)}_matrix.npz"
    np.savez(out_path,
             X_train=Xtr, y_train=train_df["Label"].to_numpy(),
             entries_train=train_df["Entry"].to_numpy(),
             X_test=Xte, y_test=test_df["Label"].to_numpy(),
             entries_test=test_df["Entry"].to_numpy())
    print(f"saved {out_path}  Xtr={Xtr.shape} Xte={Xte.shape}")


if __name__ == "__main__":
    main()
