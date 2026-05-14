"""
Phase 6 (Tier B): ESM2-35M embeddings on CPU.

Loads facebook/esm2_t12_35M_UR50D (~140 MB), runs each protein through it,
mean-pools the residue embeddings over the sequence (excluding [CLS]/[EOS]),
and writes a 480-D vector per protein.

CPU-only. Threads pinned to the host's physical core count for stable speed.
Sequences are truncated to 1024 tokens (model max) and processed one at a
time to keep peak memory under ~1 GB.

Outputs:
  features/esm2_35m_matrix.npz   X_train (N,480), y_train, entries_train, ...
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
FEAT_DIR.mkdir(exist_ok=True)

MAX_LEN = 1024

# Default; override with --model
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"

# Friendly slug for the output filename
def slugify(model_name: str) -> str:
    # facebook/esm2_t12_35M_UR50D -> esm2_t12_35M
    base = model_name.split("/")[-1]
    return base.replace("_UR50D", "").replace("esm2_", "esm2_")


def featurize(seqs: list[str], tokenizer, model) -> np.ndarray:
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(tqdm(seqs, desc="ESM2-35M embed")):
            s = "".join(c for c in s.upper() if c.isalpha())[:MAX_LEN]
            if not s:
                continue
            enc = tokenizer(s, return_tensors="pt", add_special_tokens=True,
                            truncation=True, max_length=MAX_LEN)
            o = model(**enc)
            # last_hidden_state: (1, L, H). Drop [CLS] (pos 0) and [EOS] (pos -1).
            h = o.last_hidden_state.squeeze(0)
            if h.shape[0] > 2:
                h = h[1:-1]
            out[i] = h.mean(dim=0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id (e.g. facebook/esm2_t30_150M_UR50D)")
    ap.add_argument("--out", default=None,
                    help="output .npz path (defaults to features/{slug}_matrix.npz)")
    args = ap.parse_args()

    torch.set_num_threads(8)
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"loading {args.model} (CPU, fp32) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(torch.float32)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ready in {time.time()-t0:.1f}s | hidden_size={model.config.hidden_size} | params={n_params:.1f}M")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    Xtr = featurize(train_df["Sequence"].tolist(), tok, model)
    Xte = featurize(test_df["Sequence"].tolist(), tok, model)
    out_path = Path(args.out) if args.out else FEAT_DIR / f"{slugify(args.model)}_matrix.npz"
    np.savez(out_path,
             X_train=Xtr, y_train=train_df["Label"].to_numpy(),
             entries_train=train_df["Entry"].to_numpy(),
             X_test=Xte, y_test=test_df["Label"].to_numpy(),
             entries_test=test_df["Entry"].to_numpy())
    print(f"saved {out_path}  Xtr={Xtr.shape} Xte={Xte.shape}")


if __name__ == "__main__":
    main()
