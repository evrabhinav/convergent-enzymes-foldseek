"""
Phase 10: ProstT5 embeddings (bilingual AA + 3Di model from Heinzinger 2023).

ProstT5 is a T5-based protein LM with a 'bilingual' vocabulary — it was
trained jointly on amino-acid sequences and on 3Di sequences derived from
AlphaFold structures. For each protein you can run either mode by prefixing
the input with a sentinel token:

  <AA2fold>  + space-separated AA sequence    -> structure-aware seq embedding
  <fold2AA>  + space-separated 3Di sequence   -> sequence-aware struct embedding

For our laptop (CPU-only, 16 GB RAM), we run the encoder in fp16 to keep
memory under 4 GB. Per-protein forward is the heavy step — 4-10 s on a
modern laptop CPU for a 300-residue protein. For 2400 proteins, plan
~3-8 hours total per mode.

This script supports --mode aa | 3di | both. With --mode both we run BOTH
forward passes per protein and concatenate the mean-pooled embeddings
(2048-D output). It is roughly 2x the compute.

Outputs:
  features/prostT5_{mode}_matrix.npz   X_train, y_train, entries_train, ...
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
FS_DIR = ROOT / "foldseek_workdir"

MODEL_NAME = "Rostlab/ProstT5"
MAX_LEN = 1024


def read_3di_fasta(path: Path) -> dict[str, str]:
    out, cur, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur is not None:
                out[cur] = "".join(buf)
            cur = line[1:].split()[0]
            buf = []
        elif line:
            buf.append(line.strip())
    if cur is not None:
        out[cur] = "".join(buf)
    return out


def prepare_input(seq: str, mode: str) -> str:
    """ProstT5 takes:
       AA mode  : '<AA2fold>' + ' '.join(uppercase residues)
       3Di mode : '<fold2AA>' + ' '.join(lowercase 3di letters)
    Sequence is cleaned of non-letters; rare AAs (U,Z,O,B) replaced by X.
    """
    s = "".join(c for c in seq if c.isalpha())
    if mode == "aa":
        s = re.sub(r"[UZOB]", "X", s.upper())
        prefix = "<AA2fold>"
        spaced = " ".join(s)
    else:  # 3di
        s = s.lower()
        prefix = "<fold2AA>"
        spaced = " ".join(s)
    return prefix + " " + spaced


def featurize(seqs: list[str], mode: str, tokenizer, model,
              max_len: int = MAX_LEN) -> np.ndarray:
    hid = model.config.d_model
    out = np.zeros((len(seqs), hid), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(tqdm(seqs, desc=f"ProstT5-{mode}")):
            if not seq:
                continue
            text = prepare_input(seq, mode)
            enc = tokenizer(text, return_tensors="pt",
                            add_special_tokens=True,
                            truncation=True, max_length=max_len + 1)
            o = model(**enc)
            h = o.last_hidden_state.squeeze(0)
            # tokenizer's first token is the sentinel ('<AA2fold>' or '<fold2AA>')
            # and there's a closing eos token; drop both
            if h.shape[0] > 2:
                h = h[1:-1]
            out[i] = h.mean(dim=0).to(torch.float32).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="aa", choices=["aa", "3di", "both"])
    ap.add_argument("--limit", type=int, default=0,
                    help="limit train/test size for smoke testing (0 = all)")
    args = ap.parse_args()

    torch.set_num_threads(8)
    print(f"loading {MODEL_NAME} (CPU, fp16) ...")
    t0 = time.time()
    tok = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ready in {time.time()-t0:.1f}s | hidden={model.config.d_model} "
          f"| params={n_params:.2f}B | dtype=fp16")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    if args.limit:
        train_df = train_df.head(args.limit)
        test_df = test_df.head(args.limit)

    aa_train = train_df["Sequence"].tolist()
    aa_test = test_df["Sequence"].tolist()

    di_map_tr, di_map_te = {}, {}
    if args.mode in ("3di", "both"):
        di_map_tr = read_3di_fasta(FS_DIR / "train_3di.fasta")
        di_map_te = read_3di_fasta(FS_DIR / "test_3di.fasta")
    # build aligned 3Di sequence lists
    if args.mode == "3di":
        seq_train = [di_map_tr.get(e, "") for e in train_df["Entry"]]
        seq_test = [di_map_te.get(e, "") for e in test_df["Entry"]]
        Xtr = featurize(seq_train, "3di", tok, model)
        Xte = featurize(seq_test, "3di", tok, model)
    elif args.mode == "aa":
        Xtr = featurize(aa_train, "aa", tok, model)
        Xte = featurize(aa_test, "aa", tok, model)
    else:  # both
        di_train = [di_map_tr.get(e, "") for e in train_df["Entry"]]
        di_test = [di_map_te.get(e, "") for e in test_df["Entry"]]
        Xtr_aa = featurize(aa_train, "aa", tok, model)
        Xte_aa = featurize(aa_test, "aa", tok, model)
        Xtr_di = featurize(di_train, "3di", tok, model)
        Xte_di = featurize(di_test, "3di", tok, model)
        Xtr = np.hstack([Xtr_aa, Xtr_di])
        Xte = np.hstack([Xte_aa, Xte_di])

    out_path = FEAT_DIR / f"prostT5_{args.mode}_matrix.npz"
    np.savez(out_path,
             X_train=Xtr, y_train=train_df["Label"].to_numpy(),
             entries_train=train_df["Entry"].to_numpy(),
             X_test=Xte, y_test=test_df["Label"].to_numpy(),
             entries_test=test_df["Entry"].to_numpy())
    print(f"saved {out_path}  Xtr={Xtr.shape} Xte={Xte.shape}")


if __name__ == "__main__":
    main()
