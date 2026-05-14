# ESM2-3B embeddings on a free Colab T4 GPU
# ==========================================
# This is the EXACT cell run on Google Colab to produce
# features/esm2_3b_matrix.npz. ESM2-3B (2.8B params) takes ~157 s per protein
# on the project laptop's CPU (~4 days for the full dataset), but ~1-2 s per
# protein on a T4 GPU (~60-75 min total including the model download).
#
# HOW TO RUN:
#   1. Open https://colab.research.google.com  -> New notebook
#   2. Runtime -> Change runtime type -> T4 GPU -> Save
#   3. Paste this whole file into a single cell and run it (Shift+Enter)
#   4. When it finishes, the browser downloads esm2_3b_matrix.npz (~24 MB)
#   5. Move that file into this repo's features/ directory
#
# The CPU-side equivalent is src/phase6_esm2.py (feasible only for the smaller
# ESM2-35M / 150M models).

# !pip install -q --upgrade transformers datasets

import re, time, numpy as np, torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

MODEL = "facebook/esm2_t36_3B_UR50D"
MAX_LEN = 1022  # matches the DGEB evaluation protocol

# 1) load model in fp16 on GPU
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
print("model loaded:", sum(p.numel() for p in model.parameters()) / 1e9, "B params")
print("hidden_size:", model.config.hidden_size)

# 2) load dataset directly from HuggingFace
ds = load_dataset("tattabio/convergent_enzymes")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
print("train:", len(train_df), "test:", len(test_df))


# 3) embed: mean-pool the last hidden state over residues
def featurize(seqs):
    hid = model.config.hidden_size
    out = np.zeros((len(seqs), hid), dtype=np.float32)
    for i, seq in enumerate(tqdm(seqs)):
        s = "".join(c for c in str(seq) if c.isalpha())[:MAX_LEN]
        if not s:
            continue
        enc = tok(s, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN, add_special_tokens=True).to("cuda")
        with torch.no_grad():
            o = model(**enc)
        h = o.last_hidden_state[0]
        if h.shape[0] > 2:        # drop [CLS] and [EOS]
            h = h[1:-1]
        out[i] = h.mean(0).float().cpu().numpy()
    return out


t0 = time.time()
Xtr = featurize(train_df["Sequence"].tolist())
Xte = featurize(test_df["Sequence"].tolist())
print(f"embedded in {(time.time() - t0) / 60:.1f} min")

# 4) save and download
np.savez("esm2_3b_matrix.npz",
         X_train=Xtr, y_train=train_df["Label"].to_numpy(),
         entries_train=train_df["Entry"].to_numpy(),
         X_test=Xte, y_test=test_df["Label"].to_numpy(),
         entries_test=test_df["Entry"].to_numpy())

from google.colab import files  # noqa: E402  (Colab-only import)
files.download("esm2_3b_matrix.npz")
