# ProstT5 embeddings on a free Colab T4 GPU
# ==========================================
# This is the EXACT cell run on Google Colab to produce
# features/prostT5_aa_matrix.npz. ProstT5 (1.2B params, T5 architecture)
# takes ~157 s per protein on the project laptop's CPU, but ~0.3 s per protein
# on a T4 GPU (~10 min total for all 2400 proteins, the model download adds
# a couple more minutes).
#
# HOW TO RUN:
#   1. Open https://colab.research.google.com  -> New notebook
#   2. Runtime -> Change runtime type -> T4 GPU -> Save
#   3. Paste this whole file into a single cell and run it (Shift+Enter)
#   4. When it finishes, the browser downloads prostT5_aa_matrix.npz (~10 MB)
#   5. Move that file into this repo's features/ directory
#
# The CPU-side equivalent is src/phase10_prostT5.py (impractically slow on CPU
# for the full dataset; kept for documentation).
#
# Note: ProstT5 needs the *slow* SentencePiece tokenizer. transformers' fast
# path tries to convert the vocab via tiktoken and fails on ProstT5's
# spiece.model, so we force use_fast=False and install sentencepiece.

# !pip install -q --upgrade transformers sentencepiece datasets

import re, time, numpy as np, torch
from tqdm.auto import tqdm
from transformers import T5EncoderModel, AutoTokenizer
from datasets import load_dataset

# 1) load model in fp16 on GPU (slow tokenizer required)
tok = AutoTokenizer.from_pretrained("Rostlab/ProstT5",
                                    do_lower_case=False, use_fast=False, legacy=True)
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5",
                                       torch_dtype=torch.float16).cuda().eval()
print("model loaded:", sum(p.numel() for p in model.parameters()) / 1e9, "B params")

# 2) load dataset directly from HuggingFace
ds = load_dataset("tattabio/convergent_enzymes")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
print("train:", len(train_df), "test:", len(test_df))


# 3) embed in AA mode: '<AA2fold>' prefix + space-separated residues,
#    then mean-pool the encoder's last hidden state over residues
def featurize(seqs):
    out = np.zeros((len(seqs), 1024), dtype=np.float32)
    for i, seq in enumerate(tqdm(seqs)):
        s = re.sub(r"[UZOB]", "X",
                   "".join(c for c in str(seq) if c.isalpha()).upper())[:1024]
        if not s:
            continue
        text = "<AA2fold> " + " ".join(s)
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=1025).to("cuda")
        with torch.no_grad():
            o = model(**enc)
        h = o.last_hidden_state[0]
        if h.shape[0] > 2:        # drop the sentinel and eos tokens
            h = h[1:-1]
        out[i] = h.mean(0).float().cpu().numpy()
    return out


t0 = time.time()
Xtr = featurize(train_df["Sequence"].tolist())
Xte = featurize(test_df["Sequence"].tolist())
print(f"embedded in {(time.time() - t0) / 60:.1f} min")

# 4) save and download
np.savez("prostT5_aa_matrix.npz",
         X_train=Xtr, y_train=train_df["Label"].to_numpy(),
         entries_train=train_df["Entry"].to_numpy(),
         X_test=Xte, y_test=test_df["Label"].to_numpy(),
         entries_test=test_df["Entry"].to_numpy())

from google.colab import files  # noqa: E402  (Colab-only import)
files.download("prostT5_aa_matrix.npz")
