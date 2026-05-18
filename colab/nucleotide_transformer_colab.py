# Nucleotide Transformer (NT v2 250M Multispecies) embeddings on a Colab T4 GPU
# ==============================================================================
# NT is a DNA-language foundation model. The DGEB paper reports NT-v2-250M at
# F1 = 0.506 on Convergent Enzymes Classification (the NA-track SOTA), so
# adding its embeddings to our Foldseek + AA-LM ensemble is the natural
# follow-up if we want to engage the NA track too.
#
# Inputs come from data/dna_sequences.csv produced by src/fetch_dna.py — that
# file maps each UniProt ID to its CDS DNA fetched from NCBI EFetch.
#
# HOW TO RUN:
#   1. Open https://colab.research.google.com -> New notebook
#   2. Runtime -> Change runtime type -> T4 GPU
#   3. Upload data/dna_sequences.csv (Files panel -> Upload to session storage)
#   4. Paste this whole file into one cell, Shift+Enter
#   5. When it finishes, the browser auto-downloads nt_v2_250m_matrix.npz
#      (~25 MB)
#   6. Move the .npz into features/ on your laptop
#
# Per-protein forward on T4: ~0.5-1 s. Full dataset: ~20-40 min.

# transformers 5.x broke NT v2's custom config (no `rope_theta` attribute
# on EsmConfig). Pin to 4.46.x which still supports the NT remote-code path.
# !pip install -q transformers==4.46.3 datasets

import time, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"

# 1) load model in fp16 on GPU
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True,
                                  torch_dtype=torch.float16).cuda().eval()
print("model loaded:", sum(p.numel() for p in model.parameters())/1e9, "B params")
print("hidden_size:", model.config.hidden_size)

# 2) load DNA sequences (uploaded by the user)
dna = pd.read_csv("dna_sequences.csv")
dna = dna[dna["status"] == "ok"].copy()
print(f"DNA rows available: {len(dna)}")
dna_map = dict(zip(dna["Entry"].astype(str), dna["cds_dna"].astype(str)))

# We need the same train/test splits as the AA pipeline. Fetch them from HF.
from datasets import load_dataset
ds = load_dataset("tattabio/convergent_enzymes")
train_df = ds["train"].to_pandas()
test_df = ds["test"].to_pandas()
print("train:", len(train_df), "test:", len(test_df))


# 3) embed: NT v2 uses 6-mer tokenization; truncate to 1000 tokens (~6 kbp)
MAX_TOK = 1000

def featurize(entries):
    hid = model.config.hidden_size
    out = np.zeros((len(entries), hid), dtype=np.float32)
    n_missing = 0
    for i, e in enumerate(tqdm(entries)):
        seq = dna_map.get(str(e))
        if not seq:
            n_missing += 1
            continue
        # Keep only ATCG and uppercase
        seq = "".join(c for c in seq.upper() if c in "ACGT")
        if not seq:
            n_missing += 1
            continue
        enc = tok(seq, return_tensors="pt", truncation=True,
                  max_length=MAX_TOK, padding=False).to("cuda")
        with torch.no_grad():
            o = model(**enc, output_hidden_states=False)
        # NT returns last_hidden_state of shape (1, L, H)
        h = o.last_hidden_state[0]
        # Drop CLS-like first token if model uses one
        if h.shape[0] > 2:
            h = h[1:]
        out[i] = h.mean(0).float().cpu().numpy()
    print(f"missing DNA for {n_missing}/{len(entries)} entries")
    return out


t0 = time.time()
Xtr = featurize(train_df["Entry"].tolist())
Xte = featurize(test_df["Entry"].tolist())
print(f"embedded in {(time.time()-t0)/60:.1f} min")

# 4) save and download
np.savez("nt_v2_250m_matrix.npz",
         X_train=Xtr, y_train=train_df["Label"].to_numpy(),
         entries_train=train_df["Entry"].to_numpy(),
         X_test=Xte, y_test=test_df["Label"].to_numpy(),
         entries_test=test_df["Entry"].to_numpy())

from google.colab import files  # noqa: E402  (Colab-only import)
files.download("nt_v2_250m_matrix.npz")
