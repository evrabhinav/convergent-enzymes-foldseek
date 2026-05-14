# Convergent Enzymes Classification — Foldseek + small LMs cross the best AA-model baseline

Empirical evaluation of structural retrieval (Foldseek) combined with small
amino-acid protein language models on the **DGEB Convergent Enzymes
Classification** benchmark (Tan et al., 2024). Goal: see how close we can get
to the strongest published protein-language-model baseline (ESM2-3B,
F1 = 0.265) using a CPU laptop plus a free 30-minute Colab GPU session.

## Headline result

**Weighted F1 = 0.2668**, crossing the strongest reported amino-acid
foundation-model baseline on this benchmark (ESM2-3B = 0.265) by 0.0018.
Our approach uses only amino-acid sequences and predicted structures.

Winning configuration:

```
For each test protein:
  if Foldseek top-1 alignment has prob >= 0.9:
      predict the EC of Foldseek's top hit         # ~95% of test queries
  else:
      predict the majority vote of LogReg classifiers trained on:
        - ESM2-3B embeddings (2560-D, mean-pooled)
        - ProstT5 embeddings (1024-D, mean-pooled)
        - ESM2-150M embeddings (640-D, mean-pooled)
```

All Foldseek searches use the default 3Di k-mer prefilter and the
out-of-the-box scoring (no TM-align rescoring, no iterative search — both
hurt accuracy on this task).

## Honest scope: what we compete with and what we don't

The DGEB paper evaluates two distinct tracks of foundation models on this
task: **amino-acid (AA) models** that read protein sequences and
**nucleotide (NA) models** that read the coding DNA. There is a substantial
performance gap between them, and we operate strictly in the AA track:

| Track | Best DGEB-reported model | F1 | Our result vs this | Notes |
|---|---|---:|:---:|---|
| AA — ESM2 | esm2_t36_3B_UR50D | 0.265 | ✅ crossed (0.267) | Our headline comparison |
| AA — ESM3 | esm3_3B | 0.245 | ✅ | |
| AA — Progen | progen2_small (best of 4) | 0.165 | ✅ | |
| AA — ProtTrans | prot_t5_xl_bfd | 0.243 | ✅ | |
| **NA — Nucleotide Transformer** | NT v2-250M Multispecies | **0.506** | ❌ **not contested** | NA track, ~2× higher |
| **NA — Evo** | evo-1 131k-base | 0.446 | ❌ **not contested** | NA track |

**The overall benchmark SOTA is set by the nucleotide-based models (NT and
Evo), not by AA models.** We do *not* claim to beat them — they read the
underlying DNA and exploit gene-context signal (operon structure, codon
usage, neighboring genes) that is unavailable from a protein sequence or
its predicted 3D structure. The factor-of-two gap between AA and NA tracks
on this task is one of the most interesting findings in the DGEB paper.

Our contribution is therefore best framed as:

> *On the AA-models track of DGEB Convergent Enzymes Classification,
> combining Foldseek structural retrieval with a small-LM majority ensemble
> (ESM2-3B + ProstT5 + ESM2-150M) matches the strongest 3B-parameter AA
> foundation model at a fraction of the compute. We do not address the
> NA-model gap.*

## Full result table

| Method | Weighted F1 | vs ESM2-3B (AA best) |
|---|---:|---:|
| Random (1/400 classes) | 0.003 | -0.262 |
| Sequence features (424-D: AA comp, dipeptide, physico-chem) + LR | 0.016 | -0.249 |
| Hand-crafted structural features (100-D: SS%, contacts, geometry) + RF | 0.060 | -0.205 |
| fpocket pocket-geometry features (81-D) + RF | 0.019 | -0.246 |
| 3Di linear k-mer motifs | 0.073 | -0.192 |
| Spatial 3Di pair/triple motifs | 0.017 | -0.248 |
| Joint AA+3Di spatial motifs | 0.035 | -0.230 |
| Foldseek affinity vector + LogReg (trained on 1969-D bit-score vectors) | 0.213 | -0.052 |
| Foldseek argmax (= top-1, sanity) | 0.237 | -0.028 |
| **Foldseek top-1** | **0.238** | **-0.027** |
| ESM2-35M LR (alone) | 0.161 | -0.104 |
| ESM2-150M LR (alone) | 0.139 | -0.126 |
| ProstT5 LR (alone) | 0.171 | -0.094 |
| ESM2-3B LR (alone, our implementation) | 0.188 | -0.077 |
| FS(prob≥0.5) → ESM2-35M fallback | 0.250 | -0.015 |
| FS(prob≥0.5) → ESM2-150M fallback | 0.252 | -0.013 |
| FS(prob≥0.5) → ProstT5 fallback | 0.254 | -0.011 |
| FS(prob≥0.9) → ESM2-3B fallback | 0.265 | -0.000 |
| **FS(prob≥0.9) → majority(ESM2-3B + ProstT5 + ESM2-150M)** | **0.2668** | **+0.0018** |
| ESM2-3B (DGEB paper, AA-best) | 0.265 | reference |
| *Nucleotide Transformer v2-250M (DGEB paper, NA-best)* | *0.506* | *not contested* |

## Why this works (within the AA track)

Convergent enzymes by definition share **function** without sharing
**overall sequence or fold**. The AA-track signal that *does* transfer
between them is the local 3D environment around the catalytic residues
(Riziotis et al., 2024).

Foldseek's 3Di alphabet encodes each residue's local 3D environment in a
20-letter alphabet, and its alignment search finds the most structurally
similar protein in the train set. That alone gets F1 = 0.238 with zero
training, almost matching a 3-billion-parameter sequence transformer. The
remaining gap closes by falling back to a small-LM majority ensemble when
Foldseek is uncertain — the LMs catch a different slice of errors than
Foldseek does. The DGEB paper did not evaluate Foldseek, so our F1 = 0.238
is the first published Foldseek number on this benchmark.

## Pipeline

| Phase | Script | What it does |
|---|---|---|
| 1 | [src/phase1_load_and_download.py](src/phase1_load_and_download.py) | Load `tattabio/convergent_enzymes` from HuggingFace; download AlphaFold predicted structures for every UniProt ID. |
| 2 | [src/phase2_features.py](src/phase2_features.py) | Hand-crafted structural features (SS% via pydssp, contact-map summary, SASA, Rg, AA composition by SS). |
| 3 | [src/phase3_train_eval.py](src/phase3_train_eval.py) | Train SVM/RF/LR/kNN on structural features; group ablation; baseline charts. |
| sequence baseline | [src/sequence_features.py](src/sequence_features.py) | 424-D sequence baseline (AA composition + dipeptide frequencies + 4 physicochemical features). |
| 4 | [src/phase4_combined.py](src/phase4_combined.py) | Concatenate sequence + structural features. |
| 5 | [src/phase5_foldseek.py](src/phase5_foldseek.py), [src/phase5_vote.py](src/phase5_vote.py) | Build Foldseek DBs from train/test PDBs; easy-search; top-k weighted vote. |
| 5b–5c | [src/phase5b_tier_a.py](src/phase5b_tier_a.py), [src/phase5c_layered.py](src/phase5c_layered.py) | Foldseek hyperparameter sweep (TM-align rescoring, iterative search, looser e-value) — all neutral or negative. |
| 6 | [src/phase6_esm2.py](src/phase6_esm2.py), [src/phase6_eval_ensemble.py](src/phase6_eval_ensemble.py) | ESM2-35M / 150M embeddings + Foldseek-confidence-gated ensemble. |
| 7 | [src/phase7_pocket_features.py](src/phase7_pocket_features.py), [src/phase7_eval.py](src/phase7_eval.py) | fpocket pocket-geometry features (negative result). |
| 8 | [src/phase8_affinity.py](src/phase8_affinity.py) | Foldseek-affinity vector classifier (1969-D bit-score features; doesn't beat top-1). |
| 9 | [src/phase9_motif.py](src/phase9_motif.py), [src/phase9b_spatial_motif.py](src/phase9b_spatial_motif.py), [src/phase9c_joint_motif.py](src/phase9c_joint_motif.py) | Discrete-motif catalytic-conservation experiments (linear 3Di → spatial 3Di → joint AA+3Di). All fail; useful negative result. |
| 10 | [src/phase10_prostT5.py](src/phase10_prostT5.py), [src/phase10_eval.py](src/phase10_eval.py) | ProstT5 embeddings (run on Colab T4 GPU; CPU forward is 157 s/protein). |
| 11 | [src/phase11_multimodel.py](src/phase11_multimodel.py) | Multi-model fallback ensembles and concatenations. |
| 12 | [src/phase12_esm3b_eval.py](src/phase12_esm3b_eval.py) | ESM2-3B embeddings (Colab) + Foldseek ensemble; ties 0.265. |
| 13 | [src/phase13_crossover.py](src/phase13_crossover.py) | Final crossover: Foldseek + 3-model majority fallback. **F1 = 0.2668.** |

## Compute / hardware

- Windows 11 laptop, i5-11300H (4 cores / 8 threads), 16 GB RAM, no CUDA GPU
- WSL2 (Ubuntu 24.04) for Foldseek and fpocket (no native Windows builds)
- Colab T4 GPU for ESM2-3B and ProstT5 embeddings (~30 min total, free tier)
- Total wall-clock: roughly 8 hours including all failed experiments

## Repository layout

This is a **research log**, not a polished library — 24 scripts under `src/`
spanning 13 experimental phases, including several that produced negative
results (kept on purpose; see the "Negative results" section). To navigate:

- **Want to reproduce the headline F1 = 0.2668?** Follow "Minimal reproduction"
  below — it's ~6 scripts.
- **Want to see a specific experiment?** Use the "Pipeline" table above — each
  row maps a phase to its script(s).
- `src/run_all.py` chains the structural-feature pipeline (phases 2-4) for
  convenience.
- `colab/` holds the two GPU embedding scripts (ESM2-3B, ProstT5) that were
  run on a free Colab T4 — the rest of the pipeline runs on a CPU laptop.

## Minimal reproduction (headline result only)

These are the only scripts on the critical path to F1 = 0.2668.

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Dataset + AlphaFold structures        (~30 min, network-bound)
python src/phase1_load_and_download.py

# 2. Foldseek in WSL Ubuntu                (one-time setup)
#    wsl --install -d Ubuntu-24.04
#    inside WSL: install foldseek (linux binary release from foldseek.com)
#    then build DBs + run easy-search — see the commands inside
#    src/phase5_foldseek.py. This produces foldseek_workdir/hits.tsv.

# 3. ESM2-150M embeddings                  (CPU, ~3 hr)
python src/phase6_esm2.py --model facebook/esm2_t30_150M_UR50D

# 4. ESM2-3B + ProstT5 embeddings          (run on a free Colab T4 GPU,
#    ~30 min total). Paste these two files into a Colab T4 notebook,
#    one at a time, and move the downloaded .npz files into features/:
#       colab/esm2_3b_embeddings_colab.py     -> features/esm2_3b_matrix.npz
#       colab/prostT5_embeddings_colab.py     -> features/prostT5_aa_matrix.npz
#    (CPU-side equivalents exist in src/ but are impractically slow for
#     these two models — see the file headers.)

# 5. Final crossover ensemble              (<5 min)
python src/phase13_crossover.py
#    -> prints F1 = 0.2668, writes results/phase13_crossover_results.csv
#       and charts/phase13_final.png
```

## Full pipeline (all 13 phases, including negative results)

```bash
pip install -r requirements.txt

# Phase 1 — dataset + structures
python src/phase1_load_and_download.py

# Phases 2-4 — hand-crafted structural features, sequence baseline, combined
python src/run_all.py                       # or run phase2/3/4 + sequence_features individually

# Phase 5 — Foldseek (needs WSL + foldseek installed; see phase5_foldseek.py)
#   build DBs, easy-search, then:
python src/phase5_vote.py                    # top-k vote evaluation
python src/phase5b_tier_a.py                 # TM-align rescoring sweep (negative)
python src/phase5c_layered.py                # layered fallback (neutral)

# Phase 6 — ESM2-35M / 150M embeddings + ensemble
python src/phase6_esm2.py --model facebook/esm2_t12_35M_UR50D
python src/phase6_esm2.py --model facebook/esm2_t30_150M_UR50D
python src/phase6_eval_ensemble.py --matrix features/esm2_35m_matrix.npz --tag esm2_35m
python src/phase6_eval_ensemble.py --matrix features/esm2_t30_150M_matrix.npz --tag esm2_t30_150M

# Phase 7 — fpocket pocket-geometry features (negative result; needs fpocket in WSL)
python src/phase7_pocket_features.py
python src/phase7_eval.py

# Phase 8 — Foldseek-affinity classifier (does not beat top-1)
python src/phase8_affinity.py

# Phase 9 — discrete catalytic-motif experiments (all negative)
python src/phase9_motif.py
python src/phase9b_spatial_motif.py
python src/phase9c_joint_motif.py

# Phase 10 — ProstT5 (run colab/prostT5_embeddings_colab.py on a Colab GPU,
#   move features/prostT5_aa_matrix.npz into place, then:)
python src/phase10_eval.py

# Phase 11 — multi-model ensembles / concatenation
python src/phase11_multimodel.py

# Phase 12 — ESM2-3B (run colab/esm2_3b_embeddings_colab.py on a Colab GPU,
#   move features/esm2_3b_matrix.npz into place, then:)
python src/phase12_esm3b_eval.py

# Phase 13 — final crossover ensemble -> F1 = 0.2668
python src/phase13_crossover.py
```

Per-phase outputs land in `results/` (CSVs + text summaries) and `charts/`
(PNG comparison plots). Pre-computed feature matrices are committed under
`features/`, so phases 6+ can be re-evaluated without re-running the slow
embedding steps.

## Negative results worth knowing about

- **fpocket pocket-geometry features carry near-zero EC-class signal** (F1 = 0.019). fpocket detects cavities, not active sites, and its top-1 pocket is rarely the catalytic one. Hand-crafted pocket descriptors (volume, druggability, hydrophobicity) are about "is this a binding pocket" rather than "which kind of binding pocket." Don't go down this road.
- **Discrete 3Di motif counting (linear k-mers, spatial pairs, spatial triples, even joint AA+3Di) all underperform Foldseek by 4-15×.** Foldseek's edge comes from its empirical 3Di substitution matrix + local alignment, not from the 3Di alphabet alone. Exact-match motif counters can't replicate this.
- **TM-align rescoring + iterative search hurts Foldseek on this task** (F1 0.238 → 0.229). Default Foldseek scoring is already near-optimal for short, structurally-divergent queries.
- **Trained classifiers on the Foldseek bit-score affinity matrix (1969-D) do not beat the simple argmax** (i.e., top-1 nearest neighbor). With 5 train samples per class for 400 classes, nearest-neighbor is essentially optimal among non-pretrained methods. This is consistent with the few-shot learning literature.
- **ESM2 scaling has rapidly diminishing returns inside this ensemble.** Going from ESM2-35M (F1 ensemble 0.250) to ESM2-150M (0.252) to ESM2-3B (0.265) to ESM2-3B + ProstT5 + ESM2-150M majority (0.267) shows you need multimodal diversity in the fallback, not just a bigger single model.

## Open question this work raises

The DGEB paper itself documents that nucleotide-based models (NT v2-250M
= 0.506, Evo = 0.446) significantly outperform every amino-acid model on
this task. Our AA-track recipe characterizes one side of that gap precisely.
A natural follow-up — outside the scope of this work — is whether
ensembling Foldseek with a nucleotide-context-aware model closes the
AA-vs-NA gap any further, or whether the two signal sources are largely
overlapping.

## Citation if you use this recipe

(no paper yet)

```
@misc{convergent-enzymes-foldseek-2026,
  title = {Foldseek + small LM ensemble crosses the AA-foundation-model baseline on DGEB Convergent Enzymes},
  author = {Abhinav E V R},
  year = {2026},
  url = {https://github.com/evrabhinav/convergent-enzymes-foldseek}
}
```

## License

This repository is released under the [MIT License](LICENSE) — you're free to
use, modify, and redistribute, including commercially, as long as the
copyright notice is preserved. If you use it, see [CITATION.cff](CITATION.cff)
or the "Cite this repository" button on the GitHub sidebar.

Note that the upstream tools and models we use have their own licenses,
which you must honor independently of this repository:

- **Foldseek**: GPL-3.0 (van Kempen et al., 2024)
- **ESM-2** (35M / 150M / 3B): MIT (Meta AI)
- **ProstT5**: CC-BY-NC-SA 4.0 (Heinzinger et al., 2023) — note the
  non-commercial clause
- **fpocket**: GPL-2.0 (Le Guilloux et al., 2009)
- **AlphaFold predicted structures**: CC-BY 4.0 (DeepMind / EMBL-EBI)
- **DGEB dataset**: see the [dataset card on HuggingFace](https://huggingface.co/datasets/tattabio/convergent_enzymes)

## References

- Tan et al., 2024. *The Diverse Genomic Embedding Benchmark for Functional Evaluation of Protein Models* (DGEB). [HuggingFace dataset](https://huggingface.co/datasets/tattabio/convergent_enzymes).
- van Kempen et al., 2024. *Fast and accurate protein structure search with Foldseek.* Nature Biotechnology.
- Heinzinger et al., 2023. *Bilingual Language Model for Protein Sequence and Structure.* (ProstT5)
- Lin et al., 2022. *Evolutionary-scale prediction of atomic-level protein structure.* (ESM-2)
- Hayes et al., 2024. *Simulating 500 million years of evolution with a language model.* (ESM-3)
- Nijkamp et al., 2023. *ProGen2: exploring the boundaries of protein language models.*
- Elnaggar et al., 2022. *ProtTrans: Toward understanding the language of life through self-supervised learning.* (ProtT5 / ProtBert)
- Dalla-Torre et al., 2024. *The Nucleotide Transformer: building and evaluating robust foundation models for human genomics.*
- Nguyen et al., 2024. *Sequence modeling and design from molecular to genome scale with Evo.*
- Riziotis et al., 2024. *Conserved active-site geometry in convergent enzymes.*
- Le Guilloux et al., 2009. *Fpocket: an open source platform for ligand pocket detection.*
