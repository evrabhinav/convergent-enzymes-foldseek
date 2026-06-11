# Paper draft

This directory holds the MLCB 2026 submission draft.

## Files

- `main.tex`: full paper draft (Title, Abstract, Intro, Related Work,
  Method, Results, Discussion, Conclusion, Acknowledgements)
- `refs.bib`: BibTeX references. All entries verified against the
  canonical source (DGEB is a 2024 bioRxiv preprint, doi
  10.1101/2024.07.10.602933, rejected from ICLR 2025 per OpenReview;
  AlphaFold DB 2024 in NAR; ProstT5 in NAR Genomics and Bioinformatics;
  Riziotis et al. 2025 in The FEBS Journal).
- `figures/`: symlinks / copies of charts from `../charts/` that the
  paper actually references (currently the paper uses
  `\includegraphics` paths relative to the repo root, so this is
  empty)

## Build

The draft uses plain `article` + `natbib`. It compiles standalone:

```bash
cd paper/
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

MLCB 2026 does not mandate a special style file (per mlcb.org/submit):
the full-paper track asks for 11 pt font, 1-inch margins, and a
single-column layout, which this `article`-class draft already uses.
No template needs to be downloaded.

## MLCB 2026 format requirements (verified at mlcb.org/submit)

- **8 pages, EXCLUDING references** (and, per NeurIPS-style convention,
  acknowledgements as back-matter). Over-limit submissions are
  auto-rejected.
- **11 pt font, 1-inch margins, single-column** (all satisfied here;
  the draft is `\documentclass[11pt]` with 1-in geometry).
- No special style/template file required.
- Deadline: **July 1 AOE**. Accepted 8-page papers may optionally
  appear in a PMLR MLCB section.

## Things to double-check before submission

1. **Page count.** The body (Introduction through Conclusion) fits 8
   pages at 11 pt; Acknowledgements + references are back-matter on
   pages 9-10. Re-check after any edits.
2. **AI-tool disclosure.** The Acknowledgements mentions Claude as a
   coding partner. MLCB / NeurIPS allow this disclosure but check the
   2026 call-for-papers for any format requirements.
3. **Statistical reporting.** All p-values come from a Mann-Whitney U
   test on `results/analysis_hard_queries.csv`. Re-run
   `src/analysis_hard_queries.py` to verify numbers if anything in the
   pipeline upstream changes.

## Numbers that appear in the paper, with their source

| Number | Source file |
|---|---|
| `0.267` Convergent Enzymes best | `results/phase13_crossover_results.csv` row "FS(prob>=0.9) > MAJ(3B+ProstT5+150M)" (raw value 0.2668) |
| `0.730` EC best | `results/ec_classification/crossover_results.csv` row "FS(prob>=0.3) > MAJ(4 LMs)" (raw value 0.7305) |
| `0.238` Foldseek alone on CE | `results/phase12_summary.txt` (raw value 0.2383, evaluated over all 400 test queries) |
| `0.663` Foldseek alone on EC | `results/ec_classification/crossover_results.csv` row "Foldseek top-1 alone" (raw value 0.6628) |
| Hard-query means and p-values | `results/analysis_hard_queries.csv` + `src/analysis_hard_queries.py` |
| Ablation chain F1 values | per-phase summary files in `results/` |

If any number in the paper looks off, trace it back via this table.
