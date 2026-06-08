# Paper draft

This directory holds the MLCB 2026 submission draft.

## Files

- `main.tex` — full paper draft (Title, Abstract, Intro, Related Work,
  Method, Results, Discussion, Conclusion, Acknowledgements)
- `refs.bib` — BibTeX references. **Every entry marked with a
  `TODO`/`note` comment must be verified against the canonical source
  before submission.** I've kept those flagged so they're easy to find.
- `figures/` — symlinks / copies of charts from `../charts/` that the
  paper actually references (none yet — we use `\includegraphics` paths
  relative to the repo root)

## Build

The draft uses plain `article` + `natbib`. It compiles standalone:

```bash
cd paper/
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

For the actual MLCB submission, the body should be wrapped in the
official MLCB / NeurIPS style file. The MLCB 2026 template (which uses
NeurIPS workshop format) is not included here; download it from the MLCB
website and adapt before submission.

## Things to double-check before submission

1. **Author block.** Currently lists `Abhinav E V R / Independent`.
   If Dr. Green is a co-author, add her affiliation and a corresponding-
   author indicator.
2. **MLCB style file.** Swap the `\documentclass` and packages for the
   MLCB-supplied style.
3. **All bib entries with a `note = {... to be verified ...}` line.**
   Notably DGEB authors, AlphaFold DB 2024, ProstT5 publication
   venue, Riziotis citation.
4. **The Acknowledgements paragraph** mentions Claude as a coding
   partner. MLCB / NeurIPS allow this disclosure but check the 2026
   call-for-papers for any format requirements.
5. **Page count.** Currently ~7 pages of content + 1 page refs.
   Check against the actual MLCB page limit (usually 8 incl. refs for a
   short paper).
6. **Figures.** The draft does not yet embed figures. We have
   `charts/phase13_final.png`, `charts/ec_classification_crossover.png`,
   and `charts/analysis_hard_vs_easy.png` ready to include if there is
   space; otherwise the tables carry the result.
7. **Statistical reporting.** All p-values come from a Mann-Whitney U
   test on `results/analysis_hard_queries.csv`. Re-run
   `src/analysis_hard_queries.py` to verify numbers if anything in
   the pipeline upstream changes.
8. **Wording of the AI-tool disclosure.** Sentence in
   Acknowledgements: "The implementation work used a large language
   model (Claude) as a coding partner; the experimental design, choice
   of methods, and interpretation are the author's." Adjust as needed.

## Numbers that appear in the paper, with their source

| Number | Source file |
|---|---|
| `0.267` Convergent Enzymes best | `results/phase13_crossover_results.csv` row "FS(prob>=0.9) > MAJ(3B+ProstT5+150M)" |
| `0.731` EC best | `results/ec_classification/crossover_results.csv` row "FS(prob>=0.3) > MAJ(4 LMs)" |
| `0.238` Foldseek alone on CE | `results/phase5_foldseek_summary.csv` |
| `0.663` Foldseek alone on EC | computed in `src/generic_crossover.py` run output |
| Hard-query p-values | `results/analysis_hard_queries.csv` + `src/analysis_hard_queries.py` |
| Ablation chain F1 values | per-phase results CSVs |

If any number in the paper looks off, trace it back via this table.
