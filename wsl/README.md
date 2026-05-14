# WSL-side scripts

Foldseek and fpocket have no native Windows builds, so this project ran them
inside WSL2 (Ubuntu 24.04). These scripts capture the exact commands used.

## Order of operations

```
# --- one-time, from Windows ---
wsl --install -d Ubuntu-24.04

# --- inside WSL, as root, once ---
wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/<path>/wsl/install_tools.sh

# --- on the Windows side: download + organize structures ---
python src/phase1_load_and_download.py
python src/organize_structures.py        # makes structures/train, structures/test

# --- inside WSL: Foldseek search + 3Di extraction ---
wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/<path>/wsl/run_foldseek.sh \
    "/mnt/c/<path>/Convergent Enzyme Claasification"
# -> writes foldseek_workdir/hits.tsv, train_3di.fasta, test_3di.fasta

# --- inside WSL: fpocket (only needed for Phase 7) ---
wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/<path>/wsl/run_fpocket.sh
# -> writes structures/{train,test}/<id>_out/ directories
```

`<path>` is wherever the repo lives, e.g.
`/mnt/c/Users/Abhinav/OneDrive/Desktop/Convergent Enzyme Claasification`.

## Files

| Script | What it does |
|---|---|
| `install_tools.sh` | Installs Foldseek (precompiled binary) and builds fpocket from source; symlinks both onto PATH inside WSL. |
| `run_foldseek.sh` | Builds train/test Foldseek DBs, runs the test-vs-train `easy-search` (→ `hits.tsv`), extracts 3Di FASTA sequences, and copies the artifacts back into `foldseek_workdir/`. |
| `run_fpocket.sh` | Runs fpocket 8-way parallel over every train/test PDB (Phase 7 input). |

## Notes

- Foldseek's argument parser chokes on spaces in paths; `run_foldseek.sh`
  works around this by symlinking the split directories into `/root/fs_work`.
- The headline result uses **default** Foldseek scoring. TM-align rescoring
  and iterative search were both tried (`src/phase5b_tier_a.py`) and both
  reduced accuracy on this task.
- `phase5_foldseek.py` in `src/` is a higher-level Python wrapper around the
  same `createdb` / `easy-search` calls with a pluggable backend; these shell
  scripts are the exact commands that produced the committed artifacts.
