#!/usr/bin/env bash
# Foldseek pipeline: build databases, run the test-vs-train structural search,
# and extract 3Di sequences.
# ===========================================================================
# Prerequisites:
#   - install_tools.sh has been run (foldseek on PATH)
#   - phase1_load_and_download.py + src/organize_structures.py have been run on
#     the Windows side, so structures/train/ and structures/test/ exist
#
# Usage (from Windows):
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/path/to/wsl/run_foldseek.sh \
#       "/mnt/c/Users/<you>/.../Convergent Enzyme Claasification"
#
# Foldseek's argument parser does not like spaces in paths, so we symlink the
# split directories into a no-space working dir (/root/fs_work) first.
set -euo pipefail

PROJECT="${1:?pass the project directory as /mnt/c/... }"
WORK=/root/fs_work
mkdir -p "$WORK/tmp"
cd "$WORK"

echo "=== symlink split dirs into a no-space path ==="
ln -sfn "$PROJECT/structures/train" "$WORK/train"
ln -sfn "$PROJECT/structures/test"  "$WORK/test"
echo "train PDBs: $(ls "$WORK"/train/*.pdb | wc -l)"
echo "test  PDBs: $(ls "$WORK"/test/*.pdb  | wc -l)"

echo "=== build Foldseek databases ==="
foldseek createdb train trainDB --threads 8
foldseek createdb test  testDB  --threads 8

echo "=== test-vs-train structural search (produces hits.tsv) ==="
# Default 3Di scoring, sensitivity 9.5, e-value 10, up to 50 hits per query.
# (TM-align rescoring and iterative search were tried and both HURT — see
#  src/phase5b_tier_a.py. Default scoring is used for the headline result.)
foldseek easy-search test trainDB hits.tsv tmp \
  --threads 8 -s 9.5 -e 10 --max-seqs 50 \
  --format-output "query,target,bits,evalue,prob,alntmscore,fident,lddt"
echo "hits.tsv: $(wc -l < hits.tsv) rows"

echo "=== extract 3Di sequences (for the motif experiments, phases 9/9b/9c) ==="
# The 3Di strings live in the *_ss component of each DB; convert2fasta needs
# the header (_h) component linked alongside it.
for split in train test; do
  foldseek lndb "${split}DB_ss" "${split}_3di_db"
  foldseek lndb "${split}DB_h"  "${split}_3di_db_h"
  foldseek convert2fasta "${split}_3di_db" "${split}_3di.fasta"
done

echo "=== copy the artifacts the Python side needs back into the repo ==="
mkdir -p "$PROJECT/foldseek_workdir"
cp "$WORK/hits.tsv"        "$PROJECT/foldseek_workdir/"
cp "$WORK/train_3di.fasta" "$PROJECT/foldseek_workdir/"
cp "$WORK/test_3di.fasta"  "$PROJECT/foldseek_workdir/"

echo "=== done. foldseek_workdir/ now has hits.tsv + the 3Di FASTAs. ==="
