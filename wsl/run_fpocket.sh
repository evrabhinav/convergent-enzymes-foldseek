#!/usr/bin/env bash
# Run fpocket on every train/test structure (Phase 7 input).
# ==========================================================
# fpocket writes a <name>_out/ directory next to each input PDB. We run it
# 8-way parallel via xargs. ~1 s per protein, but the WSL<->Windows filesystem
# writes dominate, so the full run takes ~1 hour.
#
# Prerequisites: install_tools.sh + run_foldseek.sh have been run, so
# /root/fs_work/{train,test} symlinks exist.
#
# Usage:
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/path/to/wsl/run_fpocket.sh
#
# After this finishes, run src/phase7_pocket_features.py on the Windows side
# to parse the *_out/ directories into a feature matrix.
set -euo pipefail

WORK=/root/fs_work
cd "$WORK"

echo "=== clearing any previous fpocket output ==="
find "$WORK/train" -maxdepth 1 -type d -name '*_out' -exec rm -rf {} + 2>/dev/null || true
find "$WORK/test"  -maxdepth 1 -type d -name '*_out' -exec rm -rf {} + 2>/dev/null || true

echo "=== running fpocket (8-way parallel) ==="
date
ls "$WORK"/train/*.pdb "$WORK"/test/*.pdb \
  | xargs -P 8 -I {} sh -c 'fpocket -f "{}" > /dev/null 2>&1'
date

echo "train _out dirs: $(find "$WORK/train" -maxdepth 1 -type d -name '*_out' | wc -l)"
echo "test  _out dirs: $(find "$WORK/test"  -maxdepth 1 -type d -name '*_out' | wc -l)"
echo "=== done. Now run src/phase7_pocket_features.py on the Windows side. ==="
