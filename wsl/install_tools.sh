#!/usr/bin/env bash
# Install Foldseek and fpocket inside a WSL2 Ubuntu environment.
# ==============================================================
# Neither tool ships a native Windows build, so the project ran them inside
# WSL2 (Ubuntu 24.04). Foldseek ships a precompiled Linux binary; fpocket is
# built from source. Run this once, as root, inside WSL:
#
#   wsl --install -d Ubuntu-24.04          # (from Windows, one-time)
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/path/to/wsl/install_tools.sh
#
# After this, `foldseek` and `fpocket` are on PATH inside WSL.
set -euo pipefail

echo "=== apt prerequisites ==="
apt-get update
apt-get install -y wget tar build-essential git libnetcdf-dev

echo "=== Foldseek (precompiled Linux binary) ==="
cd /opt
wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xzf foldseek-linux-avx2.tar.gz
rm foldseek-linux-avx2.tar.gz
ln -sf /opt/foldseek/bin/foldseek /usr/local/bin/foldseek
foldseek version

echo "=== fpocket (build from source) ==="
cd /opt
[ -d fpocket-src ] && rm -rf fpocket-src
git clone --depth 1 https://github.com/Discngine/fpocket.git fpocket-src
cd fpocket-src
make
ln -sf /opt/fpocket-src/bin/fpocket /usr/local/bin/fpocket
which fpocket

echo "=== done. foldseek + fpocket are on PATH inside WSL. ==="
