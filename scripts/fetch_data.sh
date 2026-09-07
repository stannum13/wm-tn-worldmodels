#!/usr/bin/env bash
# Fetch the public datasets used by Experiment A into data/external/.
# Idempotent: skips clones that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/external
cd data/external

if [ ! -d pt_recovery ]; then
  git clone --depth 1 https://github.com/guochu/pt_recovery
else
  echo "pt_recovery already present"
fi

if [ ! -d NMN-tomo ]; then
  git clone --depth 1 https://github.com/Christina-Giar/NMN-tomo
else
  echo "NMN-tomo already present"
fi

echo "data/external ready:"
ls -1
