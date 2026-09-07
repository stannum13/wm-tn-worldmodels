#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/external/pt_recovery}"
mkdir -p "$(dirname "$destination")"
if [ -d "$destination/.git" ]; then
  echo "Repository already exists at $destination" >&2
  exit 0
fi
git clone --filter=blob:none --sparse https://github.com/guochu/pt_recovery.git "$destination"
git -C "$destination" sparse-checkout set \
  experiment_data/RB_data_20230104/len40/idle100
echo "Fetched RB data under $destination"
