#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/rigetti_stability8_no_resets}"
file="$destination/stability_8_without_resets_raw_data.h5"
expected="721b7368a819e13eb404fea5154744a4"
url="https://zenodo.org/api/records/15364358/files/stability_8_without_resets_raw_data.h5/content"
mkdir -p "$destination"
curl -L --fail --retry 3 --connect-timeout 20 -o "$file" "$url"
if command -v md5sum >/dev/null 2>&1; then
  actual="$(md5sum "$file" | awk '{print $1}')"
else
  actual="$(md5 -q "$file")"
fi
test "$actual" = "$expected"
echo "verified $file ($actual)"
