#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/rigetti_stability8_resets}"
file="$destination/stability_8_with_resets_raw_data.h5"
expected="74458b3a992d94b76c9a7dd844007e3c"
url="https://zenodo.org/api/records/15364358/files/stability_8_with_resets_raw_data.h5/content"
mkdir -p "$destination"
curl -L --fail --retry 3 --connect-timeout 20 -o "$file" "$url"
if command -v md5sum >/dev/null 2>&1; then
  actual="$(md5sum "$file" | awk '{print $1}')"
else
  actual="$(md5 -q "$file")"
fi
test "$actual" = "$expected"
echo "verified $file ($actual)"
