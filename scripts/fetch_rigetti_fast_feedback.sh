#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/rigetti_fast_feedback}"
file="$destination/fast_feedback_raw_data.h5"
expected="3b2503a80f2b92916660489e2f07e880"
url="https://zenodo.org/api/records/15364358/files/fast_feedback_raw_data.h5/content"
mkdir -p "$destination"
curl -L --fail --retry 3 --connect-timeout 20 -o "$file" "$url"
if command -v md5sum >/dev/null 2>&1; then
  actual="$(md5sum "$file" | awk '{print $1}')"
else
  actual="$(md5 -q "$file")"
fi
test "$actual" = "$expected"
echo "verified $file ($actual)"
