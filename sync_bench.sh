#!/usr/bin/env bash
# Maintainer only: copy the bench code (and a fresh manifest) from the
# ai-evolution checkout that produced the report, so the kit runs that code.
#   ./sync_bench.sh [path/to/ai-evolution]
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-../ai-evolution}/scripts/quant/bench"
for f in fit.py common.py lineup.py prep.py stage.py; do cp "$SRC/$f" bench/; done
echo "bench/ synced from $SRC"
echo "Manifest: python3 $SRC/make_manifest.py --out kit/manifest.json (on the host with the files)"
