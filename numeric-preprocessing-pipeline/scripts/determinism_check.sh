#!/usr/bin/env bash
# Local determinism gate (no docker required): run twice in fresh processes
# with different PYTHONHASHSEED values and compare outputs.
set -euo pipefail

SRC="${1:-data/raw}"
OUT_A="$(mktemp -d)"
OUT_B="$(mktemp -d)"
trap 'rm -rf "$OUT_A" "$OUT_B"' EXIT

PYTHONHASHSEED=0 python -m numprep run --input "$SRC" --output "$OUT_A" --seed 42 --log-level ERROR
PYTHONHASHSEED=1337 python -m numprep run --input "$SRC" --output "$OUT_B" --seed 42 --log-level ERROR

python scripts/compare_deterministic.py "$OUT_A" "$OUT_B"
echo "[determinism] OK"
