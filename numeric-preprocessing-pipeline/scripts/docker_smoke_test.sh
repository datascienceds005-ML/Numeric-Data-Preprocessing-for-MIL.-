#!/usr/bin/env bash
# CI-agnostic container smoke test: build → synthesize inside a container →
# run → assert outputs. Exits non-zero on any failure; no-ops with a clear
# message when docker is unavailable (e.g. sandboxed local environments).
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "[smoke] docker not available — skipping (CI will run this gate)."
  exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[smoke] building image"
docker build -q -t numprep:smoke .

echo "[smoke] generating synthetic messy dataset inside the container"
docker run --rm -v "$WORK:/data" numprep:smoke synthesize --rows 2000 --seed 7 --out /data

echo "[smoke] running the pipeline"
docker run --rm \
  -v "$WORK:/data:ro" \
  -v "$WORK/out:/out" \
  -e NUMPREP_SEED=42 \
  numprep:smoke run --input /data --output /out --log-level ERROR

for f in train val test; do
  test -s "$WORK/out/data/$f.parquet" || { echo "[smoke] FAIL: missing $f.parquet"; exit 1; }
done
test -s "$WORK/out/artifacts/manifest.json" || { echo "[smoke] FAIL: no manifest"; exit 1; }
test -s "$WORK/out/reports/summary.md" || { echo "[smoke] FAIL: no summary report"; exit 1; }

echo "[smoke] OK — outputs: $WORK/out"
