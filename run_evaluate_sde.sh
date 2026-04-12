#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${DATAROOT:-datasets/shapenet}"
PYTHON_BIN="${PYTHON:-python3}"
EXP_NAME="${EXP_NAME:-exp}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PHASE="${PHASE:-test}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
MAX_FILES="${MAX_FILES:-0}"
INCLUDE_PCA="${INCLUDE_PCA:-1}"
OUTPUT="${OUTPUT:-}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-}"

args=(
  "$PYTHON_BIN" scripts/evaluate_sde.py
  --dataroot "$DATAROOT"
  --results-dir "$RESULTS_DIR"
  --exp-name "$EXP_NAME"
  --phase "$PHASE"
  --which-epoch "$WHICH_EPOCH"
  --max-files "$MAX_FILES"
)

if [[ "$INCLUDE_PCA" == "1" || "$INCLUDE_PCA" == "true" ]]; then
  args+=(--include-pca)
fi

if [[ -n "$OUTPUT" ]]; then
  args+=(--output "$OUTPUT")
fi

if [[ -n "$SUMMARY_OUTPUT" ]]; then
  args+=(--summary-output "$SUMMARY_OUTPUT")
fi

echo "[run_evaluate_sde] ${args[*]} $*"
"${args[@]}" "$@"
