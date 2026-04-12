#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${DATAROOT:-datasets/shapenet}"
PYTHON_BIN="${PYTHON:-python3}"
EXP_NAME="${EXP_NAME:-exp}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PHASE="${PHASE:-test}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
MAX_FILES="${MAX_FILES:-0}"
MAX_FACES="${MAX_FACES:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

args=(
  "$PYTHON_BIN" scripts/visualize_predictions.py
  --dataroot "$DATAROOT"
  --results-dir "$RESULTS_DIR"
  --exp-name "$EXP_NAME"
  --phase "$PHASE"
  --which-epoch "$WHICH_EPOCH"
  --max-files "$MAX_FILES"
  --max-faces "$MAX_FACES"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

echo "[run_visualize] ${args[*]} $*"
"${args[@]}" "$@"
