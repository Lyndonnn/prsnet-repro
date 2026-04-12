#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${DATAROOT:-datasets/shapenet}"
PYTHON_BIN="${PYTHON:-python3}"
EXP_NAME="${EXP_NAME:-exp}"
GPU_IDS="${GPU_IDS:-0}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
NUM_PLANE="${NUM_PLANE:-3}"
NUM_QUAT="${NUM_QUAT:-3}"
WHICH_EPOCH="${WHICH_EPOCH:-latest}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-}"

args=(
  "$PYTHON_BIN" test.py
  --dataroot "$DATAROOT"
  --name "$EXP_NAME"
  --gpu_ids "$GPU_IDS"
  --checkpoints_dir "$CHECKPOINTS_DIR"
  --results_dir "$RESULTS_DIR"
  --num_plane "$NUM_PLANE"
  --num_quat "$NUM_QUAT"
  --which_epoch "$WHICH_EPOCH"
)

if [[ -n "$MAX_DATASET_SIZE" ]]; then
  args+=(--max_dataset_size "$MAX_DATASET_SIZE")
fi

echo "[run_test] ${args[*]} $*"
"${args[@]}" "$@"
