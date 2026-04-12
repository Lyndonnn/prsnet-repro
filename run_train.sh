#!/usr/bin/env bash
set -euo pipefail

DATAROOT="${DATAROOT:-datasets/shapenet}"
PYTHON_BIN="${PYTHON:-python3}"
EXP_NAME="${EXP_NAME:-exp}"
GPU_IDS="${GPU_IDS:-0}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
NTHREADS="${NTHREADS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_PLANE="${NUM_PLANE:-3}"
NUM_QUAT="${NUM_QUAT:-3}"
NITER="${NITER:-100}"
NITER_DECAY="${NITER_DECAY:-100}"
LR="${LR:-0.001}"
WEIGHT="${WEIGHT:-25}"
PRINT_FREQ="${PRINT_FREQ:-100}"
DISPLAY_FREQ="${DISPLAY_FREQ:-1}"
SAVE_LATEST_FREQ="${SAVE_LATEST_FREQ:-1000}"
SAVE_EPOCH_FREQ="${SAVE_EPOCH_FREQ:-10}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-}"

args=(
  "$PYTHON_BIN" train.py
  --dataroot "$DATAROOT"
  --name "$EXP_NAME"
  --gpu_ids "$GPU_IDS"
  --checkpoints_dir "$CHECKPOINTS_DIR"
  --nThreads "$NTHREADS"
  --batchSize "$BATCH_SIZE"
  --num_plane "$NUM_PLANE"
  --num_quat "$NUM_QUAT"
  --niter "$NITER"
  --niter_decay "$NITER_DECAY"
  --lr "$LR"
  --weight "$WEIGHT"
  --print_freq "$PRINT_FREQ"
  --display_freq "$DISPLAY_FREQ"
  --save_latest_freq "$SAVE_LATEST_FREQ"
  --save_epoch_freq "$SAVE_EPOCH_FREQ"
)

if [[ -n "$MAX_DATASET_SIZE" ]]; then
  args+=(--max_dataset_size "$MAX_DATASET_SIZE")
fi

echo "[run_train] ${args[*]} $*"
"${args[@]}" "$@"
