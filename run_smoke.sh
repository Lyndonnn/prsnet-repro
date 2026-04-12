#!/usr/bin/env bash
set -euo pipefail

: "${DATAROOT:=datasets/shapenet}"
: "${PYTHON:=python3}"
: "${EXP_NAME:=prsnet_smoke}"
: "${GPU_IDS:=0}"
: "${CHECKPOINTS_DIR:=checkpoints}"
: "${RESULTS_DIR:=results}"
: "${MAX_DATASET_SIZE:=1}"
: "${BATCH_SIZE:=1}"
: "${NTHREADS:=0}"
: "${NITER:=1}"
: "${NITER_DECAY:=0}"
: "${PRINT_FREQ:=1}"
: "${DISPLAY_FREQ:=1}"
: "${SAVE_LATEST_FREQ:=1}"
: "${SAVE_EPOCH_FREQ:=1}"
: "${NUM_PLANE:=3}"
: "${NUM_QUAT:=3}"

export DATAROOT PYTHON EXP_NAME GPU_IDS CHECKPOINTS_DIR RESULTS_DIR
export MAX_DATASET_SIZE BATCH_SIZE NTHREADS NITER NITER_DECAY
export PRINT_FREQ DISPLAY_FREQ SAVE_LATEST_FREQ SAVE_EPOCH_FREQ NUM_PLANE NUM_QUAT

echo "[run_smoke] checking dataset"
"$PYTHON" scripts/check_dataset.py --dataroot "$DATAROOT" --phases train test --require-nonempty --sample-count 1

echo "[run_smoke] training one tiny epoch"
./run_train.sh

echo "[run_smoke] running inference on one test file"
WHICH_EPOCH="${WHICH_EPOCH:-latest}" ./run_test.sh

echo "[run_smoke] done"
echo "[run_smoke] checkpoint: $CHECKPOINTS_DIR/$EXP_NAME/latest_net_PRSNet.pth"
echo "[run_smoke] prediction preview: $RESULTS_DIR/$EXP_NAME/test_${WHICH_EPOCH:-latest}/example_prediction.json"
