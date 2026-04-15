#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E3SYM_ROOT="${E3SYM_ROOT:-$ROOT_DIR/external/e3sym}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results_e3sym/official/test_pretrained}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [ ! -d "$E3SYM_ROOT/.git" ]; then
    mkdir -p "$(dirname "$E3SYM_ROOT")"
    echo "[run_e3sym_test] cloning E3Sym into $E3SYM_ROOT"
    git clone --depth 1 https://github.com/renwuli/e3sym.git "$E3SYM_ROOT"
fi

case "${INSTALL_DEPS:-0}" in
    1|colab|modern)
        echo "[run_e3sym_test] installing Colab-compatible E3Sym inference dependencies"
        python3 -m pip install -U pip setuptools wheel ninja
        python3 -m pip install easydict PyYAML tqdm scipy "numba>=0.59" "numpy<2.2"
        ;;
    legacy)
        echo "[run_e3sym_test] installing upstream pinned E3Sym dependencies"
        python3 -m pip install -r "$E3SYM_ROOT/requirements.txt" "scipy>=1.6"
        ;;
    0|"")
        ;;
    *)
        echo "[run_e3sym_test] unknown INSTALL_DEPS=$INSTALL_DEPS; use 0, 1, colab, modern, or legacy" >&2
        exit 2
        ;;
esac

mkdir -p "$E3SYM_ROOT/.tmp"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

cmd=(
    python3 "$ROOT_DIR/scripts/e3sym_export_predictions.py"
    --e3sym-root "$E3SYM_ROOT"
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
)

if [ -n "${E3SYM_CONFIG:-}" ]; then
    cmd+=(--config "$E3SYM_CONFIG")
fi
if [ -n "${E3SYM_EVAL_ROOT:-}" ]; then
    cmd+=(--eval-root "$E3SYM_EVAL_ROOT")
fi
if [ -n "${E3SYM_BENCHMARK_TXT:-}" ]; then
    cmd+=(--benchmark-txt "$E3SYM_BENCHMARK_TXT")
fi
if [ -n "${E3SYM_WEIGHTS:-}" ]; then
    cmd+=(--weights "$E3SYM_WEIGHTS")
fi
if [ -n "${E3SYM_NPOINTS:-}" ]; then
    cmd+=(--npoints "$E3SYM_NPOINTS")
fi
if [ -n "${MAX_SHAPES:-}" ]; then
    cmd+=(--max-shapes "$MAX_SHAPES")
fi

echo "[run_e3sym_test] ${cmd[*]}"
"${cmd[@]}"
