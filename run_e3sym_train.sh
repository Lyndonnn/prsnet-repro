#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E3SYM_ROOT="${E3SYM_ROOT:-$ROOT_DIR/external/e3sym}"
GPU_IDS="${GPU_IDS:-0}"
CONFIG="${E3SYM_CONFIG:-configs/train.yaml}"

if [ ! -d "$E3SYM_ROOT/.git" ]; then
    mkdir -p "$(dirname "$E3SYM_ROOT")"
    echo "[run_e3sym_train] cloning E3Sym into $E3SYM_ROOT"
    git clone --depth 1 https://github.com/renwuli/e3sym.git "$E3SYM_ROOT"
fi

if [ -n "${SHAPENET_DIR:-}" ] && [ ! -e "$E3SYM_ROOT/dataset/ShapeNetCore.v2" ]; then
    echo "[run_e3sym_train] linking ShapeNetCore.v2 from $SHAPENET_DIR"
    ln -s "$SHAPENET_DIR" "$E3SYM_ROOT/dataset/ShapeNetCore.v2"
fi

if [ "${INSTALL_DEPS:-0}" = "1" ]; then
    echo "[run_e3sym_train] installing E3Sym dependencies"
    python3 -m pip install -r "$E3SYM_ROOT/requirements.txt"
fi

mkdir -p "$E3SYM_ROOT/.tmp"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

cd "$E3SYM_ROOT"
echo "[run_e3sym_train] python train.py --config $CONFIG"
python train.py --config "$CONFIG"
