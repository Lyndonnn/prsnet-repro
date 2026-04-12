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
PLANE_SCALE="${PLANE_SCALE:-0.35}"
PLANE_ALPHA="${PLANE_ALPHA:-0.18}"
MESH_ALPHA="${MESH_ALPHA:-0.9}"
MESH_EDGES="${MESH_EDGES:-0}"
ZOOM="${ZOOM:-1.15}"
RENDER_MODE="${RENDER_MODE:-points}"
SPLIT_PLANES="${SPLIT_PLANES:-0}"
SHOW_REFLECTION="${SHOW_REFLECTION:-0}"
MAX_POINTS="${MAX_POINTS:-1000}"
PAPER_STYLE="${PAPER_STYLE:-0}"
PLANE_IDS="${PLANE_IDS:-all}"
POINT_SIZE="${POINT_SIZE:-5}"
POINT_ALPHA="${POINT_ALPHA:-0.7}"
VIEW_ELEV="${VIEW_ELEV:-22}"
VIEW_AZIM="${VIEW_AZIM:-38}"
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
  --plane-scale "$PLANE_SCALE"
  --plane-alpha "$PLANE_ALPHA"
  --mesh-alpha "$MESH_ALPHA"
  --zoom "$ZOOM"
  --render-mode "$RENDER_MODE"
  --max-points "$MAX_POINTS"
  --plane-ids "$PLANE_IDS"
  --point-size "$POINT_SIZE"
  --point-alpha "$POINT_ALPHA"
  --view-elev "$VIEW_ELEV"
  --view-azim "$VIEW_AZIM"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

if [[ "$MESH_EDGES" == "1" || "$MESH_EDGES" == "true" ]]; then
  args+=(--mesh-edges)
fi

if [[ "$SPLIT_PLANES" == "1" || "$SPLIT_PLANES" == "true" ]]; then
  args+=(--split-planes)
fi

if [[ "$SHOW_REFLECTION" == "1" || "$SHOW_REFLECTION" == "true" ]]; then
  args+=(--show-reflection)
fi

if [[ "$PAPER_STYLE" == "1" || "$PAPER_STYLE" == "true" ]]; then
  args+=(--paper-style)
fi

echo "[run_visualize] ${args[*]} $*"
"${args[@]}" "$@"
