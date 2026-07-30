#!/usr/bin/env bash
# Single-process multi-NPU model-parallel training for operational 25km GraphCast.
# NOTE: Run inside Docker container. Check NPU availability before running!

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'USAGE'
Usage:
  bash run_25km_model_parallel_train.sh

Required env (set before running):
  ASCEND_RT_VISIBLE_DEVICES=8,9,10,11   # physical NPU IDs (check with npu-smi first!)
  PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

Optional env:
  DEVICES=npu:0,npu:1,npu:2,npu:3
  STEPS=3                                # training steps
  LR=0.000001
  TRAIN_SCOPE=all                        # all or decoder-grid
  MESH2GRID_GRID_PARTITIONS=8            # grid partitions for backward
  ACTIVATION_CHECKPOINTING=1
  GRID2MESH_NODE_CHUNK_SIZE=8192
  MESH2GRID_EDGE_CHUNK_SIZE=2048
  MESH2GRID_NODE_CHUNK_SIZE=8192
  MESH2GRID_DECODER_CHUNK_SIZE=8192
  GRAPHCAST_FLAGOS_MODE=none
  GRAPHCAST_FLAGGEMS_OPS=addmm,silu,layer_norm,cat,index,index_add_,add
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICES="${DEVICES:-npu:0,npu:1,npu:2,npu:3}"
STEPS="${STEPS:-3}"
LR="${LR:-0.000001}"
OPTIMIZER="${OPTIMIZER:-sgd}"
TRAIN_SCOPE="${TRAIN_SCOPE:-all}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
GRID2MESH_NODE_CHUNK_SIZE="${GRID2MESH_NODE_CHUNK_SIZE:-8192}"
MESH2GRID_EDGE_CHUNK_SIZE="${MESH2GRID_EDGE_CHUNK_SIZE:-2048}"
MESH2GRID_NODE_CHUNK_SIZE="${MESH2GRID_NODE_CHUNK_SIZE:-8192}"
MESH2GRID_DECODER_CHUNK_SIZE="${MESH2GRID_DECODER_CHUNK_SIZE:-8192}"
MESH2GRID_GRID_PARTITIONS="${MESH2GRID_GRID_PARTITIONS:-8}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
GRAPHCAST_FLAGOS_MODE="${GRAPHCAST_FLAGOS_MODE:-none}"
GRAPHCAST_FLAGGEMS_OPS="${GRAPHCAST_FLAGGEMS_OPS:-addmm,silu,layer_norm,cat,index,index_add_,add}"

RUN_NAME="${RUN_NAME:-operational_25km_model_parallel_${STEPS}step_${TRAIN_SCOPE}}"
REPORT="${REPORT:-outputs/training/reports/${RUN_NAME}_report.md}"
CHECKPOINT="${CHECKPOINT:-outputs/training/checkpoints/${RUN_NAME}.pt}"

cmd=(
  "${PYTHON_BIN}" scripts/training/lib/train_mp.py
  --devices "${DEVICES}"
  --steps "${STEPS}" --lr "${LR}" --optimizer "${OPTIMIZER}"
  --train-scope "${TRAIN_SCOPE}"
  --activation-checkpointing "${ACTIVATION_CHECKPOINTING}"
  --grid2mesh-node-chunk-size "${GRID2MESH_NODE_CHUNK_SIZE}"
  --mesh2grid-edge-chunk-size "${MESH2GRID_EDGE_CHUNK_SIZE}"
  --mesh2grid-node-chunk-size "${MESH2GRID_NODE_CHUNK_SIZE}"
  --mesh2grid-decoder-chunk-size "${MESH2GRID_DECODER_CHUNK_SIZE}"
  --mesh2grid-grid-partitions "${MESH2GRID_GRID_PARTITIONS}"
  --save-checkpoint "${SAVE_CHECKPOINT}"
  --report "${REPORT}" --checkpoint "${CHECKPOINT}"
  --flagos-mode "${GRAPHCAST_FLAGOS_MODE}"
  --flaggems-ops "${GRAPHCAST_FLAGGEMS_OPS}"
)

printf 'running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
