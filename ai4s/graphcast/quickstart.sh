#!/usr/bin/env bash
# GraphCast one-click quickstart
# Usage:
#   bash quickstart.sh              # default: infer (quick env check)
#   bash quickstart.sh infer        # inference only
#   bash quickstart.sh train        # training only (4-GPU model parallel)
#   bash quickstart.sh all          # train then infer
#   bash quickstart.sh train --steps 10 --flagos-mode flaggems  # with flags
set -euo pipefail

RELEASE_URL="https://github.com/yuk1n4/ai4s_graphcast/releases/download/v1.0-data-25km/graphcast_25km_data.tar.gz"
DATA_ARCHIVE="graphcast_25km_data.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── helpers ─────────────────────────────────────────────────────────────────

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

usage() {
    cat << 'EOF'
GraphCast one-click quickstart

Usage:
  bash quickstart.sh [MODE] [OPTIONS...]

Modes:
  (default)       Run inference (fast environment check, ~30 sec)
  infer           Run inference only
  train           Run 4-GPU model-parallel training (3 steps)
  all             Run training, then inference

Options (forwarded to train.sh / infer.sh):
  --device DEV         Accelerator: cuda (default) | npu
  --steps N            Training steps (default: 3)
  --flagos-mode MODE   FlagOS mode: off (default) | flaggems
  --flaggems-ops OPS   FlagGems ops: comma-separated list
                       (default: addmm,silu,layer_norm,cat,index,add)

Examples:
  bash quickstart.sh                          # quick inference check
  bash quickstart.sh train --steps 10         # 10-step training
  bash quickstart.sh train --flagos-mode flaggems \
      --flaggems-ops "addmm,silu,layer_norm,index,add"   # FlagGems training
  bash quickstart.sh infer --device npu       # Ascend inference
EOF
    exit 0
}

download_data() {
    if [ -d "data/cases/operational_25km" ] && [ -f "data/checkpoints/google_graphcast_operational_25km_compat_state.pt" ]; then
        return 0
    fi
    bold "Downloading data (~1 GB)..."
    if command -v curl &>/dev/null; then
        curl -L -o "${DATA_ARCHIVE}" "${RELEASE_URL}"
    elif command -v wget &>/dev/null; then
        wget -O "${DATA_ARCHIVE}" "${RELEASE_URL}"
    else
        red "ERROR: curl or wget required to download data."
        exit 1
    fi
    tar -xzf "${DATA_ARCHIVE}"
    rm -f "${DATA_ARCHIVE}"
    green "Data ready."
}

run_infer() {
    bold "=== Inference ==="
    bash scripts/infer.sh "$@"
    green "Inference done."
}

run_train() {
    bold "=== Training ==="
    bash scripts/train.sh "$@"
    green "Training done."
}

# ── main ────────────────────────────────────────────────────────────────────

MODE="${1:-infer}"
case "${MODE}" in
    -h|--help|help) usage ;;
    infer)  shift ;;
    train)  shift ;;
    all)    shift ;;
    *)      ;;  # unknown → pass through as args to default (infer)
esac

download_data

case "${MODE}" in
    infer|"")  run_infer "$@" ;;
    train)     run_train "$@" ;;
    all)       run_train "$@" && run_infer "$@" ;;
    *)         run_infer "${MODE}" "$@" ;;  # treat first arg as flag, run infer
esac
