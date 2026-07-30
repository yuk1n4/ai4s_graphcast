#!/usr/bin/env bash
# GraphCast 25km inference entry point
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/inference/run_25km_inference.sh" "$@"
