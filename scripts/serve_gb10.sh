#!/bin/bash
# Phase 3: Serve Nemotron-3-Super-120B-A12B-NVFP4 on GB10 via spark-vllm-docker
# Run from the spark-vllm-docker directory.
#
# Usage:
#   Single node (NVFP4):  ./serve_gb10.sh
#   Dual node (FP8):      ./serve_gb10.sh --fp8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPARK_DIR="${SCRIPT_DIR}/../spark-vllm-docker"

if [[ "${1:-}" == "--fp8" ]]; then
    echo "=== GB10 Dual-Node: FP8 mode (TP=2) ==="
    cd "$SPARK_DIR"
    ./run-recipe.sh nemotron-3-super-nvfp4 --setup  # TODO: create FP8 recipe in Phase 5
    echo "NOTE: FP8 recipe not yet created — using NVFP4 as placeholder"
else
    echo "=== GB10 Single-Node: NVFP4 mode ==="
    cd "$SPARK_DIR"
    ./run-recipe.sh nemotron-3-super-nvfp4 --solo --setup
fi
