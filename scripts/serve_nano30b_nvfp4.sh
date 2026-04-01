#!/bin/bash
# Serve Nemotron-3-Nano-30B-A3B-NVFP4 on GB10 via spark-vllm-docker
#
# This model is a Mamba2-Transformer Hybrid MoE:
#   - 30B total params, ~3.5B active per token
#   - NVFP4 weights (~15 GB) — fastest option on GB10
#   - 256K default context
#
# Usage:
#   ./serve_nano30b_nvfp4.sh              # default setup
#   ./serve_nano30b_nvfp4.sh --setup      # first-time: build container + download model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPARK_DIR="${SCRIPT_DIR}/../spark-vllm-docker"

echo "=== GB10 Single-Node: Nemotron-3-Nano-30B-A3B-NVFP4 ==="
cd "$SPARK_DIR"
./run-recipe.sh nemotron-3-nano-nvfp4 --solo "$@"
