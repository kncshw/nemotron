#!/bin/bash
# Serve Nemotron-3-Nano-30B-A3B-FP8 on GB10 via spark-vllm-docker
#
# This model is a Mamba2-Transformer Hybrid MoE:
#   - 30B total params, ~3.5B active per token
#   - FP8 weights (~30 GB) — fits easily in 128 GB unified memory
#   - 256K default context
#
# Usage:
#   ./serve_nano30b.sh              # default setup
#   ./serve_nano30b.sh --setup      # first-time: build container + download model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPARK_DIR="${SCRIPT_DIR}/../spark-vllm-docker"

echo "=== GB10 Single-Node: Nemotron-3-Nano-30B-A3B-FP8 ==="
cd "$SPARK_DIR"
./run-recipe.sh nemotron-3-nano-fp8 --solo "$@"
