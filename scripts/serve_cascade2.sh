#!/bin/bash
# Serve Nemotron-Cascade-2-30B-A3B on GB10 via spark-vllm-docker
#
# BF16 model (~63 GB weights) — single GB10 node (128 GB unified memory)
# Supports thinking mode and instruct mode, tool calling
#
# Usage:
#   ./serve_cascade2.sh              # default setup
#   ./serve_cascade2.sh --setup      # first-time: build container + download model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPARK_DIR="${SCRIPT_DIR}/../spark-vllm-docker"

echo "=== GB10 Single-Node: Nemotron-Cascade-2-30B-A3B (BF16) ==="
cd "$SPARK_DIR"
./run-recipe.sh nemotron-cascade-2 --solo "$@"
