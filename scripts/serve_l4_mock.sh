#!/bin/bash
# Phase 1 Mock: Serve Nemotron-3-Nano-4B on L4 (x86_64)
# This is a throwaway script — GB10 uses spark-vllm-docker instead.

set -euo pipefail

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

echo "=== L4 Mock Run: Serving $MODEL ==="
echo "Endpoint: http://${HOST}:${PORT}/v1"

vllm serve "$MODEL" \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 10 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  --host "$HOST" \
  --port "$PORT"
