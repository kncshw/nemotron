#!/bin/bash
# Serve a GGUF model via llama-server (OpenAI-compatible API)
#
# Usage:
#   ./serve.sh bugtrace-apex              # default port 9000
#   ./serve.sh bugtrace-apex --port 9001  # custom port
#
# The server exposes /v1/chat/completions and /v1/models — same
# OpenAI-compatible API as vLLM. Open WebUI can connect to it.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"
MODELS_DIR="$SCRIPT_DIR/models"
MODELS_YAML="$SCRIPT_DIR/models.yaml"

MODEL_NAME="${1:?Usage: $0 <model-name> [--port PORT] [--threads N]}"
shift

# Defaults
PORT=9000
THREADS=8
HOST="0.0.0.0"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check llama-server exists
if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "Error: llama-server not found at $LLAMA_SERVER"
    echo "Run ./build.sh first."
    exit 1
fi

# Simple YAML parser
parse_field() {
    local model="$1" field="$2"
    awk -v m="$model" -v f="$field" '
        /^[a-zA-Z]/ { current = $0; gsub(/:.*/, "", current) }
        current == m && $0 ~ "^  " f ":" {
            val = $0; sub(/^[^:]*: */, "", val); gsub(/^"|"$/, "", val); print val
        }
    ' "$MODELS_YAML"
}

FILE=$(parse_field "$MODEL_NAME" "file")
CTX_SIZE=$(parse_field "$MODEL_NAME" "ctx_size")
GPU_LAYERS=$(parse_field "$MODEL_NAME" "gpu_layers")

if [[ -z "$FILE" ]]; then
    echo "Error: model '$MODEL_NAME' not found in models.yaml"
    exit 1
fi

MODEL_PATH="$MODELS_DIR/$MODEL_NAME/$FILE"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: model file not found: $MODEL_PATH"
    echo "Run ./download-model.sh $MODEL_NAME first."
    exit 1
fi

CTX_SIZE="${CTX_SIZE:-8192}"
GPU_LAYERS="${GPU_LAYERS:-99}"

echo "=== Serving $MODEL_NAME ==="
echo "  Model:      $MODEL_PATH"
echo "  Context:    $CTX_SIZE"
echo "  GPU layers: $GPU_LAYERS"
echo "  Threads:    $THREADS"
echo "  Endpoint:   http://$HOST:$PORT/v1/chat/completions"
echo ""

exec "$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --n-gpu-layers "$GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS"
