#!/bin/bash
# Build llama.cpp from source for GB10 (sm_121, aarch64)
# Run once on the GB10 where you want to serve GGUF models.
#
# Prerequisites: git, cmake 3.14+, CUDA toolkit (13.0+)
#
# Usage:
#   ./build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"

if [[ -d "$LLAMA_DIR" ]]; then
    echo "llama.cpp already cloned at $LLAMA_DIR"
    echo "To rebuild, delete it first: rm -rf $LLAMA_DIR"
    exit 1
fi

echo "=== Cloning llama.cpp ==="
git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"

echo "=== Building with CUDA (sm_121 for GB10) ==="
mkdir -p "$LLAMA_DIR/build"
cd "$LLAMA_DIR/build"
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DGGML_CUDA_F16=ON
make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Server binary: $LLAMA_DIR/build/bin/llama-server"
echo ""
echo "Next steps:"
echo "  1. Download a GGUF model:  ./download-model.sh bugtrace-apex"
echo "  2. Serve it:               ./serve.sh bugtrace-apex"
