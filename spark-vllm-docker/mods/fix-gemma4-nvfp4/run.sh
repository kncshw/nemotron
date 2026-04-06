#!/bin/bash
# Replace vLLM's stock gemma4.py with the community NVFP4-aware version
# from bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4. The patched file adds
# load_weights() handling for ModelOpt NVFP4 expert weight names
# (e.g. layers.N.moe.experts.M.down_proj.weight) which the stock loader
# does not recognize and crashes with KeyError on.
#
# The file uses relative imports (from .interfaces / from .utils) so it
# must be installed at the original path inside vllm/model_executor/models/.
set -e

TARGET=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py

if [[ ! -f "$TARGET" ]]; then
    echo "fix-gemma4-nvfp4: ERROR — vLLM gemma4.py not found at $TARGET"
    exit 1
fi

# Back up original once (idempotent across re-applies)
if [[ ! -f "${TARGET}.orig" ]]; then
    cp "$TARGET" "${TARGET}.orig"
fi

cp gemma4_patched.py "$TARGET"
echo "fix-gemma4-nvfp4: replaced $TARGET with NVFP4-aware version"
