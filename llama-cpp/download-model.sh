#!/bin/bash
# Download a GGUF model from models.yaml
#
# Usage:
#   ./download-model.sh bugtrace-apex

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
MODELS_YAML="$SCRIPT_DIR/models.yaml"

MODEL_NAME="${1:?Usage: $0 <model-name>  (see models.yaml for available models)}"

# Simple YAML parser — extract repo and file for the given model name
parse_field() {
    local model="$1" field="$2"
    awk -v m="$model" -v f="$field" '
        /^[a-zA-Z]/ { current = $0; gsub(/:.*/, "", current) }
        current == m && $0 ~ "^  " f ":" {
            val = $0; sub(/^[^:]*: */, "", val); gsub(/^"|"$/, "", val); print val
        }
    ' "$MODELS_YAML"
}

REPO=$(parse_field "$MODEL_NAME" "repo")
FILE=$(parse_field "$MODEL_NAME" "file")

if [[ -z "$REPO" || -z "$FILE" ]]; then
    echo "Error: model '$MODEL_NAME' not found in models.yaml"
    echo ""
    echo "Available models:"
    grep -E '^[a-zA-Z]' "$MODELS_YAML" | sed 's/:$//'
    exit 1
fi

DEST="$MODELS_DIR/$MODEL_NAME"

if [[ -f "$DEST/$FILE" ]]; then
    echo "Model already downloaded: $DEST/$FILE"
    exit 0
fi

echo "=== Downloading $MODEL_NAME ==="
echo "  Repo: $REPO"
echo "  File: $FILE"
echo "  Dest: $DEST/"
echo ""

huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST"

echo ""
echo "=== Download complete ==="
echo "Model file: $DEST/$FILE"
echo ""
echo "Serve it: ./serve.sh $MODEL_NAME"
