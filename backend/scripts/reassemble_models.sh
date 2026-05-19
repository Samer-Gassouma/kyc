#!/bin/bash
# Reassemble model files after clone
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="$DIR/weights/ocr_cleaner_finetuned"

if [ -f "$WEIGHTS/model.safetensors" ]; then
    echo "model.safetensors already exists."
    exit 0
fi

# Option 1: compressed single file
if [ -f "$WEIGHTS/model.safetensors.zst" ]; then
    echo "Decompressing model.safetensors.zst..."
    zstd -d "$WEIGHTS/model.safetensors.zst" -o "$WEIGHTS/model.safetensors"
    echo "Done."
    exit 0
fi

# Option 2: split parts
if ls "$WEIGHTS"/model.safetensors.part-* 1> /dev/null 2>&1; then
    echo "Reassembling model.safetensors from split parts..."
    cat "$WEIGHTS"/model.safetensors.part-* > "$WEIGHTS/model.safetensors"
    echo "Done."
    exit 0
fi

echo "No model files found (neither .zst nor split parts)."
