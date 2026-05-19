#!/bin/bash
# Reassemble split model files after clone
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS="$DIR/weights/ocr_cleaner_finetuned"

if [ -f "$WEIGHTS/model.safetensors" ]; then
    echo "model.safetensors already exists."
    exit 0
fi

if ls "$WEIGHTS"/model.safetensors.part-* 1> /dev/null 2>&1; then
    echo "Reassembling model.safetensors..."
    cat "$WEIGHTS"/model.safetensors.part-* > "$WEIGHTS/model.safetensors"
    echo "Done."
else
    echo "No split parts found."
fi
