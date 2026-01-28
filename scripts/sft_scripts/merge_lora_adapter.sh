#!/usr/bin/env bash
set -euo pipefail

# Merge a LoRA adapter into the base model for deployment.
# Override BASE_MODEL, ADAPTER, or OUTPUT via environment variables if needed.
BASE_MODEL=${BASE_MODEL:-./models/Qwen3-0.6B}
ADAPTER=${ADAPTER:-./models/sft_model}
OUTPUT=${OUTPUT:-./models/sft_model_merged}

python merge_adapter.py --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "$OUTPUT"
