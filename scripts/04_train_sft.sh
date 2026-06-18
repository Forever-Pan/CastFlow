#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-32588}"
SFT_DATASET_PATH="${SFT_DATASET_PATH:-data/sft/cross_domain_sft.csv}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-../models/Qwen3-4B}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-models/sft_cross_domain}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-14000}"

torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" -m cli train-sft \
  --dataset-path "$SFT_DATASET_PATH" \
  --model-path "$BASE_MODEL_PATH" \
  --output-dir "$SFT_OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation "$GRADIENT_ACCUMULATION" \
  --learning-rate "$LEARNING_RATE" \
  --num-epochs "$NUM_EPOCHS" \
  --max-length "$MAX_LENGTH" \
  "$@"
