#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli train-rlvr \
  --dataset-path "${RL_DATASET_PATH:-data/rl/cross_domain_rl.parquet}" \
  --validation-path "${RL_VALIDATION_PATH:-data/rl/cross_domain_validation.parquet}" \
  --model-path "${SFT_MODEL_PATH:-models/sft_cross_domain}" \
  --output-dir "${RL_OUTPUT_DIR:-models/rl_cross_domain}" \
  "$@"
