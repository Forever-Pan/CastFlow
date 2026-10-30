#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli prepare-rl-data \
  --input "${RL_INPUT_PATH:-data/sft/cross_domain_sft.csv}" \
  --output "${RL_DATASET_PATH:-data/rl/cross_domain_rl.parquet}" \
  "$@"
