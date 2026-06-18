#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli train-rlvr \
  --dataset-path data/rl/cross_domain_rl.parquet \
  --model-path models/sft_cross_domain \
  --output-dir models/rl_cross_domain \
  "$@"
