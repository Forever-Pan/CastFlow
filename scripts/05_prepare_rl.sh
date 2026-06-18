#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli prepare-rl-data \
  --input data/sft/cross_domain_sft.csv \
  --output data/rl/cross_domain_rl.parquet \
  "$@"
