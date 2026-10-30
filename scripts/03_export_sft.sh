#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli export-memory-data \
  --memory "${MEMORY_PATH:-memory/cross_domain/memory.json}" \
  --output "${SFT_DATASET_PATH:-data/sft/cross_domain_sft.csv}" \
  "$@"
