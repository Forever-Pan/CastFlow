#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli export-memory-data \
  --memory memory/cross_domain/memory.json \
  --output data/sft/cross_domain_sft.csv \
  "$@"
