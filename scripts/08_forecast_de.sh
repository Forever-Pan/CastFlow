#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli forecast \
  --data data/raw/test/EPF_DE_test.csv \
  --anchor-library case_library/EPF_DE/anchor_library.json \
  --memory memory/cross_domain/memory.json \
  --output predictions/de_forecast.csv
