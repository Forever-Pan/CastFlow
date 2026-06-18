#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m cli evaluate \
  --csv-file predictions/de_forecast.csv \
  --answer-col answer \
  --ground-truth-col ground_truth \
  --output predictions/de_metrics.csv
