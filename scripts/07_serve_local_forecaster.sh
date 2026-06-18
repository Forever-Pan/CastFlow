#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_FORECAST_MODEL_PATH="${LOCAL_FORECAST_MODEL_PATH:-models/sft_cross_domain}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8002}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-castflow-forecast}"
LOCAL_MODEL_API_KEY="${LOCAL_MODEL_API_KEY:-EMPTY}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-14000}"

if [[ ! -d "$LOCAL_FORECAST_MODEL_PATH" ]]; then
  echo "Local forecasting model directory not found: $LOCAL_FORECAST_MODEL_PATH" >&2
  echo "Set LOCAL_FORECAST_MODEL_PATH to a trained SFT/RL model directory before starting this service." >&2
  exit 1
fi

vllm serve "$LOCAL_FORECAST_MODEL_PATH" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --api-key "$LOCAL_MODEL_API_KEY" \
  --trust-remote-code \
  --dtype "$VLLM_DTYPE" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  "$@"
