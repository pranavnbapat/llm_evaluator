#!/bin/bash
# Wrapper script that reads current.env and launches vLLM
# This script is called by supervisord and should not be called directly.
#
# The switch_model.sh script updates /workspace/vllm/current.env
# and restarts the vLLM service via supervisorctl.

set -e

ENV_FILE="/workspace/vllm/current.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Missing ${ENV_FILE}"
  echo "Run switch_model.sh <model_name> first to create the config."
  exit 1
fi

# Auto-export all vars from env file
set -a
source "$ENV_FILE"
set +a

# Validate required vars
[[ -z "${MODEL_PATH:-}" ]] && { echo "ERROR: MODEL_PATH not set in ${ENV_FILE}"; exit 1; }
[[ -z "${API_KEY:-}" ]] && { echo "ERROR: API_KEY not set in ${ENV_FILE}"; exit 1; }

# Build args array
ARGS=(
  serve "$MODEL_PATH"
  --host 0.0.0.0
  --port 8000
  --api-key "$API_KEY"
  --tensor-parallel-size 1
  --dtype auto
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.75}"
)

# Add quant flag only if present
if [[ -n "${QUANT_KIND:-}" ]]; then
  ARGS+=(--quantization "${QUANT_KIND}")
fi

echo "Starting vLLM with model: ${MODEL_NAME:-unknown}"
echo "Path: ${MODEL_PATH}"
echo "Args: ${ARGS[*]}"

exec vllm "${ARGS[@]}"
