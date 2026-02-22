#!/usr/bin/env bash
# Download HF models into deterministic local directories under /workspace/models
# Run: export HF_TOKEN=... ; bash download_models.sh
#
# This script downloads models to local directories with predictable paths,
# making it easy to switch between models without re-downloading.

set -euo pipefail

BASE_DIR="/workspace/models"
HF_HOME="/workspace/.cache/huggingface"
export HF_HOME HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1

[[ -z "${HF_TOKEN:-}" ]] && { echo "ERROR: HF_TOKEN not set"; exit 1; }

apt-get update -y && apt-get install -y git git-lfs ca-certificates
git lfs install
python3 -m pip install -U "huggingface_hub[cli]" hf_transfer

mkdir -p "${BASE_DIR}"

# Use HF_TOKEN env, no persistent login
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

declare -A MODELS=(
  ["qwen3_30b_a3b_awq"]="Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ"
  ["eurollm_9b"]="utter-project/EuroLLM-9B-Instruct-2512"
  ["mistral_small_24b"]="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  ["mixtral_8x7b"]="mistralai/Mixtral-8x7B-Instruct-v0.1"
  ["deepseek_14b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
)

for name in "${!MODELS[@]}"; do
  repo="${MODELS[$name]}"
  target="${BASE_DIR}/${name}"
  
  echo "📥 ${name} -> ${target}"
  
  huggingface-cli download "${repo}" \
    --local-dir "${target}" \
    --local-dir-use-symlinks False \
    --resume-download \
    --exclude "*.h5" "*.msgpack" "*.ot"

  echo "✅ ${name}: $(du -sh "${target}" | cut -f1)"
done

echo ""
echo "All models downloaded to ${BASE_DIR}"
