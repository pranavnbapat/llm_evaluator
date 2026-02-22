#!/bin/bash
# ============================================================================
# One-Time Setup Script for RunPod LLM Multi-Model Evaluation
# ============================================================================
# This script sets up the complete evaluation environment on a fresh RunPod
# instance with persistent storage at /workspace
#
# Prerequisites:
#   - RunPod instance with A40 48GB GPU or similar
#   - Persistent storage mounted at /workspace
#   - Environment variables set (HF_TOKEN, VLLM_API_KEY, OPENAI_API_KEY)
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export VLLM_API_KEY="sk_..."
#   export OPENAI_API_KEY="sk-..."
#   bash setup.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/workspace"

echo "========================================"
echo "  RunPod LLM Evaluation Setup"
echo "========================================"
echo ""

# ----------------------------------------------------------------------------
# Pre-flight Checks
# ----------------------------------------------------------------------------

echo "🔍 Checking prerequisites..."

# Check for required environment variables
[[ -z "${HF_TOKEN:-}" ]] && { echo "❌ HF_TOKEN not set"; exit 1; }
[[ -z "${VLLM_API_KEY:-}" ]] && { echo "⚠️  VLLM_API_KEY not set, using default insecure key"; export VLLM_API_KEY="change-me-insecure"; }
[[ -z "${OPENAI_API_KEY:-}" ]] && { echo "⚠️  OPENAI_API_KEY not set - evaluation will fail"; }

# Check we're on RunPod with /workspace
if [[ ! -d "$WORKSPACE_DIR" ]]; then
  echo "❌ /workspace directory not found. Are you on RunPod?"
  exit 1
fi

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
  echo "❌ nvidia-smi not found. GPU not available?"
  exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# ----------------------------------------------------------------------------
# System Dependencies
# ----------------------------------------------------------------------------

echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq supervisor git git-lfs ca-certificates curl jq
echo "✓ System dependencies installed"
echo ""

# ----------------------------------------------------------------------------
# Python Environment
# ----------------------------------------------------------------------------

echo "🐍 Setting up Python environment..."

# Create virtual environment for llm_evaluator
if [[ ! -d "${WORKSPACE_DIR}/llm_evaluator/.venv" ]]; then
  echo "  Creating virtual environment..."
  cd "${WORKSPACE_DIR}"
  
  # Check repo is present
  if [[ ! -d "${WORKSPACE_DIR}/llm_evaluator" ]]; then
    echo "❌ Repo not found at ${WORKSPACE_DIR}/llm_evaluator"
    echo "Clone it first: git clone <your-repo-url> ${WORKSPACE_DIR}/llm_evaluator"
    exit 1
  fi
  
  cd "${WORKSPACE_DIR}/llm_evaluator"
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -r requirements.txt
  echo "✓ Virtual environment created"
else
  echo "✓ Virtual environment already exists"
fi

# Install vLLM system-wide (for the vLLM service)
echo "  Installing vLLM..."
pip install -U vllm huggingface-hub hf_transfer
echo "✓ Python environment ready"
echo ""

# ----------------------------------------------------------------------------
# Directory Structure
# ----------------------------------------------------------------------------

echo "📁 Creating directory structure..."
mkdir -p "${WORKSPACE_DIR}/models"
mkdir -p "${WORKSPACE_DIR}/vllm/{scripts,logs}"
mkdir -p "${WORKSPACE_DIR}/ops"
mkdir -p "${WORKSPACE_DIR}/logs"
mkdir -p "${WORKSPACE_DIR}/evaluation_results"
mkdir -p "${WORKSPACE_DIR}/.cache/huggingface"
echo "✓ Directories created"
echo ""

# ----------------------------------------------------------------------------
# Copy Scripts and Configs
# ----------------------------------------------------------------------------

echo "📋 Installing scripts and configurations..."

# Copy vLLM scripts
cp "${SCRIPT_DIR}/scripts/"*.sh "${WORKSPACE_DIR}/vllm/scripts/"
chmod +x "${WORKSPACE_DIR}/vllm/scripts/"*.sh

# Copy supervisord config
cp "${SCRIPT_DIR}/config/supervisord.conf" "${WORKSPACE_DIR}/ops/supervisord.conf"

# Copy environment template
cp "${SCRIPT_DIR}/.env.runpod.example" "${WORKSPACE_DIR}/.env.runpod.example"

echo "✓ Scripts and configs installed"
echo ""

# ----------------------------------------------------------------------------
# Initial Configuration
# ----------------------------------------------------------------------------

echo "⚙️  Creating initial configuration..."

# Create initial vLLM env (will be updated by switch_model.sh)
mkdir -p "${WORKSPACE_DIR}/vllm"
cat > "${WORKSPACE_DIR}/vllm/current.env" << EOF
# Initial placeholder - will be updated by switch_model.sh
MODEL_NAME=none
MODEL_PATH=/workspace/models/eurollm_9b
QUANT_FLAG=
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.75
API_KEY=${VLLM_API_KEY}
EOF

echo "✓ Initial configuration created"
echo ""

# ----------------------------------------------------------------------------
# Download Models (Optional - can be done separately)
# ----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Download models (this takes 1-3 hours):"
echo "   export HF_TOKEN=\"${HF_TOKEN}\""
echo "   bash /workspace/vllm/scripts/download_models.sh"
echo ""
echo "2. Start supervisord:"
echo "   supervisord -c /workspace/ops/supervisord.conf"
echo ""
echo "3. Run batch evaluation:"
echo "   export VLLM_API_KEY=\"${VLLM_API_KEY}\""
echo "   export OPENAI_API_KEY=\"${OPENAI_API_KEY:-}\""
echo "   bash /workspace/vllm/scripts/batch_evaluate.sh"
echo ""
echo "For detailed documentation, see:"
echo "   cat ${SCRIPT_DIR}/README.md"
echo ""

# Ask if user wants to download models now
if [[ "${DOWNLOAD_MODELS_NOW:-}" == "yes" ]]; then
  echo "⬇️  Starting model download..."
  bash "${WORKSPACE_DIR}/vllm/scripts/download_models.sh"
else
  read -p "Download models now? (takes 1-3 hours) [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "${WORKSPACE_DIR}/vllm/scripts/download_models.sh"
  fi
fi
