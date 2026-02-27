#!/bin/bash
# ============================================================================
# Setup Script for RunPod LLM Evaluation
# ============================================================================
# Run this once on a fresh RunPod instance
#
# Usage:
#   cd /workspace/llm_evaluator/runpod_setup
#   bash setup.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  RunPod LLM Evaluator Setup"
echo "========================================"
echo ""

# ----------------------------------------------------------------------------
# Check Prerequisites
# ----------------------------------------------------------------------------

if [[ ! -d "/workspace" ]]; then
    echo "❌ Error: /workspace not found. Are you on RunPod?"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. GPU not available?"
    exit 1
fi

echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
echo ""

# ----------------------------------------------------------------------------
# Install System Dependencies
# ----------------------------------------------------------------------------

echo "📦 Installing system packages..."
if command -v git >/dev/null 2>&1 && command -v git-lfs >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    echo "✓ System packages already installed"
else
    apt-get update -qq
    apt-get install -y -qq git git-lfs ca-certificates curl
fi

# Install git-lfs
git lfs install
echo "✓ System packages installed"
echo ""

# ----------------------------------------------------------------------------
# Setup Project Virtual Environment
# ----------------------------------------------------------------------------

echo "🔧 Setting up project virtual environment..."
cd "$REPO_DIR"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv for this script run
# Note: to keep it active in your shell, run: source setup.sh
source .venv/bin/activate

# Install project dependencies
if [[ ! -f ".venv/.deps_installed" ]]; then
    echo "  Installing project dependencies..."
    pip install -q -r requirements.txt
    touch .venv/.deps_installed
else
    echo "✓ Project dependencies already installed"
fi

# Install vLLM in venv (avoids system conflicts)
if [[ ! -f ".venv/.vllm_installed" ]]; then
    echo "  Installing vLLM (this may take a few minutes)..."
    if [[ -n "${VLLM_PIP_SPEC}" ]]; then
        pip install -q ${VLLM_PIP_SPEC}
    else
        pip install -q vllm==0.15.1
    fi
    pip install -q huggingface-hub hf_transfer pyyaml tqdm
    touch .venv/.vllm_installed
else
    echo "✓ vLLM already installed"
fi

echo "✓ All dependencies installed"
echo ""

# ----------------------------------------------------------------------------
# Create Directory Structure
# ----------------------------------------------------------------------------

echo "📁 Creating directories..."
mkdir -p /workspace/models
mkdir -p /workspace/evaluation_results
mkdir -p /workspace/.cache/huggingface
mkdir -p "$REPO_DIR/results"

echo "✓ Directories created"
echo ""

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Set your Hugging Face token:"
echo "   echo \"HF_TOKEN=your_token\" > /workspace/llm_evaluator/runpod_setup/.env"
echo ""
echo "2. Download models:"
echo "   python3 /workspace/llm_evaluator/runpod_setup/download_models.py"
echo ""
echo "3. Run evaluation:"
echo "   python3 /workspace/llm_evaluator/runpod_setup/evaluate.py"
echo ""
echo "Note:"
echo "  To keep the virtualenv active in your shell:"
echo "  source /workspace/llm_evaluator/.venv/bin/activate"
echo ""
