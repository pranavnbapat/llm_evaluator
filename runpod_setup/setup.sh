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
apt-get update -qq
apt-get install -y -qq git git-lfs ca-certificates curl

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

# Install project dependencies
echo "  Installing project dependencies..."
.venv/bin/pip install -q -r requirements.txt

# Install vLLM in venv (avoids system conflicts)
echo "  Installing vLLM (this may take a few minutes)..."
.venv/bin/pip install -q vllm huggingface-hub hf_transfer pyyaml tqdm

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
echo "1. Edit config.yaml with your tokens:"
echo "   nano /workspace/llm_evaluator/runpod_setup/config.yaml"
echo ""
echo "2. Download models:"
echo "   python3 /workspace/llm_evaluator/runpod_setup/download_models.py"
echo ""
echo "3. Run evaluation:"
echo "   python3 /workspace/llm_evaluator/runpod_setup/evaluate.py"
echo ""
