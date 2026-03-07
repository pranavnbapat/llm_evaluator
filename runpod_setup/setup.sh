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

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)"
GPU_ARCH_FAMILY="ampere_or_hopper"
if [[ "$GPU_NAME" == *"B200"* || "$GPU_NAME" == *"GB200"* || "$GPU_NAME" == *"Blackwell"* ]]; then
    GPU_ARCH_FAMILY="blackwell"
elif [[ "$GPU_NAME" == *"H200"* || "$GPU_NAME" == *"H100"* || "$GPU_NAME" == *"H20"* ]]; then
    GPU_ARCH_FAMILY="hopper"
elif [[ "$GPU_NAME" == *"A100"* || "$GPU_NAME" == *"A40"* ]]; then
    GPU_ARCH_FAMILY="ampere"
fi
echo "✓ GPU family detected: $GPU_ARCH_FAMILY"
echo ""

# ----------------------------------------------------------------------------
# Install System Dependencies
# ----------------------------------------------------------------------------

echo "📦 Installing system packages..."
PACKAGES=(
    git
    git-lfs
    ca-certificates
    curl
    nano
    ripgrep
    jq
    htop
    tmux
    less
    unzip
    zip
    tree
)

NEED_INSTALL=0
for pkg in "${PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        NEED_INSTALL=1
        break
    fi
done

if [[ $NEED_INSTALL -eq 0 ]]; then
    echo "✓ System packages already installed"
else
    apt-get update -qq
    apt-get install -y -qq "${PACKAGES[@]}"
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
    TMP_REQ_FILE="$(mktemp)"
    # Install core deps first; GPU serving stack is installed in the vLLM step below.
    grep -vE '^[[:space:]]*(torch|torchaudio|torchvision|vllm|flashinfer-python|compressed-tensors)([<>=!~].*)?$' requirements.txt > "$TMP_REQ_FILE"
    pip install -q -r "$TMP_REQ_FILE"
    rm -f "$TMP_REQ_FILE"
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
        if [[ "$GPU_ARCH_FAMILY" == "blackwell" ]]; then
            echo "  Detected Blackwell GPU. Installing CUDA 12.8 wheels for PyTorch + vLLM..."
            pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
            pip install -q --upgrade vllm --extra-index-url https://download.pytorch.org/whl/cu128
        else
            pip install -q torch==2.9.1
            pip install -q vllm==0.17.0
        fi
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
echo "2. Activate virtualenv in your shell:"
echo "   source /workspace/llm_evaluator/.venv/bin/activate"
echo ""
echo "3. Configure Git PAT push auth (one-time per pod):"
echo "   bash /workspace/llm_evaluator/runpod_setup/git_bootstrap.sh"
echo ""
echo "4. Download models:"
echo "   python /workspace/llm_evaluator/runpod_setup/download_models.py"
echo ""
echo "5. Run context evaluation:"
echo "   Foreground:  python /workspace/llm_evaluator/runpod_setup/evaluate_context.py"
echo "   Background:  bash /workspace/llm_evaluator/runpod_setup/run_evaluate_context_background.sh"
echo ""
echo "6. Run scoring:"
echo "   Foreground:  python /workspace/llm_evaluator/runpod_setup/evaluate_context_results.py"
echo "   Background:  bash /workspace/llm_evaluator/runpod_setup/run_evaluate_context_results_background.sh"
echo ""
