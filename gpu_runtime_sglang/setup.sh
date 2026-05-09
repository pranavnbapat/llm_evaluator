#!/bin/bash
# ============================================================================
# Setup Script for LLM Evaluation (SGLang)
# ============================================================================
# Run this once on a fresh instance
#
# Usage:
#   cd /workspace/llm_evaluator/gpu_runtime_sglang
#   bash setup.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPT_START_EPOCH="$(date +%s)"
SCRIPT_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"

format_duration() {
    local total="$1"
    local h=$((total / 3600))
    local m=$(((total % 3600) / 60))
    local s=$((total % 60))
    printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
}

step_start() {
    STEP_LABEL="$1"
    STEP_START_EPOCH="$(date +%s)"
    STEP_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "⏱️  [$STEP_LABEL] started at $STEP_START_HUMAN"
}

step_end() {
    local end_epoch
    end_epoch="$(date +%s)"
    local elapsed=$((end_epoch - STEP_START_EPOCH))
    echo "⏱️  [$STEP_LABEL] finished in $(format_duration "$elapsed")"
}

echo "========================================"
echo " LLM Evaluator (SGLang) Setup"
echo "========================================"
echo "Started at: $SCRIPT_START_HUMAN"
echo ""

# ----------------------------------------------------------------------------
# Check Prerequisites
# ----------------------------------------------------------------------------

if [[ ! -d "/workspace" ]]; then
    echo "❌ Error: /workspace not found."
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
elif [[ "$GPU_NAME" == *"L40S"* || "$GPU_NAME" == *"L40"* ]]; then
    GPU_ARCH_FAMILY="ada"
elif [[ "$GPU_NAME" == *"A100"* || "$GPU_NAME" == *"A40"* || "$GPU_NAME" == *"3090"* ]]; then
    GPU_ARCH_FAMILY="ampere"
fi
echo "✓ GPU family detected: $GPU_ARCH_FAMILY"
echo ""

# ----------------------------------------------------------------------------
# Install System Dependencies
# ----------------------------------------------------------------------------

step_start "System packages"
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
step_end
echo ""

# ----------------------------------------------------------------------------
# Install uv
# ----------------------------------------------------------------------------

step_start "uv install"
if command -v uv >/dev/null 2>&1; then
    echo "✓ uv already installed"
else
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv >/dev/null 2>&1; then
        echo "✓ uv installed"
    else
        echo "❌ uv installation finished but uv is not on PATH."
        echo "   Expected location: $HOME/.local/bin/uv"
        exit 1
    fi
fi
step_end
echo ""

# ----------------------------------------------------------------------------
# Setup Project Virtual Environment
# ----------------------------------------------------------------------------

step_start "Python environment and dependencies"
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

PYTHON_BIN="$REPO_DIR/.venv/bin/python"
echo "✓ Using uv for Python package installation"
py_install() {
    uv pip install --python "$PYTHON_BIN" "$@"
}

# Install project dependencies (skip GPU serving stack; installed in SGLang step below)
if [[ ! -f ".venv/.deps_installed" ]]; then
    echo "  Installing project dependencies..."
    TMP_REQ_FILE="$(mktemp)"
    grep -vE '^[[:space:]]*(torch|torchaudio|torchvision|vllm|sglang|flashinfer-python|compressed-tensors)([<>=!~].*)?$' requirements.txt > "$TMP_REQ_FILE"
    CORE_DEPS_START_EPOCH="$(date +%s)"
    echo "  ⏱️  [Core dependencies] installing..."
    py_install -q -r "$TMP_REQ_FILE"
    CORE_DEPS_END_EPOCH="$(date +%s)"
    echo "  ⏱️  [Core dependencies] finished in $(format_duration "$((CORE_DEPS_END_EPOCH - CORE_DEPS_START_EPOCH))")"
    rm -f "$TMP_REQ_FILE"
    touch .venv/.deps_installed
else
    echo "✓ Project dependencies already installed"
fi

# Install SGLang in venv (avoids system conflicts).
# Override the default version with SGLANG_PIP_SPEC if you need a specific build,
# e.g. SGLANG_PIP_SPEC='sglang[all]==0.4.5' bash gpu_runtime_sglang/setup.sh
if [[ ! -f ".venv/.sglang_installed" ]]; then
    echo "  Installing SGLang (this may take a few minutes)..."
    SGLANG_DEPS_START_EPOCH="$(date +%s)"
    # SGLang pulls flashinfer pre-releases on some pinned versions; allow them.
    if [[ -n "${SGLANG_PIP_SPEC}" ]]; then
        py_install -q --prerelease=allow ${SGLANG_PIP_SPEC}
    else
        if [[ "$GPU_ARCH_FAMILY" == "blackwell" ]]; then
            echo "  Detected Blackwell GPU. Installing CUDA 12.8 wheels for PyTorch + SGLang..."
            py_install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
            py_install -q --upgrade --prerelease=allow "sglang[all]" --extra-index-url https://download.pytorch.org/whl/cu128
        else
            py_install -q torch==2.9.1
            py_install -q --prerelease=allow "sglang[all]"
        fi
    fi
    py_install -q huggingface-hub hf_transfer pyyaml tqdm
    SGLANG_DEPS_END_EPOCH="$(date +%s)"
    echo "  ⏱️  [SGLang/runtime dependencies] finished in $(format_duration "$((SGLANG_DEPS_END_EPOCH - SGLANG_DEPS_START_EPOCH))")"
    touch .venv/.sglang_installed
else
    echo "✓ SGLang already installed"
fi

echo "✓ All dependencies installed"
step_end
echo ""

# ----------------------------------------------------------------------------
# Create Directory Structure
# ----------------------------------------------------------------------------

step_start "Directory setup"
echo "📁 Creating directories..."
mkdir -p /workspace/models
mkdir -p /workspace/evaluation_results
mkdir -p /workspace/.cache/huggingface
mkdir -p "$REPO_DIR/results_sglang"

echo "✓ Directories created"
step_end
echo ""

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
SCRIPT_END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
SCRIPT_END_EPOCH="$(date +%s)"
SCRIPT_TOTAL_ELAPSED="$((SCRIPT_END_EPOCH - SCRIPT_START_EPOCH))"
echo "Finished at: $SCRIPT_END_HUMAN"
echo "Total setup time: $(format_duration "$SCRIPT_TOTAL_ELAPSED")"
echo ""
echo "Next steps:"
echo ""
echo "1. Set your Hugging Face token:"
echo "   echo \"HF_TOKEN=your_token\" > /workspace/llm_evaluator/gpu_runtime_sglang/.env"
echo ""
echo "2. Activate virtualenv in your shell:"
echo "   source /workspace/llm_evaluator/.venv/bin/activate"
echo ""
echo "3. Configure Git PAT push auth (one-time per pod):"
echo "   bash /workspace/llm_evaluator/gpu_runtime_sglang/git_bootstrap.sh"
echo ""
echo "4. Download models:"
echo "   python /workspace/llm_evaluator/gpu_runtime_sglang/download_models.py"
echo ""
echo "5. Run context evaluation:"
echo "   Foreground:  python /workspace/llm_evaluator/gpu_runtime_sglang/evaluate_context.py"
echo "   Background:  bash /workspace/llm_evaluator/gpu_runtime_sglang/run_evaluate_context_background.sh"
echo ""
echo "6. Run scoring:"
echo "   Foreground:  python /workspace/llm_evaluator/gpu_runtime_sglang/evaluate_context_results.py"
echo "   Background:  bash /workspace/llm_evaluator/gpu_runtime_sglang/run_evaluate_context_results_background.sh"
echo ""
