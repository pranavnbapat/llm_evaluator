#!/bin/bash
# ============================================================================
# Batch Evaluation Script for Multi-Model LLM Evaluation on RunPod
# ============================================================================
# This script automates the full evaluation workflow:
#   1. Cycles through all configured models
#   2. Switches vLLM to each model
#   3. Waits for vLLM to be ready
#   4. Runs evaluation via the Evaluator API
#   5. Collects and organizes results
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export VLLM_API_KEY="sk_..."
#   export OPENAI_API_KEY="sk-..."  # For judge/evaluation
#   bash batch_evaluate.sh [model1] [model2] ...
#
#   Or evaluate all models:
#   bash batch_evaluate.sh
#
#   Or specific models only:
#   bash batch_evaluate.sh eurollm qwen3 deepseek
# ============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="/workspace"
SUPERVISOR_CONFIG="/workspace/ops/supervisord.conf"
RESULTS_DIR="/workspace/evaluation_results"
VLLM_URL="http://localhost:8000"
EVALUATOR_URL="http://localhost:8080"
VLLM_HEALTH_URL="${VLLM_URL}/health"

# Default: evaluate all models
# Order matters: start with smaller/faster models for testing
ALL_MODELS=("eurollm" "qwen3" "deepseek" "mixtral" "mistral-small")

# Parse command line arguments
if [[ $# -gt 0 ]]; then
  MODELS_TO_EVAL=("$@")
else
  MODELS_TO_EVAL=("${ALL_MODELS[@]}")
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

check_prerequisites() {
  log_info "Checking prerequisites..."
  
  # Check environment variables
  [[ -z "${HF_TOKEN:-}" ]] && { log_error "HF_TOKEN not set"; exit 1; }
  [[ -z "${VLLM_API_KEY:-}" ]] && { log_warn "VLLM_API_KEY not set, using default"; }
  
  # Check supervisord is running
  if ! pgrep -f "supervisord" > /dev/null; then
    log_error "Supervisord is not running. Start it with:"
    log_error "  supervisord -c ${SUPERVISOR_CONFIG}"
    exit 1
  fi
  
  # Check that models directory exists and has content
  if [[ ! -d "${WORKSPACE_DIR}/models" ]] || [[ -z "$(ls -A ${WORKSPACE_DIR}/models 2>/dev/null)" ]]; then
    log_error "No models found in ${WORKSPACE_DIR}/models"
    log_error "Run download_models.sh first"
    exit 1
  fi
  
  # Note: Per-model directory validation is handled by switch_model.sh
  
  # Create results directory
  mkdir -p "$RESULTS_DIR"
  
  log_success "Prerequisites check passed"
}

# ============================================================================
# Service Management
# ============================================================================

start_evaluator() {
  log_info "Starting Evaluator API..."
  supervisorctl -c "$SUPERVISOR_CONFIG" start evaluator
  
  # Wait for evaluator to be ready
  log_info "Waiting for Evaluator API to be ready..."
  local retries=30
  while [[ $retries -gt 0 ]]; do
    if curl -s "${EVALUATOR_URL}/health" > /dev/null 2>&1; then
      log_success "Evaluator API is ready"
      return 0
    fi
    sleep 2
    ((retries--))
  done
  
  log_error "Evaluator API failed to start"
  exit 1
}

wait_for_vllm() {
  local model=$1
  log_info "Waiting for vLLM (${model}) to be ready..."
  
  local retries=60
  while [[ $retries -gt 0 ]]; do
    # Check vLLM health endpoint
    if curl -s "${VLLM_HEALTH_URL}" > /dev/null 2>&1; then
      # Also verify it's responding to chat completions
      if curl -s -X POST "${VLLM_URL}/v1/chat/completions" \
        -H "Authorization: Bearer ${VLLM_API_KEY:-}" \
        -H "Content-Type: application/json" \
        -d '{"model":"default","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' \
        > /dev/null 2>&1; then
        log_success "vLLM (${model}) is ready"
        return 0
      fi
    fi
    
    # Check if process died
    if ! supervisorctl -c "$SUPERVISOR_CONFIG" status vllm | grep -q "RUNNING"; then
      log_error "vLLM process died during startup. Check logs:"
      log_error "  supervisorctl -c ${SUPERVISOR_CONFIG} tail vllm stderr"
      return 1
    fi
    
    echo -n "."
    sleep 5
    ((retries--))
  done
  
  log_error "vLLM failed to become ready within timeout"
  return 1
}

stop_vllm() {
  log_info "Stopping vLLM..."
  supervisorctl -c "$SUPERVISOR_CONFIG" stop vllm 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 3
}

# ============================================================================
# Evaluation
# ============================================================================

run_evaluation() {
  local model=$1
  local timestamp=$(date +%Y%m%d_%H%M%S)
  local result_file="${RESULTS_DIR}/${model}_${timestamp}.json"
  
  log_info "Starting evaluation for: ${model}"
  
  # Prepare request payload
  local payload=$(cat <<EOF
{
  "model_name": "${model}",
  "model_url": "${VLLM_URL}/v1/chat/completions",
  "api_key": "${VLLM_API_KEY:-}",
  "languages": ["all"],
  "num_runs": 3,
  "temperature": 0.0
}
EOF
)
  
  log_info "Sending evaluation request..."
  
  # Run evaluation via API
  local response
  if response=$(curl -s -X POST "${EVALUATOR_URL}/evaluate" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>&1); then
    
    # Save raw response
    echo "$response" > "$result_file"
    
    # Check for API errors
    if echo "$response" | grep -q '"error"'; then
      log_error "Evaluation API returned error for ${model}"
      log_error "Response saved to: ${result_file}"
      return 1
    fi
    
    log_success "Evaluation completed for ${model}"
    log_info "Results saved to: ${result_file}"
    
    # Also fetch the report
    sleep 2
    local report_file="${RESULTS_DIR}/${model}_${timestamp}_report.json"
    curl -s "${EVALUATOR_URL}/report/${model}" > "$report_file" 2>&1 || true
    
    return 0
  else
    log_error "Failed to communicate with Evaluator API"
    log_error "Curl output: ${response}"
    return 1
  fi
}

# ============================================================================
# Main Workflow
# ============================================================================

main() {
  echo "========================================"
  echo "  LLM Batch Evaluation on RunPod"
  echo "========================================"
  echo ""
  log_info "Models to evaluate: ${MODELS_TO_EVAL[*]}"
  log_info "Results directory: ${RESULTS_DIR}"
  echo ""
  
  # Pre-flight checks
  check_prerequisites
  
  # Start evaluator service
  start_evaluator
  
  # Track results
  local -a successful_models=()
  local -a failed_models=()
  
  # Evaluate each model
  for model in "${MODELS_TO_EVAL[@]}"; do
    echo ""
    echo "========================================"
    log_info "Processing model: ${model}"
    echo "========================================"
    
    # Switch to this model
    log_info "Switching vLLM to ${model}..."
    if ! "${SCRIPT_DIR}/switch_model.sh" "$model"; then
      log_error "Failed to switch to model: ${model}"
      failed_models+=("$model")
      continue
    fi
    
    # Wait for vLLM to be ready
    if ! wait_for_vllm "$model"; then
      log_error "vLLM failed to start for model: ${model}"
      failed_models+=("$model")
      stop_vllm
      continue
    fi
    
    # Run evaluation
    if run_evaluation "$model"; then
      successful_models+=("$model")
    else
      failed_models+=("$model")
    fi
    
    # Cool down between models (GPU thermal management)
    log_info "Cooling down for 10 seconds..."
    sleep 10
  done
  
  # Summary
  echo ""
  echo "========================================"
  echo "  Evaluation Summary"
  echo "========================================"
  echo ""
  
  if [[ ${#successful_models[@]} -gt 0 ]]; then
    log_success "Successful evaluations (${#successful_models[@]}):"
    for model in "${successful_models[@]}"; do
      echo "  ✓ ${model}"
    done
  fi
  
  if [[ ${#failed_models[@]} -gt 0 ]]; then
    log_error "Failed evaluations (${#failed_models[@]}):"
    for model in "${failed_models[@]}"; do
      echo "  ✗ ${model}"
    done
  fi
  
  echo ""
  log_info "All results saved to: ${RESULTS_DIR}"
  echo ""
  
  # Stop vLLM at the end (keep evaluator running for queries)
  log_info "Stopping vLLM (evaluator still running)..."
  stop_vllm
  
  # Return appropriate exit code
  if [[ ${#failed_models[@]} -gt 0 ]]; then
    exit 1
  fi
  exit 0
}

# Run main function
main "$@"
