#!/bin/bash
# ============================================================================
# Health Check Script for RunPod vLLM + Evaluator Setup
# ============================================================================
# Checks the status of all services and provides diagnostic information
#
# Usage: bash health_check.sh
# ============================================================================

SUPERVISOR_CONFIG="/workspace/ops/supervisord.conf"
VLLM_URL="http://localhost:8000"
EVALUATOR_URL="http://localhost:8080"

echo "========================================"
echo "  Service Health Check"
echo "========================================"
echo ""

# Check supervisord
echo "Supervisord Status:"
if pgrep -f "supervisord" > /dev/null; then
  echo "  ✓ Running"
  supervisorctl -c "$SUPERVISOR_CONFIG" status 2>/dev/null | while read line; do
    echo "    $line"
  done
else
  echo "  ✗ Not running"
  echo "    Start with: supervisord -c ${SUPERVISOR_CONFIG}"
fi
echo ""

# Check vLLM
echo "vLLM Service (port 8000):"
if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
  echo "  ✓ Healthy"
  # Try to get model info (with auth if key is set)
  model_info=$(curl -s "${VLLM_URL}/v1/models" \
    -H "Authorization: Bearer ${VLLM_API_KEY:-}" 2>/dev/null | head -c 200)
  echo "  Model info: ${model_info}..."
else
  echo "  ✗ Not responding"
fi
echo ""

# Check Evaluator
echo "Evaluator API (port 8080):"
if curl -s "${EVALUATOR_URL}/health" > /dev/null 2>&1; then
  echo "  ✓ Healthy"
else
  echo "  ✗ Not responding"
fi
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
  echo "  $line"
done
echo ""

# Check disk space
echo "Disk Space:"
df -h /workspace 2>/dev/null | tail -1 | awk '{print "  Used: " $3 " / " $2 " (" $5 ")"}'
echo ""

# Check model directories
echo "Model Directories:"
for dir in /workspace/models/*/; do
  if [[ -d "$dir" ]]; then
    name=$(basename "$dir")
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  ✓ ${name}: ${size}"
  fi
done
echo ""

# Recent logs
echo "Recent Logs (last 5 lines):"
echo "  vLLM:"
tail -5 /workspace/vllm/logs/vllm.log 2>/dev/null | sed 's/^/    /' || echo "    No logs"
echo "  Evaluator:"
tail -5 /workspace/logs/evaluator.log 2>/dev/null | sed 's/^/    /' || echo "    No logs"
