#!/bin/bash
# ============================================================================
# Download Results Script - Run from YOUR LAPTOP to get results from RunPod
# ============================================================================
#
# Usage:
#   1. Set your RunPod IP
#   2. Run: bash download_results.sh
#
# This will download:
#   - evaluation_results/ (JSON files)
#   - evaluation_results.db (SQLite database)
# ============================================================================

# CONFIGURE THIS
RUNPOD_IP="YOUR_RUNPOD_IP_HERE"
RUNPOD_SSH_KEY=""
LOCAL_DIR="./runpod_results_$(date +%Y%m%d_%H%M%S)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check IP is set
if [[ "$RUNPOD_IP" == "YOUR_RUNPOD_IP_HERE" ]]; then
    echo -e "${RED}ERROR: Please edit this script and set RUNPOD_IP${NC}"
    exit 1
fi

echo "========================================"
echo "  Downloading from RunPod: ${RUNPOD_IP}"
echo "========================================"
echo ""

mkdir -p "$LOCAL_DIR"

echo "📥 Downloading evaluation_results/..."
if [[ -n "$RUNPOD_SSH_KEY" ]]; then
    scp -i "$RUNPOD_SSH_KEY" -r "root@${RUNPOD_IP}:/workspace/evaluation_results/" "$LOCAL_DIR/"
else
    scp -r "root@${RUNPOD_IP}:/workspace/evaluation_results/" "$LOCAL_DIR/"
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓${NC} JSON results downloaded"
else
    echo -e "${YELLOW}⚠${NC} evaluation_results/ not found or empty"
fi

echo ""
echo "📥 Downloading SQLite database..."
if [[ -n "$RUNPOD_SSH_KEY" ]]; then
    scp -i "$RUNPOD_SSH_KEY" "root@${RUNPOD_IP}:/workspace/llm_evaluator/results/evaluation_results.db" "$LOCAL_DIR/" 2>/dev/null
else
    scp "root@${RUNPOD_IP}:/workspace/llm_evaluator/results/evaluation_results.db" "$LOCAL_DIR/" 2>/dev/null
fi

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✓${NC} SQLite database downloaded"
else
    echo -e "${YELLOW}⚠${NC} SQLite database not found (evaluations may still be running)"
fi

echo ""
echo "========================================"
echo "  Download Complete!"
echo "========================================"
echo ""
echo "Results saved to: ${LOCAL_DIR}/"
echo ""
ls -la "$LOCAL_DIR/"
echo ""
echo "Next steps:"
echo "  1. Analyze results:"
echo "     python runpod_setup/analyze_results.py ${LOCAL_DIR}/"
echo ""
echo "  2. Or stop your RunPod to save money"
