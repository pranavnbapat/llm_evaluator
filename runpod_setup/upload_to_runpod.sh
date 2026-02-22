#!/bin/bash
# ============================================================================
# Upload Script - Run from YOUR LAPTOP to upload repo to RunPod
# ============================================================================
#
# Usage:
#   1. Set your RunPod IP
#   2. Run: bash upload_to_runpod.sh
#
# This will upload the entire repo to /workspace/llm_evaluator on RunPod
# ============================================================================

# CONFIGURE THIS
RUNPOD_IP="YOUR_RUNPOD_IP_HERE"  # e.g., 123.45.67.89
RUNPOD_SSH_KEY=""                 # Leave empty if using default SSH key

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check IP is set
if [[ "$RUNPOD_IP" == "YOUR_RUNPOD_IP_HERE" ]]; then
    echo -e "${RED}ERROR: Please edit this script and set RUNPOD_IP${NC}"
    echo "Edit: nano upload_to_runpod.sh"
    exit 1
fi

echo "========================================"
echo "  Uploading to RunPod: ${RUNPOD_IP}"
echo "========================================"
echo ""

# Check we're in the repo root
if [[ ! -f "run_evaluation.py" ]]; then
    echo -e "${RED}ERROR: Not in llm_evaluator repo root${NC}"
    echo "Run this script from the directory containing run_evaluation.py"
    exit 1
fi

echo "📦 Creating archive..."
tar -czf /tmp/llm_evaluator_upload.tar.gz \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.idea' \
    --exclude='*.tar.gz' \
    .

echo -e "${GREEN}✓${NC} Archive created: /tmp/llm_evaluator_upload.tar.gz"
echo ""

echo "📤 Uploading to RunPod..."
if [[ -n "$RUNPOD_SSH_KEY" ]]; then
    scp -i "$RUNPOD_SSH_KEY" /tmp/llm_evaluator_upload.tar.gz "root@${RUNPOD_IP}:/workspace/"
else
    scp /tmp/llm_evaluator_upload.tar.gz "root@${RUNPOD_IP}:/workspace/"
fi

if [[ $? -ne 0 ]]; then
    echo -e "${RED}ERROR: Upload failed${NC}"
    echo "Check:"
    echo "  1. RunPod is running"
    echo "  2. IP address is correct"
    echo "  3. SSH key is configured"
    exit 1
fi

echo -e "${GREEN}✓${NC} Upload complete"
echo ""

echo "📂 Extracting on RunPod..."
if [[ -n "$RUNPOD_SSH_KEY" ]]; then
    ssh -i "$RUNPOD_SSH_KEY" "root@${RUNPOD_IP}" "cd /workspace && tar -xzf llm_evaluator_upload.tar.gz -C llm_evaluator && rm llm_evaluator_upload.tar.gz && ls -la llm_evaluator/"
else
    ssh "root@${RUNPOD_IP}" "cd /workspace && tar -xzf llm_evaluator_upload.tar.gz -C llm_evaluator && rm llm_evaluator_upload.tar.gz && ls -la llm_evaluator/"
fi

if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}✓${NC} Setup complete on RunPod!"
    echo ""
    echo "Next steps:"
    echo "  1. SSH into RunPod:"
    echo "     ssh root@${RUNPOD_IP}"
    echo ""
    echo "  2. Set your tokens:"
    echo "     source /workspace/.env.runpod"
    echo ""
    echo "  3. Run setup:"
    echo "     cd /workspace/llm_evaluator"
    echo "     bash runpod_setup/setup.sh"
    echo ""
    echo "See QUICKSTART.md for full instructions."
else
    echo -e "${RED}ERROR: Extraction failed${NC}"
    exit 1
fi

# Cleanup
rm -f /tmp/llm_evaluator_upload.tar.gz
