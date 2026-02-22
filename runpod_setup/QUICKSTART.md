# RunPod Quickstart Guide

## Your Workflow Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Your Laptop    │────▶│   RunPod GPU     │────▶│  Download to    │
│                 │     │   (A40 48GB)     │     │  Your Laptop    │
│  1. Upload repo │     │                  │     │                 │
│  2. SSH in      │     │  • Download      │     │  • SQLite DB    │
│  3. Run setup   │     │  • Evaluate      │     │  • JSON results │
│                 │     │  • Get results   │     │  • Analysis     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Step-by-Step Setup

### Step 1: Upload This Repository to RunPod

**Option A: SCP (if you have the repo locally)**

```bash
# From your laptop, in the project directory
tar -czf llm_evaluator.tar.gz .
scp llm_evaluator.tar.gz root@YOUR_RUNPOD_IP:/workspace/
```

Then on RunPod:
```bash
ssh root@YOUR_RUNPOD_IP
cd /workspace
tar -xzf llm_evaluator.tar.gz
```

**Option B: Git Clone (recommended)**

```bash
ssh root@YOUR_RUNPOD_IP
cd /workspace
git clone https://github.com/YOUR_USERNAME/llm_evaluator.git
cd llm_evaluator
```

### Step 2: Set Environment Variables

On RunPod, create `/workspace/.env.runpod`:

```bash
cat > /workspace/.env.runpod << 'EOF'
# Required tokens
export HF_TOKEN="hf_your_huggingface_token_here"
export VLLM_API_KEY="sk-any-secure-random-key-you-want"
export OPENAI_API_KEY="sk-openai-your-key-for-evaluation"
EOF
```

Then source it:
```bash
source /workspace/.env.runpod
```

### Step 3: Run Setup

```bash
cd /workspace/llm_evaluator
bash runpod_setup/setup.sh
```

This will:
- Install system packages (supervisor, git, etc.)
- Install vLLM and Python dependencies
- Create directory structure
- Copy all scripts

### Step 4: Download Models

```bash
# This takes 1-3 hours and downloads ~150GB
source /workspace/.env.runpod
bash /workspace/vllm/scripts/download_models.sh
```

**Monitor progress:** The script will show each model downloading. You can detach and check later with:
```bash
tail -f /workspace/models/download.log 2>/dev/null || echo "Check console"
```

### Step 5: Start Services & Run Evaluation

```bash
# 1. Start supervisord
supervisord -c /workspace/ops/supervisord.conf

# 2. Source your env vars
source /workspace/.env.runpod

# 3. Run full batch evaluation
bash /workspace/vllm/scripts/batch_evaluate.sh
```

This will:
1. Start the Evaluator API
2. Cycle through all 5 models
3. Run evaluations for each
4. Save results to `/workspace/evaluation_results/`

**Monitor progress:**
```bash
# Watch overall progress
supervisorctl -c /workspace/ops/supervisord.conf status

# Watch vLLM logs
supervisorctl -c /workspace/ops/supervisord.conf tail -f vllm

# Watch evaluation logs
ls -lt /workspace/evaluation_results/
```

### Step 6: Download Results

Once evaluation is complete (~2-3 hours), download results to your laptop:

**Option A: SCP**

```bash
# From your laptop
scp -r root@YOUR_RUNPOD_IP:/workspace/evaluation_results ./
scp root@YOUR_RUNPOD_IP:/workspace/llm_evaluator/results/evaluation_results.db ./
```

**Option B: Zip and Download**

On RunPod:
```bash
cd /workspace
tar -czf results_$(date +%Y%m%d).tar.gz \
  evaluation_results/ \
  llm_evaluator/results/evaluation_results.db
```

Then from laptop:
```bash
scp root@YOUR_RUNPOD_IP:/workspace/results_$(date +%Y%m%d).tar.gz ./
```

## File Reference

### What Gets Created on RunPod

```
/workspace/
├── models/                          # ~150GB - Your downloaded models
│   ├── qwen3_30b_a3b_awq/
│   ├── eurollm_9b/
│   ├── mistral_small_24b/
│   ├── mixtral_8x7b/
│   └── deepseek_14b/
├── vllm/                            # vLLM runtime
│   ├── scripts/                     # All the scripts
│   ├── logs/
│   └── current.env                  # Active model config
├── llm_evaluator/                   # This repo
│   ├── app/                         # Evaluator API
│   ├── results/
│   │   └── evaluation_results.db   # ⭐ SQLite database
│   └── .venv/
├── evaluation_results/              # ⭐ JSON outputs
│   ├── eurollm_20240220_143022.json
│   ├── qwen3_20240220_150145.json
│   └── ...
└── ops/
    └── supervisord.conf
```

### What You Need to Download Back

| File/Directory | Purpose | Size |
|---------------|---------|------|
| `/workspace/evaluation_results/` | JSON results for each model | ~10-50MB |
| `/workspace/llm_evaluator/results/evaluation_results.db` | SQLite database with all runs | ~100-500MB |

You can **discard** the models (~150GB) - they're too big to download and you can re-download anytime.

## Quick Commands Cheat Sheet

```bash
# Check everything is running
bash /workspace/vllm/scripts/health_check.sh

# View all logs
supervisorctl -c /workspace/ops/supervisord.conf tail -f vllm
supervisorctl -c /workspace/ops/supervisord.conf tail -f evaluator

# Run just one model
bash /workspace/vllm/scripts/batch_evaluate.sh eurollm

# Stop everything
supervisorctl -c /workspace/ops/supervisord.conf shutdown

# Restart a service
supervisorctl -c /workspace/ops/supervisord.conf restart vllm
```

## Troubleshooting

### "Supervisord not running"
```bash
supervisord -c /workspace/ops/supervisord.conf
```

### "Model directory not found"
```bash
# Re-download missing model
bash /workspace/vllm/scripts/download_models.sh
```

### "OOM / Out of Memory"
```bash
# Check GPU usage
nvidia-smi

# View vLLM error
supervisorctl -c /workspace/ops/supervisord.conf tail -50 vllm stderr

# If Mistral Small fails, skip it and run others:
bash /workspace/vllm/scripts/batch_evaluate.sh eurollm qwen3 deepseek mixtral
```

### "Evaluation failed / API error"
```bash
# Check evaluator is running
supervisorctl -c /workspace/ops/supervisord.conf status evaluator

# Restart evaluator
supervisorctl -c /workspace/ops/supervisord.conf restart evaluator
```

## Cost Optimization Tips

1. **Download models once** - They persist in `/workspace/` across pod restarts
2. **Stop the pod** when not evaluating - You only pay for storage (~$0.20/GB/month)
3. **Run batch overnight** - Set it up and let it run, download results in morning

## Next Steps (After Downloading Results)

Once you have the SQLite file locally, you can:

```bash
# On your laptop
cd llm_evaluator
python -c "
import sqlite3
import pandas as pd
conn = sqlite3.connect('evaluation_results.db')
df = pd.read_sql('SELECT * FROM evaluations', conn)
print(df.groupby('model_name')['quality_score'].mean())
"
```

Or analyze with the evaluation tools in this repo.
