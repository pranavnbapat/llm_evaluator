# RunPod LLM Multi-Model Evaluation Setup

Complete production-ready setup for evaluating multiple LLMs on RunPod with A40 48GB VRAM.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RunPod Instance                                │
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐    │
│  │   supervisord   │────▶│   vLLM Service      │────▶│  Models         │    │
│  │   (manager)     │     │   Port: 8000        │     │  /workspace/    │    │
│  │                 │     │                     │     │  models/        │    │
│  │                 │────▶│   Evaluator API     │     │                 │    │
│  │                 │     │   Port: 8080        │     │  • eurollm_9b   │    │
│  └─────────────────┘     └─────────────────────┘     │  • qwen3_30b_a3b_awq │
│           │                                          │  • deepseek_14b │    │
│           │                                          │  • mixtral_8x7b │    │
│           ▼                                          │  • mistral_24b  │    │
│  ┌─────────────────────────────────────────────────┐ └─────────────────┘    │
│  │           Batch Evaluation Script               │                        │
│  │   • Cycles through models                       │                        │
│  │   • Switches vLLM for each                      │                        │
│  │   • Runs evaluations                            │                        │
│  │   • Saves results to /workspace/eval_results    │                        │
│  └─────────────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start (5 Steps)

### Step 1: Create RunPod Instance

1. Go to [runpod.io](https://runpod.io)
2. Deploy a GPU pod with:
   - **GPU**: RTX A40 (48GB VRAM) or similar
   - **Storage**: At least 200GB persistent volume at `/workspace`
   - **Template**: PyTorch or CUDA
3. SSH into your instance

### Step 2: Set Environment Variables

```bash
# Required: HuggingFace token (for downloading models)
export HF_TOKEN="hf_your_huggingface_token_here"

# Required: Secure key for vLLM API
export VLLM_API_KEY="sk-your-secure-random-key"

# Required: OpenAI API key (for evaluation/judging)
export OPENAI_API_KEY="sk-openai-your-key-here"
```

### Step 3: Clone and Setup

```bash
cd /workspace

# Clone this repository
git clone https://github.com/yourusername/llm_evaluator.git
cd llm_evaluator

# Run setup (installs dependencies, creates directories)
bash runpod_setup/setup.sh
```

### Step 4: Download Models

```bash
# This downloads ~120-150GB of models (takes 1-3 hours)
export HF_TOKEN="hf_your_token"
bash /workspace/vllm/scripts/download_models.sh
```

### Step 5: Run Batch Evaluation

```bash
# Start supervisord
supervisord -c /workspace/ops/supervisord.conf

# Set your keys
export VLLM_API_KEY="sk-your-key"
export OPENAI_API_KEY="sk-openai-your-key"

# Run evaluation on all models
bash /workspace/vllm/scripts/batch_evaluate.sh

# Or evaluate specific models only
bash /workspace/vllm/scripts/batch_evaluate.sh eurollm qwen3 deepseek
```

## Detailed Documentation

### File Structure

```
/workspace/
├── models/                          # Downloaded model weights (~150GB)
│   ├── qwen3_30b_a3b_awq/          # Qwen3-30B-A3B-AWQ (~16-18GB)
│   ├── eurollm_9b/                  # EuroLLM 9B (~18GB)
│   ├── mistral_small_24b/           # Mistral Small 24B (~48GB)
│   ├── mixtral_8x7b/                # Mixtral 8x7B MoE (~45-50GB)
│   └── deepseek_14b/                # DeepSeek 14B (~28GB)
│
├── vllm/                            # vLLM service configuration
│   ├── scripts/
│   │   ├── download_models.sh       # Download all models
│   │   ├── switch_model.sh          # Switch active model
│   │   ├── run_vllm.sh              # Wrapper script (reads current.env)
│   │   └── batch_evaluate.sh        # Main batch evaluation script
│   ├── current.env                  # Active model config (auto-generated)
│   └── logs/
│       ├── vllm.log                 # vLLM stdout
│       └── vllm.err.log             # vLLM stderr
│
├── llm_evaluator/                   # This repository
│   ├── app/                         # FastAPI evaluator service
│   ├── metrics/                     # Evaluation metrics
│   ├── data/                        # Test data
│   ├── .venv/                       # Python virtual environment
│   └── run_evaluation.py            # CLI client
│
├── ops/
│   └── supervisord.conf             # Supervisord configuration
│
├── logs/
│   ├── supervisord.log              # Supervisor logs
│   ├── evaluator.log                # Evaluator API logs
│   └── evaluator.err.log
│
└── evaluation_results/              # Evaluation outputs
    ├── eurollm_20240220_143022.json
    ├── qwen3_20240220_150145.json
    └── ...
```

### Available Models

| Model | Directory | Size | VRAM Usage | Notes |
|-------|-----------|------|------------|-------|
| **EuroLLM-9B** | `eurollm_9b` | ~18GB | ✅ Comfortable | Good for EU languages |
| **Qwen3-30B-A3B-AWQ** | `qwen3_30b_a3b_awq` | ~16-18GB | ✅ Comfortable | AWQ quantized |
| **DeepSeek-R1-14B** | `deepseek_14b` | ~28GB | ✅ Comfortable | Reasoning model |
| **Mixtral-8x7B** | `mixtral_8x7b` | ~45-50GB | ⚠️ Tight | MoE (sparse) |
| **Mistral-Small-24B** | `mistral_small_24b` | ~48GB | ⚠️ Very Tight | May OOM on long context |

⚠️ **Mistral Small Warning**: On 48GB A40, this runs with `max_model_len=4096`. For longer contexts or more headroom, use a quantized variant.

### Managing Services

#### Start Everything

```bash
supervisord -c /workspace/ops/supervisord.conf
```

#### Check Status

```bash
# View all services
supervisorctl -c /workspace/ops/supervisord.conf status

# Example output:
# evaluator                        RUNNING   pid 1234, uptime 0:10:23
# vllm                             STOPPED   Not started
```

#### View Logs

```bash
# vLLM logs
supervisorctl -c /workspace/ops/supervisord.conf tail -f vllm
supervisorctl -c /workspace/ops/supervisord.conf tail -200 vllm stderr

# Evaluator logs
supervisorctl -c /workspace/ops/supervisord.conf tail -f evaluator
supervisorctl -c /workspace/ops/supervisord.conf tail -200 evaluator stderr

# Health check (comprehensive status)
bash /workspace/vllm/scripts/health_check.sh
```

#### Stop/Restart Services

```bash
# Stop vLLM (but keep evaluator running)
supervisorctl -c /workspace/ops/supervisord.conf stop vllm

# Restart evaluator
supervisorctl -c /workspace/ops/supervisord.conf restart evaluator

# Stop everything
supervisorctl -c /workspace/ops/supervisord.conf shutdown
```

### Manual Model Switching

If you want to manually switch models (outside of batch evaluation):

```bash
# Switch to a specific model
/workspace/vllm/scripts/switch_model.sh eurollm

# Available models:
#   eurollm, qwen3, deepseek, mixtral, mistral-small

# Check if vLLM is ready
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

### Running Individual Evaluations

If you want to run a single evaluation (not batch):

```bash
# Ensure evaluator is running
supervisorctl -c /workspace/ops/supervisord.conf start evaluator

# Switch to desired model
/workspace/vllm/scripts/switch_model.sh eurollm

# Wait for vLLM to be ready, then run evaluation
cd /workspace/llm_evaluator
.venv/bin/python run_evaluation.py \
  --api-url http://localhost:8080 \
  --model eurollm \
  --languages EN DE FR \
  --runs 3 \
  --report
```

### Batch Evaluation Options

The batch script supports several modes:

```bash
# Evaluate all models (default)
bash /workspace/vllm/scripts/batch_evaluate.sh

# Evaluate specific models only
bash /workspace/vllm/scripts/batch_evaluate.sh eurollm qwen3

# With custom settings (edit the script or use env vars)
export NUM_RUNS_PER_QUESTION=5
export DEFAULT_TEMPERATURE=0.7
bash /workspace/vllm/scripts/batch_evaluate.sh
```

## Troubleshooting

### "Model directory not found"

```bash
# Check what models you have
ls -la /workspace/models/

# Re-download missing models
bash /workspace/vllm/scripts/download_models.sh
```

### "vLLM failed to start" / OOM Errors

```bash
# Check GPU memory
nvidia-smi

# View vLLM error logs
supervisorctl -c /workspace/ops/supervisord.conf tail -50 vllm stderr

# If OOM on mistral-small, reduce max_model_len in switch_model.sh
# Edit /workspace/vllm/scripts/switch_model.sh:
#   ["mistral-small"]="4096"   # Reduce this further
```

### "Evaluator API not responding"

```bash
# Check if evaluator is running
supervisorctl -c /workspace/ops/supervisord.conf status evaluator

# Restart evaluator
supervisorctl -c /workspace/ops/supervisord.conf restart evaluator

# Check evaluator logs
supervisorctl -c /workspace/ops/supervisord.conf tail -50 evaluator stderr
```

### "Supervisord not running"

```bash
# Check if supervisord process exists
ps aux | grep supervisord

# Kill any stuck processes and restart
pkill supervisord
supervisord -c /workspace/ops/supervisord.conf
```

### "Download interrupted / corrupted"

```bash
# Remove corrupted model directory
rm -rf /workspace/models/<model_name>

# Re-download
bash /workspace/vllm/scripts/download_models.sh
```

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | ✅ Yes | HuggingFace token for downloading models |
| `VLLM_API_KEY` | ✅ Yes | API key for securing vLLM endpoint |
| `OPENAI_API_KEY` | ✅ Yes | OpenAI key for evaluation/judging |
| `NUM_RUNS_PER_QUESTION` | ❌ No | Default: 3 |
| `DEFAULT_TEMPERATURE` | ❌ No | Default: 0.0 |
| `EMBEDDING_MODEL` | ❌ No | Default: paraphrase-multilingual-mpnet-base-v2 |

## Advanced Configuration

### Adding a New Model

1. Add to `download_models.sh`:
```bash
declare -A MODELS=(
  # ... existing models ...
  ["my_new_model"]="org/model-name"
)
```

2. Add to `switch_model.sh`:
```bash
declare -A PATHS=(
  # ... existing models ...
  ["my_new_model"]="/workspace/models/my_new_model"
)
declare -A QUANTS=(
  # ... existing models ...
  ["my_new_model"]="--quantization awq"  # or empty
)
declare -A MAX_LENS=(
  ["my_new_model"]="8192"
)
declare -A GPU_UTILS=(
  ["my_new_model"]="0.80"
)
```

3. Download and run:
```bash
bash /workspace/vllm/scripts/download_models.sh
bash /workspace/vllm/scripts/batch_evaluate.sh my_new_model
```

### Customizing Evaluation Parameters

Edit `/workspace/llm_evaluator/.env` or set environment variables:

```bash
export NUM_RUNS_PER_QUESTION=5
export DEFAULT_TEMPERATURE=0.5
export DEFAULT_MAX_TOKENS=4096
```

## Storage Requirements

| Component | Size |
|-----------|------|
| Models (5 total) | ~150 GB |
| HF Cache | ~20 GB |
| Evaluation Results | ~1 GB |
| OS + Dependencies | ~20 GB |
| **Total Recommended** | **~200 GB** |

## Performance Expectations

On A40 48GB:

| Model | Load Time | Eval Speed (24 langs × 5 Q × 3 runs) |
|-------|-----------|--------------------------------------|
| EuroLLM-9B | ~30s | ~15-20 min |
| Qwen3-AWQ | ~20s | ~12-15 min |
| DeepSeek-14B | ~45s | ~20-25 min |
| Mixtral-8x7B | ~60s | ~25-30 min |
| Mistral-Small-24B | ~60s | ~30-35 min |

**Total batch time**: ~2-3 hours for all 5 models

## Support & Debugging

### Get Help

```bash
# Health check everything
bash /workspace/vllm/scripts/health_check.sh

# Check all logs
tail -100 /workspace/vllm/logs/vllm.err.log
tail -100 /workspace/logs/evaluator.err.log
tail -100 /workspace/logs/supervisord.log
```

### Reset Everything

```bash
# Stop all services
supervisorctl -c /workspace/ops/supervisord.conf shutdown 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true

# Clear results (keep models)
rm -rf /workspace/evaluation_results/*

# Or clear everything including models
rm -rf /workspace/models/*
rm -rf /workspace/.cache/huggingface/*

# Re-download models
bash /workspace/vllm/scripts/download_models.sh
```
