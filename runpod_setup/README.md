# RunPod LLM Evaluation - Simplified Setup

Evaluate multiple LLMs on RunPod with A40 48GB GPU.

## Quick Start

### 1. Clone Repo on RunPod

```bash
ssh root@YOUR_RUNPOD_IP
cd /workspace
git clone https://github.com/YOUR_USERNAME/llm_evaluator.git
cd llm_evaluator
```

### 2. Run Setup

```bash
cd runpod_setup
bash setup.sh
```

This installs vLLM, creates venv, and sets up directories.

### 3. Configure

Edit `config.yaml` with your token:

```yaml
hf_token: "hf_your_token"           # For downloading models
vllm_api_key: "sk-anything"          # Can leave as-is
```

### 4. Download Models

```bash
export HF_TOKEN="hf_your_token"
.venv/bin/python runpod_setup/download_models.py
```

Takes 1-3 hours, downloads ~150GB to `/workspace/models/`.

### 5. Run Evaluation

```bash
export OPENAI_API_KEY="sk_your_key"
.venv/bin/python runpod_setup/evaluate_context.py
```

This will:
1. Start vLLM with model 1
2. Run evaluation (ask questions in multiple languages)
3. Save results to SQLite + JSON
4. Stop vLLM, load model 2, repeat...

Takes ~2-3 hours for all 5 models.

### 6. Download Results

On your laptop:

```bash
scp -r root@YOUR_RUNPOD_IP:/workspace/evaluation_results ./
```

## File Structure

```
runpod_setup/
├── config.yaml          # Edit this with your tokens
├── setup.sh             # Run once to install deps
├── download_models.py   # Download all models
├── evaluate.py          # Main evaluation script
└── README.md            # This file
```

On RunPod:

```
/workspace/
├── models/                         # Downloaded LLMs (~150GB)
│   ├── eurollm_9b/
│   ├── qwen3_30b_a3b_awq/
│   ├── deepseek_14b/
│   ├── mixtral_8x7b/
│   └── mistral_small_24b/
│
├── evaluation_results/             # ⭐ Your results
│   ├── evaluation_results.db      # SQLite database
│   └── *.json                     # Per-model summaries
│
└── llm_evaluator/                  # This repo
    └── ...
```

## Models

| Model | Size | VRAM | Status |
|-------|------|------|--------|
| EuroLLM-9B | ~18GB | ✅ Comfortable | EU languages |
| Qwen3-30B-AWQ | ~16GB | ✅ Comfortable | AWQ quantized |
| DeepSeek-14B | ~28GB | ✅ Comfortable | Reasoning |
| Mixtral-8x7B | ~45GB | ⚠️ Tight | MoE |
| Mistral-Small-24B | ~48GB | ⚠️ Very Tight | May OOM |

## Troubleshooting

### "Out of Memory"

```bash
# Check GPU
nvidia-smi

# If Mistral fails, skip it and run others:
# Edit config.yaml and comment out mistral_small_24b
```

### "Model download failed"

```bash
# Retry download
.venv/bin/python runpod_setup/download_models.py
```

### "vLLM won't start"

```bash
# Check logs
ps aux | grep vllm
killall vllm  # Force kill

# Then retry
.venv/bin/python runpod_setup/evaluate_context.py
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Which languages to evaluate
evaluation:
  languages: ["EN", "DE", "FR"]  # Default: 5 EU languages
  num_runs: 3                     # Runs per question

# Which models to evaluate
models:
  eurollm_9b:
    # ... keep or remove models as needed
```

## Results

After evaluation, you'll have:

1. **SQLite database** (`evaluation_results.db`)
   - All responses with metadata
   - Query with: `sqlite3 evaluation_results.db "SELECT * FROM evaluations LIMIT 5;"`

2. **JSON summaries** (one per model)
   - Summary statistics
   - Success rates

Download both to your laptop for analysis.

## Costs

- **Compute**: ~$0.50-1.00/hour for A40 on RunPod
- **Storage**: ~$0.20/GB/month for persistent volume
- **Total for full eval**: ~$3-6 (3-6 hours runtime)

Tip: Stop pod when not using. Models persist in `/workspace/`.
