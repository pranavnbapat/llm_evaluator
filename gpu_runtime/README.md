# GPU Server Context Evaluation Guide

Run context-based EU-FarmBook evaluation on a GPU server.

## Quick Start

### 1. Clone on GPU server

```bash
ssh YOUR_GPU_SERVER_IP_CONFIG
cd /workspace
git clone https://github.com/pranavnbapat/llm_evaluator
cd llm_evaluator
```

### 2. Setup

```bash
bash runpod_setup/setup.sh
source .venv/bin/activate
```

### 3. Configure tokens

```bash
echo "HF_TOKEN=hf_your_token" > runpod_setup/.env
export OPENAI_API_KEY="sk_your_key"
```

### 4. Generate GPU-specific model config

```bash
python runpod_setup/generate_gpu_config.py a40 \
  --repos-file runpod_setup/model_repos.txt \
  --config-file runpod_setup/config.yaml \
  --concurrent-users 50 \
  --target-max-output-tokens 512
```

### 5. Download models

```bash
python runpod_setup/download_models.py
```

### 6. Run evaluation

Foreground:
```bash
python runpod_setup/evaluate_context.py
```

Background:
```bash
bash runpod_setup/run_evaluate_context_background.sh
```

### 7. Run scoring

Foreground:
```bash
python runpod_setup/evaluate_context_results.py
```

Background:
```bash
bash runpod_setup/run_evaluate_context_results_background.sh
```

### 8. Generate insights (run after scoring)

Run these scripts in order for a specific run:

```bash
RUN_DIR="results/runs/<gpu_bucket>/<run_id>"

python insights/generate_context_charts.py --run-dir "$RUN_DIR"
python insights/generate_context_insights.py --run-dir "$RUN_DIR"
python insights/generate_presentation_qa.py --run-dir "$RUN_DIR"
python insights/gpu_efficiency/generate_gpu_efficiency_report.py --run-dir "$RUN_DIR"
```

Notes:
- Insights scripts are not auto-triggered by evaluation/scoring.
- You can also run each script without `--run-dir` to process runs in bulk mode.

## Output Layout

Evaluation and scoring write to:

`results/runs/<gpu_bucket>/<run_id>/`

Subfolders:
- `raw/` evaluation DB, per-model JSON summaries, Excel exports
- `scores/` scoring DB and score exports
- `logs/` evaluate/scoring logs, `gpu_metrics.csv`
- `insights/` generated plots/reports
- `metadata/` run metadata and model status

Latest symlink:

`results/latest/<gpu_bucket> -> ../runs/<gpu_bucket>/<run_id>`

## Useful Commands

Generate only (no file write):
```bash
python runpod_setup/generate_gpu_config.py a40 --dry-run
```

Include tighter fits:
```bash
python runpod_setup/generate_gpu_config.py a40 --allow-fits "comfortable,tight"
```

Force run folder controls:
```bash
export EVAL_RUN_GPU=a40
export EVAL_RUN_ID=2026-03-07_120000_context_eval
export EVAL_RUN_DIR=/workspace/llm_evaluator/results/runs/a40/2026-03-07_120000_context_eval
```

View latest evaluation log:
```bash
LOG=$(ls -1t /workspace/llm_evaluator/results/runs/a40/*/logs/evaluate_context_*.log | head -1)
less "$LOG"
```

## Troubleshooting

### vLLM startup fails

```bash
nvidia-smi
pkill -f "vllm.*serve" || true
```

Check model-specific logs:

```bash
tail -n 120 /tmp/vllm_<model_name>.log
```

### Database/path issues

Check status file and run metadata:

```bash
cat results/runs/<gpu_bucket>/<run_id>/metadata/model_status.json
cat results/runs/<gpu_bucket>/<run_id>/metadata/run_info.json
```

### Push prompts on every GPU server session

Configure once per server:

```bash
bash runpod_setup/git_bootstrap.sh
```
