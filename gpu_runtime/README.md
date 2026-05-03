# GPU Server Context Evaluation Guide

Run context-based EU-FarmBook evaluation on a GPU server.

Examples below use `/workspace/llm_evaluator`. If that folder does not exist yet on your machine, create `/workspace` first and then clone the repo there.

Python version: use `python3`. In this project, the current validated interpreter is Python 3.12.3, and the setup examples assume that `python3` resolves to that interpreter or a compatible Python 3.12 install.

Currently tuned GPU targets for config generation are:
- `a40`
- `l40s`
- `3090`
- `a100`
- `h200_sxm`
- `b200`

Those are the GPUs with first-class sizing support in `gpu_runtime/generate_gpu_config.py` and `model_static_check.py` today.

## Quick Start

### 1. Clone on GPU server

```bash
ssh YOUR_GPU_SERVER_IP_CONFIG
mkdir -p /workspace
cd /workspace
git clone https://github.com/pranavnbapat/llm_evaluator
cd llm_evaluator
```

### 2. Configure Git Push Auth Once

```bash
bash gpu_runtime/git_bootstrap.sh
```

This stores your GitHub PAT in your home directory credential store so `git push` does not keep prompting for username and password on that server.

### 3. Setup

```bash
bash gpu_runtime/setup.sh
source .venv/bin/activate
```

### 4. Configure tokens

```bash
echo "HF_TOKEN=hf_your_token" > gpu_runtime/.env
```

`HF_TOKEN` is required for downloading gated Hugging Face models and configs.

`OPENAI_API_KEY` is not required for the core GPU runtime flow.

An LLM API key is only optional for some insight-generation helpers outside the core runtime flow, such as presentation/report generation scripts under `insights/`.

### 5. Generate GPU-specific model config

```bash
python gpu_runtime/generate_gpu_config.py a40 \
  --repos-file gpu_runtime/model_repos.txt \
  --config-file gpu_runtime/config.yaml \
  --concurrent-users 50 \
  --target-max-output-tokens 512
```

Replace `a40` with one of: `a40`, `l40s`, `3090`, `a100`, `h200_sxm`, `b200`.

`h100_sxm` is detected by the runtime for run bucketing, but it does not currently have a dedicated config-generation profile.

For practical sizing:
- `l40s` currently uses the same VRAM class assumptions as `a40` (48 GB).
- `3090` uses a 24 GB profile, so expect a much smaller model set to pass config generation.

### 6. Download models

```bash
python gpu_runtime/download_models.py
```

### 7. Run evaluation

Foreground:
```bash
python gpu_runtime/evaluate_context.py
```

Background:
```bash
bash gpu_runtime/run_evaluate_context_background.sh
```

### 8. Run scoring

Foreground:
```bash
python gpu_runtime/evaluate_context_results.py
```

Background:
```bash
bash gpu_runtime/run_evaluate_context_results_background.sh
```

Background scorer default behavior:
- If `EVAL_RUN_DIR` or `EVAL_RUN_ID` is set: scores that single run.
- If neither is set: scores all runs under `results/runs/<detected_or_forced_gpu_bucket>/`.

Examples:

```bash
# Score one run
EVAL_RUN_GPU=a40 EVAL_RUN_DIR="results/runs/a40/<run_id>" \
bash gpu_runtime/run_evaluate_context_results_background.sh

# Score all a40 runs (useful when hardware is A100 but run data is in a40 bucket)
EVAL_RUN_GPU=a40 bash gpu_runtime/run_evaluate_context_results_background.sh
```

Scoring performance env (recommended):

```bash
export EVALUATOR_METRICS_DEVICE=cuda
export EVALUATOR_SCORE_COMMIT_EVERY=500
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Suggested batch size by GPU:
# A100 / H200-SXM / B200: 128
# A40 / L40S: 96
# 3090: start lower and validate on your selected models
export EVALUATOR_SCORE_BATCH_SIZE=96
```

You can also put the same keys in root `.env` (copy from `.env.sample`) instead of exporting every session.

### 9. Generate insights (run after scoring)

Recommended (single command):

```bash
RUN_DIR="results/runs/<gpu_bucket>/<run_id>"
bash gpu_runtime/run_post_scoring_insights.sh --run-dir "$RUN_DIR"
```

All runs:

```bash
bash gpu_runtime/run_post_scoring_insights.sh --all-runs
```

Regenerate outputs:

```bash
RUN_DIR="results/runs/<gpu_bucket>/<run_id>"
bash gpu_runtime/run_post_scoring_insights.sh --run-dir "$RUN_DIR" --force
```

Equivalent manual order for one run:

```bash
RUN_DIR="results/runs/<gpu_bucket>/<run_id>"
GPU_BUCKET="<gpu_bucket>"

python insights/generate_context_charts.py --run-dir "$RUN_DIR"
python insights/generate_presentation_qa.py --run-dir "$RUN_DIR"
python insights/generate_context_token_budget.py --run-dir "$RUN_DIR"
python insights/generate_context_vram_docs.py --run-dir "$RUN_DIR"
python insights/gpu_efficiency/generate_gpu_efficiency_report.py --run-dir "$RUN_DIR"
python insights/generate_gpu_insights_report.py --gpu "$GPU_BUCKET"
```

What each script does:
- `generate_context_charts.py`: creates `insights/charts/*.png` and summary CSVs in `insights/data/`.
- `generate_presentation_qa.py`: creates presentation QA artifacts (`insights/Presentation_QA.md`, `insights/data/presentation_qa.*`).
- `generate_context_token_budget.py`: creates token-budget CSVs in `insights/data/`.
- `generate_context_vram_docs.py`: creates VRAM/context markdown docs from token-budget outputs.
- `generate_gpu_insights_report.py`: creates the combined run-level insights report markdown.
- `gpu_efficiency/generate_gpu_efficiency_report.py`: creates GPU utilization/phase efficiency charts + markdown in `insights/gpu_efficiency/`.
- `gpu_runtime/run_post_scoring_insights.sh`: runs the full post-scoring pipeline in this order.

Notes:
- Insights scripts are not auto-triggered by evaluation/scoring.
- Most scripts support bulk mode without `--run-dir`.
- Run `gpu_efficiency` after evaluation has produced `logs/gpu_metrics.csv`.

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
python gpu_runtime/generate_gpu_config.py a40 --dry-run
```

Include tighter fits:
```bash
python gpu_runtime/generate_gpu_config.py a40 --allow-fits "comfortable,tight"
```

Force run folder controls:
```bash
export EVAL_RUN_GPU=a40
export EVAL_RUN_ID=2026-03-07_120000_context_eval
export EVAL_RUN_DIR="/workspace/llm_evaluator/results/runs/a40/2026-03-07_120000_context_eval"
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
bash gpu_runtime/git_bootstrap.sh
```

## Adding Another GPU Target

To add another GPU beyond the currently supported set, update these files:

- `model_static_check.py`: add VRAM size, CLI aliases, and any fit-report text.
- `gpu_runtime/generate_gpu_config.py`: update CLI help/error text for the new target.
- `gpu_runtime/evaluate_context.py`
- `gpu_runtime/evaluate_context_results.py`
- `gpu_runtime/run_evaluate_context_background.sh`
- `gpu_runtime/run_evaluate_context_results_background.sh`

Those runtime files need a matching `nvidia-smi` name check if you want automatic run bucketing.

Suggested validation flow for a new GPU:

1. Start with the closest existing VRAM class.
2. Generate config with the nearest supported profile.
3. Force `EVAL_RUN_GPU` to the new bucket name while testing.
4. Run a small validation pass before treating it as fully supported.
