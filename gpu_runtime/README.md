# GPU Server Context Evaluation Guide

Run context-based EU-FarmBook evaluation on a GPU server.

This folder now supports two GPU evaluation paths:
- text/context evaluation: `generate_gpu_config.py` + `evaluate_context.py`
- multimodal image/PDF evaluation: `generate_gpu_vision_config.py` + `evaluate_vision.py`

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

## TL;DR — Minimum Steps (text/context path)

Pick your GPU class once and substitute it for `<GPU>` below (one of `a40`, `l40s`,
`3090`, `a100`, `h200_sxm`, `b200`).

```bash
# 1. Clone and enter the repo (e.g. on RunPod)
cd /workspace
git clone https://github.com/pranavnbapat/llm_evaluator
cd llm_evaluator

# 2. Install dependencies and create the venv
bash gpu_runtime/setup.sh

# 3. Activate the venv
source .venv/bin/activate

# 4. Set your HF token (needed for gated model downloads)
echo "HF_TOKEN=hf_your_token_here" > gpu_runtime/.env

# 5. Generate a GPU-specific model config
python gpu_runtime/generate_gpu_config.py <GPU> \
  --repos-file gpu_runtime/model_repos.txt \
  --config-file gpu_runtime/config.yaml \
  --concurrent-users 50 \
  --target-max-output-tokens 512

# 6. Download the models that survived config generation
python gpu_runtime/download_models.py

# 7. Run the context evaluation (foreground; use the background variant for long runs)
python gpu_runtime/evaluate_context.py

# 8. Score the responses produced in step 7
python gpu_runtime/evaluate_context_results.py
```

After step 8 you have raw results in `results/runs/<GPU>/<run_id>/raw/` and scores in
`results/runs/<GPU>/<run_id>/scores/`. To turn those into the combined insights
report, run the post-scoring pipeline (optional but recommended):

```bash
bash gpu_runtime/run_post_scoring_insights.sh --all-runs --gpu <GPU>
```

The detailed reference for every step (background runners, env tuning, vision path,
troubleshooting) is below.

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

Text/context models:

```bash
python gpu_runtime/generate_gpu_config.py a40 \
  --repos-file gpu_runtime/model_repos.txt \
  --config-file gpu_runtime/config.yaml \
  --concurrent-users 50 \
  --target-max-output-tokens 512
```

Vision / multimodal models:

```bash
python gpu_runtime/generate_gpu_vision_config.py l40s \
  --repos-file gpu_runtime/model_repos.txt \
  --config-file gpu_runtime/config.yaml \
  --concurrent-users 25 \
  --target-max-output-tokens 512 \
  --allow-fits "comfortable,tight"
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

Text/context path

Foreground:
```bash
python gpu_runtime/evaluate_context.py
```

Background:
```bash
bash gpu_runtime/run_evaluate_context_background.sh
```

Multimodal image/PDF path

Foreground:
```bash
python gpu_runtime/evaluate_vision.py
```

Background:
```bash
bash gpu_runtime/run_evaluate_vision_background.sh
```

### 8. Run scoring

Text/context path

Foreground:
```bash
python gpu_runtime/evaluate_context_results.py
```

Background:
```bash
# One run (CLI flag preferred)
bash gpu_runtime/run_evaluate_context_results_background.sh \
  --run-dir "results/runs/<gpu_bucket>/<run_id>"

# All runs in the detected GPU bucket
bash gpu_runtime/run_evaluate_context_results_background.sh --all-runs
```

Multimodal image/PDF path

Foreground:
```bash
python gpu_runtime/evaluate_vision_results.py
```

Background:
```bash
EVAL_RUN_DIR="results/runs/<gpu_bucket>/<run_id>" \
bash gpu_runtime/run_evaluate_vision_results_background.sh
```

Single-run resolution order for the context background scorer:
1. `--run-dir <path>` CLI flag
2. `EVAL_RUN_DIR` env
3. `EVAL_RUN_ID` env (joined with the detected GPU bucket)
4. legacy flat `results/` fallback only when no run dir is provided

Default behavior: if none of the above is set and `--all-runs` was not explicitly
passed, the script auto-flips to **`--all-runs`** for the detected GPU bucket. Be
deliberate: forgetting `--run-dir` re-scores every run in the bucket.

Resumable scoring:
- Scores are upserted on `(model_name, language, question_id, run_number)`.
  If you Ctrl-C or crash, just rerun — already-scored cells are not re-evaluated
  duplicately, only overwritten in place.
- Set `FORCE_RESCORE=1` to wipe existing rows before scoring.

Examples:

```bash
# Score one run
bash gpu_runtime/run_evaluate_context_results_background.sh \
  --run-dir "results/runs/a40/<run_id>"

# Score all a40 runs (useful when hardware is A100 but run data lives in a40 bucket)
EVAL_RUN_GPU=a40 bash gpu_runtime/run_evaluate_context_results_background.sh --all-runs

# Force a full re-score of one run
FORCE_RESCORE=1 \
bash gpu_runtime/run_evaluate_context_results_background.sh \
  --run-dir "results/runs/a40/<run_id>"
```

Scoring performance env (recommended):

```bash
export EVALUATOR_METRICS_DEVICE=cuda     # cuda | cpu | auto (default)
export EVALUATOR_SCORE_COMMIT_EVERY=500
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Suggested batch size by GPU:
# A100 / H200-SXM / B200: 128
# A40 / L40S: 96
# 3090: start lower and validate on your selected models
export EVALUATOR_SCORE_BATCH_SIZE=96
```

`EVALUATOR_METRICS_DEVICE` controls only the scoring metrics (xlm-roberta NLI/zero-shot,
sentence embeddings). It is independent of vLLM's GPU usage during evaluation.

You can also put the same keys in root `.env` (copy from `.env.sample`) instead of
exporting every session.

### 9. Generate insights (run after scoring)

Recommended (single command):

```bash
# One run + refresh that bucket's aggregate report
bash gpu_runtime/run_post_scoring_insights.sh \
  --run-dir "results/runs/<gpu_bucket>/<run_id>"

# Every run, every bucket
bash gpu_runtime/run_post_scoring_insights.sh --all-runs

# Every run in one bucket only
bash gpu_runtime/run_post_scoring_insights.sh --all-runs --gpu l40s

# Regenerate even when output artifacts already exist
bash gpu_runtime/run_post_scoring_insights.sh --run-dir "$RUN_DIR" --force
```

Run dirs without `scores/evaluation_scores_euf_context.db` are skipped with a
warning, so you can safely re-run while some buckets are still scoring.

Pipeline order (each step's output feeds the next):

| # | Script | Output | Consumed by |
|---|---|---|---|
| 1 | `insights/generate_context_charts.py` | `insights/charts/*.png`, summary CSVs in `insights/data/` | downstream charts |
| 2 | `insights/generate_presentation_qa.py` | `insights/Presentation_QA.md`, `insights/data/presentation_qa.*` | — |
| 3 | `insights/generate_context_token_budget.py` | `insights/data/token_budget_response_details.csv` and friends | step 6 (token / truncation tables, context utilisation, failure-mode breakdown) |
| 4 | `insights/generate_context_vram_docs.py` | VRAM/context markdown | — |
| 5 | `insights/gpu_efficiency/generate_gpu_efficiency_report.py` | `insights/gpu_efficiency/*` charts + markdown | — |
| 6 | `insights/generate_gpu_insights_report.py --gpu <bucket>` | `results/runs/<gpu_bucket>/CONTEXT_EVALUATION_INSIGHTS_REPORT.md` (the combined report) | — |

Equivalent manual invocation for one run:

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

Notes:
- Insights are not auto-triggered by evaluation/scoring.
- Step 5 needs `logs/gpu_metrics.csv` from evaluation; missing it just skips the GPU section.
- Step 6 reads the metric weights live from `metrics/metrics_config.yaml`; the report's
  methodology section reflects whatever the active `context` profile actually uses.
- Step 6 also runs an integrity check that flags suspicious patterns in the scores
  (boundary saturation, perfectly collinear metrics, etc). Read the
  "Score Integrity Checks" section in the generated report — that's where the
  `prompt_alignment ≡ relevance` and `completeness saturation` issues surfaced.
- The failure-mode breakdown for low-quality languages uses optional
  `langdetect`. Install with `pip install langdetect` to get the
  `lang_mismatch_pct_sample30` column populated; without it you'll see `NA`.
- Post-scoring insights are currently context-oriented. The multimodal path has
  raw results and scores, but it does not yet use the same report stack.

## Multimodal Dataset

The multimodal runner reads:

`data/evaluation_vision_questions.json`

Supported task types:
- `image` + `qa`
- `image` + `summary`
- `pdf` + `qa`
- `pdf` + `summary`

Supported media roots:
- `image_root`
- `pdf_root`

Current behavior:
- the same images/PDFs can be reused across all 24 EU languages
- prompts can be localized with `question_translations` and `summary_prompt_translations`
- PDF evaluation renders pages to PNGs with `pdftoppm`, then runs chunked map/reduce prompts through the VLM

Current starter dataset points at files under:

`files/`

If you add your own media, update the filenames in `data/evaluation_vision_questions.json`.

For first validation runs:
- keep `evaluation.num_runs: 1`
- keep the model set small
- start with one or two models before scaling up to the full dataset

The current starter multimodal dataset expands to `264` tasks per model:
- `11` base tasks
- `24` EU languages

PDF prerequisite:

```bash
command -v pdftoppm
```

If `pdftoppm` is missing, PDF tasks will not run correctly.

Install it with:

```bash
apt-get update && apt-get install -y poppler-utils
```

## Output Layout

Evaluation and scoring write to:

`results/runs/<gpu_bucket>/<run_id>/`

Subfolders:
- `raw/` evaluation DB, per-model JSON summaries, Excel exports
- `scores/` scoring DB and score exports
- `logs/` evaluate/scoring logs, `gpu_metrics.csv`
- `insights/` generated plots/reports
- `metadata/` run metadata and model status

Typical DB names:
- text/context eval: `raw/evaluation_results_euf_context.db`
- text/context scores: `scores/evaluation_scores_euf_context.db`
- multimodal eval: `raw/evaluation_results_euf_vision.db`
- multimodal scores: `scores/evaluation_scores_euf_vision.db`

## Useful Commands

Generate only (no file write):
```bash
python gpu_runtime/generate_gpu_config.py a40 --dry-run
```

Multimodal generate only:
```bash
python gpu_runtime/generate_gpu_vision_config.py l40s --dry-run
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

## Tmux basics

The `run_evaluate_*_background.sh` scripts launch the evaluation/scoring inside
a `tmux` session so it survives SSH disconnects. The session name is printed
once at launch — the cheat sheet below is for after that scrolls off.

| Action | Command |
|---|---|
| List all sessions | `tmux ls` |
| Attach to a session | `tmux attach -t <name>` (short form: `tmux a -t <name>`) |
| Detach from current session | Press `Ctrl+b`, release, then press `d` (chord, not simultaneous) |
| Kill a session | `tmux kill-session -t <name>` |
| Kill every session | `tmux kill-server` |
| Scroll inside attached session | `Ctrl+b` then `[`, navigate with arrows / PageUp, press `q` to exit |

Session names this repo uses (across both engines, in case both are running):

| Script | Session name |
|---|---|
| `gpu_runtime/run_evaluate_context_background.sh` | `eval_context` |
| `gpu_runtime/run_evaluate_context_results_background.sh` | `eval_context_scores` |
| `gpu_runtime/run_evaluate_vision_background.sh` | `eval_vision` |
| `gpu_runtime/run_evaluate_vision_results_background.sh` | `eval_vision_scores` |
| `gpu_runtime_sglang/run_evaluate_context_background.sh` | `eval_context_sglang` |
| `gpu_runtime_sglang/run_evaluate_context_results_background.sh` | `eval_context_scores_sglang` |

Important:
- Detaching keeps the run alive. Closing the SSH terminal also keeps it alive.
- The only way to actually stop the evaluation is `tmux kill-session -t <name>`
  or `Ctrl+C` while attached.

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

### PDF/media issues

Check that the dataset points at real files:

```bash
python - <<'PY'
import json
from pathlib import Path
data = json.load(open("data/evaluation_vision_questions.json"))
print("image_root:", data.get("image_root"))
print("pdf_root:", data.get("pdf_root"))
PY
```

Check PDF rendering dependency:

```bash
command -v pdftoppm
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
- `gpu_runtime/generate_gpu_vision_config.py`: update multimodal sizing/help text for the new target.
- `gpu_runtime/evaluate_context.py`
- `gpu_runtime/evaluate_context_results.py`
- `gpu_runtime/evaluate_vision.py`
- `gpu_runtime/evaluate_vision_results.py`
- `gpu_runtime/run_evaluate_context_background.sh`
- `gpu_runtime/run_evaluate_context_results_background.sh`
- `gpu_runtime/run_evaluate_vision_background.sh`
- `gpu_runtime/run_evaluate_vision_results_background.sh`

Those runtime files need a matching `nvidia-smi` name check if you want automatic run bucketing.

Suggested validation flow for a new GPU:

1. Start with the closest existing VRAM class.
2. Generate config with the nearest supported profile.
3. Force `EVAL_RUN_GPU` to the new bucket name while testing.
4. Run a small validation pass before treating it as fully supported.
