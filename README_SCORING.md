# Scoring Run Guide (RunPod)

This guide runs only the scoring step (`gpu_runtime/evaluate_context_results.py`) from an existing context-evaluation database.

## 1) Clone and Set Up Python Env

```bash
git clone <repo>
cd llm_evaluator
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements_scoring.txt
```

## 2) Install tmux (for detached/background runs)

```bash
apt-get update -qq
apt-get install -y -qq tmux
```

Verify:

```bash
tmux -V
```

## 2.5) Configure Scoring Performance Env (Recommended)

You can set scoring env once in root `.env` or export in shell.

Option A: `.env` (persistent for the repo)

```bash
cp .env.sample .env
```

Set these keys in `.env`:

```bash
EVALUATOR_METRICS_DEVICE=cuda
EVALUATOR_SCORE_COMMIT_EVERY=500
TRANSFORMERS_VERBOSITY=error
HF_HUB_DISABLE_PROGRESS_BARS=1
```

Batch size by GPU:
- A100 SXM: `EVALUATOR_SCORE_BATCH_SIZE=128`
- A40: `EVALUATOR_SCORE_BATCH_SIZE=96`

Option B: export in current shell (non-persistent)

```bash
export EVALUATOR_METRICS_DEVICE=cuda
export EVALUATOR_SCORE_BATCH_SIZE=128   # use 96 on A40
export EVALUATOR_SCORE_COMMIT_EVERY=500
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

## 3) Run Scoring (Foreground)

```bash
python gpu_runtime/evaluate_context_results.py
```

By default, scoring resolves paths in this order:

1. `EVAL_RUN_DIR` (if set)
2. `results/latest/<detected_gpu_bucket>` (for example `results/latest/b200`)
3. legacy `results/` (fallback)

## 4) Run Scoring Detached (tmux background)

```bash
bash gpu_runtime/run_evaluate_context_results_background.sh
```

Default behavior of background launcher:

- If `EVAL_RUN_DIR` or `EVAL_RUN_ID` is set: scores that specific run.
- If neither is set: scores all runs under `results/runs/<detected_or_forced_gpu_bucket>/`.

Examples:

```bash
# Score one specific run
EVAL_RUN_GPU=a40 EVAL_RUN_DIR="results/runs/a40/<run_id>" \
bash gpu_runtime/run_evaluate_context_results_background.sh

# Score all runs in a40 bucket (even on non-a40 hardware)
EVAL_RUN_GPU=a40 bash gpu_runtime/run_evaluate_context_results_background.sh
```

Default tmux session name: `eval_context_scores`

Useful commands:

```bash
tmux ls
tmux attach -t eval_context_scores
```

Detach from tmux:

- `Ctrl+b`, then `d`

## 5) Outputs

Created/updated in the active run folder (recommended):

- `results/runs/<gpu_bucket>/<run_id>/scores/evaluation_scores_euf_context.db`
- `results/runs/<gpu_bucket>/<run_id>/scores/evaluation_scores_euf_context.xlsx`

Important behavior:

- `gpu_runtime/evaluate_context_results.py` performs a full rescore for the selected run.
- It clears existing rows in the `scores` table (`DELETE FROM scores`) before inserting new scores.
- So rerunning scoring overwrites prior score rows for that run DB instead of appending duplicates.

Legacy fallback (if no run folder is resolved):

- `evaluation_scores_euf_context.db`
- `evaluation_scores_euf_context.xlsx`

Logs:

- Background launcher writes to run logs when run folder is known:
  - `results/runs/<gpu_bucket>/<run_id>/logs/evaluate_context_results_<timestamp>.log`
- Otherwise fallback:
  - `logs/evaluate_context_results_<timestamp>.log`

Optional overrides:

- `EVAL_RUN_GPU` (force GPU bucket, e.g. `a40`, `a100`, `b200`, `h200_sxm`)
- `EVAL_RUN_ID` (select a specific run id under `results/runs/<gpu_bucket>/...`)
- `EVAL_RUN_DIR` (absolute override for a specific run folder)

## 6) Generate Insights and Charts (After Scoring)

These scripts are run-folder aware and support both single-run and bulk modes.

Dependency order:

1. `gpu_runtime/evaluate_context_results.py` (creates `scores/*.db` + `scores/*.xlsx`)
2. `insights/generate_context_charts.py` (creates `insights/data/*.csv` + `insights/charts/*.png`)
3. `insights/generate_presentation_qa.py` (creates `insights/Presentation_QA.md` + `insights/data/presentation_qa.*`)
4. `insights/generate_context_token_budget.py` (creates token-budget CSVs in `insights/data/`)
5. `insights/generate_context_vram_docs.py` (creates VRAM/context markdown docs from token-budget data)
6. `insights/generate_gpu_insights_report.py` (creates the combined run-level insights report markdown)
7. `insights/gpu_efficiency/generate_gpu_efficiency_report.py` (creates `insights/gpu_efficiency/*`)

Single run (recommended, one command):

```bash
RUN_DIR="results/runs/a40/2026-03-01_230442_context_eval"
bash gpu_runtime/run_post_scoring_insights.sh --run-dir "$RUN_DIR"
```

Single run (manual equivalent):

```bash
RUN_DIR="results/runs/a40/2026-03-01_230442_context_eval"
GPU_BUCKET="a40"

EVAL_RUN_GPU=a40 EVAL_RUN_DIR="$RUN_DIR" \
python gpu_runtime/evaluate_context_results.py

python insights/generate_context_charts.py --run-dir "$RUN_DIR"
python insights/generate_presentation_qa.py --run-dir "$RUN_DIR"
python insights/generate_context_token_budget.py --run-dir "$RUN_DIR"
python insights/generate_context_vram_docs.py --run-dir "$RUN_DIR"
python insights/gpu_efficiency/generate_gpu_efficiency_report.py --run-dir "$RUN_DIR"
python insights/generate_gpu_insights_report.py --gpu "$GPU_BUCKET"
```

Bulk mode (all runs, one command):

```bash
bash gpu_runtime/run_post_scoring_insights.sh --all-runs
```

Bulk mode (manual equivalent):

```bash
python insights/generate_context_charts.py
python insights/generate_presentation_qa.py
python insights/generate_context_token_budget.py
python insights/generate_context_vram_docs.py
python insights/gpu_efficiency/generate_gpu_efficiency_report.py
# per GPU bucket:
python insights/generate_gpu_insights_report.py --gpu a40
python insights/generate_gpu_insights_report.py --gpu a100
python insights/generate_gpu_insights_report.py --gpu b200
```

Bulk mode behavior:

- Skips runs that are already complete.
- Generates only missing artifacts.
- Use `--force` to regenerate.

## 7) Typical "Missing Input" Messages

If you see:

- `generate_context_charts.py ... missing ... scores/evaluation_scores_euf_context.db`
- `generate_context_token_budget.py ... missing ... raw/evaluation_results_euf_context.db`
- `generate_gpu_insights_report.py ... missing ... insights/data/*.csv`

Then scoring has not completed for that run yet. Run:

```bash
EVAL_RUN_GPU=a40 EVAL_RUN_DIR="results/runs/a40/2026-03-01_230442_context_eval" \
python gpu_runtime/evaluate_context_results.py
```

If you see:

- `generate_presentation_qa.py ... missing ... insights/data/model_summary.csv`

Then charts step has not been run yet. Run:

```bash
python insights/generate_context_charts.py --run-dir "results/runs/a40/2026-03-01_230442_context_eval"
```

## 8) Quick Verification

```bash
RUN_DIR="results/runs/a40/2026-03-01_230442_context_eval"

ls -l "$RUN_DIR"/scores/evaluation_scores_euf_context.db \
      "$RUN_DIR"/scores/evaluation_scores_euf_context.xlsx

find "$RUN_DIR"/insights -maxdepth 3 -type f | sort
```
