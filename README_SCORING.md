# Scoring Run Guide (RunPod)

This guide runs only the scoring step (`runpod_setup/evaluate_context_results.py`) from an existing context-evaluation database.

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

## 3) Run Scoring (Foreground)

```bash
python runpod_setup/evaluate_context_results.py
```

By default, scoring resolves paths in this order:

1. `EVAL_RUN_DIR` (if set)
2. `results/latest/<detected_gpu_bucket>` (for example `results/latest/b200`)
3. legacy `results/` (fallback)

## 4) Run Scoring Detached (tmux background)

```bash
bash runpod_setup/run_evaluate_context_results_background.sh
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

- `runpod_setup/evaluate_context_results.py` performs a full rescore for the selected run.
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

1. `runpod_setup/evaluate_context_results.py` (creates `scores/*.db` + `scores/*.xlsx`)
2. `insights/generate_context_charts.py` (creates `insights/data/*.csv` + `insights/charts/*.png`)
3. `insights/generate_context_insights.py` (creates `insights/EVALUATION_CONTEXT_REPORT.md`)
4. `insights/generate_presentation_qa.py` (creates `insights/Presentation_QA.md` + `insights/data/presentation_qa.*`)
5. `insights/gpu_efficiency/generate_gpu_efficiency_report.py` (creates `insights/gpu_efficiency/*`)

Single run (recommended):

```bash
RUN_DIR="results/runs/a40/2026-03-01_230442_context_eval"

EVAL_RUN_GPU=a40 EVAL_RUN_DIR="$RUN_DIR" \
python runpod_setup/evaluate_context_results.py

python insights/generate_context_charts.py --run-dir "$RUN_DIR"
python insights/generate_context_insights.py --run-dir "$RUN_DIR"
python insights/generate_presentation_qa.py --run-dir "$RUN_DIR"
python insights/gpu_efficiency/generate_gpu_efficiency_report.py --run-dir "$RUN_DIR"
```

Bulk mode (all runs under `results/runs/*/*`):

```bash
python insights/generate_context_charts.py
python insights/generate_context_insights.py
python insights/generate_presentation_qa.py
python insights/gpu_efficiency/generate_gpu_efficiency_report.py
```

Bulk mode behavior:

- Skips runs that are already complete.
- Generates only missing artifacts.
- Use `--force` to regenerate.

## 7) Typical "Missing Input" Messages

If you see:

- `generate_context_charts.py ... missing ... scores/evaluation_scores_euf_context.db`
- `generate_context_insights.py ... missing ... scores/evaluation_scores_euf_context.xlsx`

Then scoring has not completed for that run yet. Run:

```bash
EVAL_RUN_GPU=a40 EVAL_RUN_DIR="results/runs/a40/2026-03-01_230442_context_eval" \
python runpod_setup/evaluate_context_results.py
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
