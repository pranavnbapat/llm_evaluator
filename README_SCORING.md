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
