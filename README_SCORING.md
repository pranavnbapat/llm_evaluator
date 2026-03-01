# Scoring Run Guide (RunPod)

This guide runs only the scoring step (`evaluate_context_results.py`) from an existing context-evaluation database.

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
python evaluate_context_results.py
```

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

Created/updated in `results/`:

- `evaluation_scores_euf_context.db`
- `evaluation_scores_euf_context.xlsx`

Logs in `logs/`:

- `evaluate_context_results_<timestamp>.log`
