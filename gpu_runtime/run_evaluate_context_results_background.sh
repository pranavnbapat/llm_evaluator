#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/logs"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
EVAL_SCRIPT="$REPO_DIR/gpu_runtime/evaluate_context_results.py"

MODE="tmux"
SESSION_NAME="eval_context_scores"
ALL_RUNS=0
ALL_RUNS_EXPLICIT=0
RUN_DIR_ARG=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--mode tmux|nohup] [--session NAME] [--run-dir PATH] [--all-runs]

Starts gpu_runtime/evaluate_context_results.py in a persistent background run.
Defaults:
  --mode tmux
  --session eval_context_scores
  auto mode (when neither --run-dir, --all-runs, EVAL_RUN_DIR nor EVAL_RUN_ID is set):
    score every run under results/runs/<detected_gpu_bucket>/

Single-run resolution order:
  1. --run-dir CLI flag
  2. EVAL_RUN_DIR env
  3. EVAL_RUN_ID env (joined with detected GPU bucket)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --session)
      SESSION_NAME="${2:-}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR_ARG="${2:-}"
      shift 2
      ;;
    --all-runs)
      ALL_RUNS=1
      ALL_RUNS_EXPLICIT=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -n "$RUN_DIR_ARG" && $ALL_RUNS_EXPLICIT -eq 1 ]]; then
  echo "❌ --run-dir and --all-runs are mutually exclusive."
  exit 2
fi

if [[ "$MODE" != "tmux" && "$MODE" != "nohup" ]]; then
  echo "Invalid --mode '$MODE'. Use tmux or nohup."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN"
  echo "Run gpu_runtime/setup.sh first."
  exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Missing evaluation script: $EVAL_SCRIPT"
  exit 1
fi

detect_gpu_bucket() {
  if [[ -n "${EVAL_RUN_GPU:-}" ]]; then
    echo "${EVAL_RUN_GPU,,}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local name
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr '[:upper:]' '[:lower:]')"
    if [[ "$name" == *"b200"* || "$name" == *"gb200"* ]]; then echo "b200"; return; fi
    if [[ "$name" == *"h200"* ]]; then echo "h200_sxm"; return; fi
    if [[ "$name" == *"h100"* ]]; then echo "h100_sxm"; return; fi
    if [[ "$name" == *"l40s"* || "$name" == *"l40"* ]]; then echo "l40s"; return; fi
    if [[ "$name" == *"3090"* ]]; then echo "3090"; return; fi
    if [[ "$name" == *"a100"* ]]; then echo "a100"; return; fi
    if [[ "$name" == *"a40"* ]]; then echo "a40"; return; fi
  fi
  echo "unknown_gpu"
}

GPU_BUCKET="$(detect_gpu_bucket)"
RUNS_BASE="$REPO_DIR/results/runs/$GPU_BUCKET"
if [[ -n "$RUN_DIR_ARG" ]]; then
  RUN_DIR="$RUN_DIR_ARG"
elif [[ -n "${EVAL_RUN_DIR:-}" ]]; then
  RUN_DIR="$EVAL_RUN_DIR"
elif [[ -n "${EVAL_RUN_ID:-}" ]]; then
  RUN_DIR="$REPO_DIR/results/runs/$GPU_BUCKET/$EVAL_RUN_ID"
else
  RUN_DIR=""
fi

# Auto behavior: if no run is pinned (CLI/env) and --all-runs was not set
# explicitly, score every run in the detected GPU bucket.
if [[ $ALL_RUNS_EXPLICIT -eq 0 && -z "$RUN_DIR_ARG" && -z "${EVAL_RUN_DIR:-}" && -z "${EVAL_RUN_ID:-}" ]]; then
  ALL_RUNS=1
fi

if [[ $ALL_RUNS -eq 1 ]]; then
  LOG_DIR="$REPO_DIR/logs"
elif [[ -n "$RUN_DIR" ]]; then
  LOG_DIR="$RUN_DIR/logs"
fi
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/evaluate_context_results_${TS}.log"

cd "$REPO_DIR"

if [[ "$MODE" == "tmux" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Falling back to nohup mode."
    MODE="nohup"
  fi
fi

if [[ "$MODE" == "tmux" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
  fi

  if [[ $ALL_RUNS -eq 1 ]]; then
    CMD="cd '$REPO_DIR' && found=0; for d in '$RUNS_BASE'/*; do [ -d \"\$d\" ] || continue; db=\"\$d/raw/evaluation_results_euf_context.db\"; [ -f \"\$db\" ] || continue; found=1; mkdir -p \"\$d/logs\"; ts=\$(date +%Y%m%d_%H%M%S); run_log=\"\$d/logs/evaluate_context_results_\${ts}.log\"; echo \"Scoring run: \$d\"; EVAL_RUN_GPU='$GPU_BUCKET' EVAL_RUN_DIR=\"\$d\" '$PYTHON_BIN' '$EVAL_SCRIPT' 2>&1 | tee \"\$run_log\"; done; if [ \"\$found\" -eq 0 ]; then echo \"No run DBs found under $RUNS_BASE\"; fi"
  else
    CMD="cd '$REPO_DIR' && EVAL_RUN_GPU='$GPU_BUCKET'"
    if [[ -n "$RUN_DIR" ]]; then
      CMD="$CMD EVAL_RUN_DIR='$RUN_DIR'"
    fi
    CMD="$CMD '$PYTHON_BIN' '$EVAL_SCRIPT' 2>&1 | tee '$LOG_FILE'"
  fi
  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "Started in tmux session: $SESSION_NAME"
  if [[ $ALL_RUNS -eq 1 ]]; then
    echo "Mode: all-runs"
    echo "Runs base: $RUNS_BASE"
  elif [[ -n "$RUN_DIR" ]]; then
    echo "Run dir: $RUN_DIR"
  fi
  echo "GPU bucket: $GPU_BUCKET"
  if [[ $ALL_RUNS -eq 0 ]]; then
    echo "Log file: $LOG_FILE"
  fi
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Detach: Ctrl+b then d"
  exit 0
fi

if [[ $ALL_RUNS -eq 1 ]]; then
  nohup bash -lc "cd '$REPO_DIR' && found=0; for d in '$RUNS_BASE'/*; do [ -d \"\$d\" ] || continue; db=\"\$d/raw/evaluation_results_euf_context.db\"; [ -f \"\$db\" ] || continue; found=1; mkdir -p \"\$d/logs\"; ts=\$(date +%Y%m%d_%H%M%S); run_log=\"\$d/logs/evaluate_context_results_\${ts}.log\"; echo \"Scoring run: \$d\"; EVAL_RUN_GPU='$GPU_BUCKET' EVAL_RUN_DIR=\"\$d\" '$PYTHON_BIN' '$EVAL_SCRIPT' >\"\$run_log\" 2>&1; done; if [ \"\$found\" -eq 0 ]; then echo \"No run DBs found under $RUNS_BASE\"; fi" >"$LOG_FILE" 2>&1 &
elif [[ -n "$RUN_DIR" ]]; then
  EVAL_RUN_GPU="$GPU_BUCKET" EVAL_RUN_DIR="$RUN_DIR" nohup "$PYTHON_BIN" "$EVAL_SCRIPT" >"$LOG_FILE" 2>&1 &
else
  EVAL_RUN_GPU="$GPU_BUCKET" nohup "$PYTHON_BIN" "$EVAL_SCRIPT" >"$LOG_FILE" 2>&1 &
fi
PID=$!
PID_FILE="$LOG_DIR/evaluate_context_results_${TS}.pid"
echo "$PID" > "$PID_FILE"
ln -sfn "$(basename "$PID_FILE")" "$LOG_DIR/evaluate_context_results.pid"

echo "Started with nohup (pid: $PID)"
if [[ $ALL_RUNS -eq 1 ]]; then
  echo "Mode: all-runs"
  echo "Runs base: $RUNS_BASE"
elif [[ -n "$RUN_DIR" ]]; then
  echo "Run dir: $RUN_DIR"
fi
echo "GPU bucket: $GPU_BUCKET"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE  (symlinked: $LOG_DIR/evaluate_context_results.pid)"
echo "Tail logs: tail -f '$LOG_FILE'"
echo "Stop: kill $PID"
