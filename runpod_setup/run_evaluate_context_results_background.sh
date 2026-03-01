#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/logs"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
EVAL_SCRIPT="$REPO_DIR/runpod_setup/evaluate_context_results.py"

MODE="tmux"
SESSION_NAME="eval_context_scores"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--mode tmux|nohup] [--session NAME]

Starts runpod_setup/evaluate_context_results.py in a persistent background run.
Defaults:
  --mode tmux
  --session eval_context_scores
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

if [[ "$MODE" != "tmux" && "$MODE" != "nohup" ]]; then
  echo "Invalid --mode '$MODE'. Use tmux or nohup."
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv python: $PYTHON_BIN"
  echo "Run runpod_setup/setup.sh first."
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
    if [[ "$name" == *"a100"* ]]; then echo "a100"; return; fi
    if [[ "$name" == *"a40"* ]]; then echo "a40"; return; fi
  fi
  echo "unknown_gpu"
}

GPU_BUCKET="$(detect_gpu_bucket)"
if [[ -n "${EVAL_RUN_DIR:-}" ]]; then
  RUN_DIR="$EVAL_RUN_DIR"
elif [[ -n "${EVAL_RUN_ID:-}" ]]; then
  RUN_DIR="$REPO_DIR/results/runs/$GPU_BUCKET/$EVAL_RUN_ID"
elif [[ -L "$REPO_DIR/results/latest/$GPU_BUCKET" || -e "$REPO_DIR/results/latest/$GPU_BUCKET" ]]; then
  RUN_DIR="$(readlink -f "$REPO_DIR/results/latest/$GPU_BUCKET")"
else
  RUN_DIR=""
fi

if [[ -n "$RUN_DIR" ]]; then
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

  CMD="cd '$REPO_DIR' && EVAL_RUN_GPU='$GPU_BUCKET'"
  if [[ -n "$RUN_DIR" ]]; then
    CMD="$CMD EVAL_RUN_DIR='$RUN_DIR'"
  fi
  CMD="$CMD '$PYTHON_BIN' '$EVAL_SCRIPT' 2>&1 | tee '$LOG_FILE'"
  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "Started in tmux session: $SESSION_NAME"
  if [[ -n "$RUN_DIR" ]]; then echo "Run dir: $RUN_DIR"; fi
  echo "GPU bucket: $GPU_BUCKET"
  echo "Log file: $LOG_FILE"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Detach: Ctrl+b then d"
  exit 0
fi

if [[ -n "$RUN_DIR" ]]; then
  EVAL_RUN_GPU="$GPU_BUCKET" EVAL_RUN_DIR="$RUN_DIR" nohup "$PYTHON_BIN" "$EVAL_SCRIPT" >"$LOG_FILE" 2>&1 &
else
  EVAL_RUN_GPU="$GPU_BUCKET" nohup "$PYTHON_BIN" "$EVAL_SCRIPT" >"$LOG_FILE" 2>&1 &
fi
PID=$!
echo "$PID" > "$LOG_DIR/evaluate_context_results.pid"

echo "Started with nohup (pid: $PID)"
if [[ -n "$RUN_DIR" ]]; then echo "Run dir: $RUN_DIR"; fi
echo "GPU bucket: $GPU_BUCKET"
echo "Log file: $LOG_FILE"
echo "PID file: $LOG_DIR/evaluate_context_results.pid"
echo "Tail logs: tail -f '$LOG_FILE'"
echo "Stop: kill $PID"
