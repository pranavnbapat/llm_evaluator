#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
SCORE_SCRIPT="$SCRIPT_DIR/evaluate_vision_results.py"

MODE="tmux"
SESSION_NAME="eval_vision_results"
DATASET_PATH=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--mode tmux|nohup] [--session NAME] [--dataset PATH]

Starts gpu_runtime/evaluate_vision_results.py in a persistent background run.
Defaults:
  --mode tmux
  --session eval_vision_results
  --dataset read from EVAL_VISION_DATASET or run metadata
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
    --dataset)
      DATASET_PATH="${2:-}"
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
  echo "Run gpu_runtime/setup.sh first."
  exit 1
fi

if [[ ! -f "$SCORE_SCRIPT" ]]; then
  echo "Missing scoring script: $SCORE_SCRIPT"
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
RUN_ID="${EVAL_RUN_ID:-}"
RUN_DIR="${EVAL_RUN_DIR:-}"
if [[ -z "$RUN_DIR" && -n "$RUN_ID" ]]; then
  RUN_DIR="$REPO_DIR/results/runs/$GPU_BUCKET/$RUN_ID"
fi
if [[ -z "$RUN_DIR" ]]; then
  echo "Set EVAL_RUN_DIR (preferred) or EVAL_RUN_ID before running this scorer."
  exit 1
fi

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/evaluate_vision_results_${TS}.log"

cd "$REPO_DIR"

if [[ "$MODE" == "tmux" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed. Falling back to nohup mode."
    MODE="nohup"
  fi
fi

DATASET_ENV=""
if [[ -n "$DATASET_PATH" ]]; then
  DATASET_ENV="EVAL_VISION_DATASET='$DATASET_PATH'"
fi

if [[ "$MODE" == "tmux" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
  fi

  CMD="cd '$SCRIPT_DIR' && EVAL_RUN_GPU='$GPU_BUCKET' EVAL_RUN_DIR='$RUN_DIR' $DATASET_ENV '$PYTHON_BIN' '$SCORE_SCRIPT' 2>&1 | tee '$LOG_FILE'"
  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "Started in tmux session: $SESSION_NAME"
  echo "Run dir: $RUN_DIR"
  echo "GPU bucket: $GPU_BUCKET"
  echo "Log file: $LOG_FILE"
  [[ -n "$DATASET_PATH" ]] && echo "Dataset: $DATASET_PATH"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Detach: Ctrl+b then d"
  exit 0
fi

if [[ -n "$DATASET_PATH" ]]; then
  EVAL_RUN_GPU="$GPU_BUCKET" EVAL_RUN_DIR="$RUN_DIR" \
    EVAL_VISION_DATASET="$DATASET_PATH" \
    nohup "$PYTHON_BIN" "$SCORE_SCRIPT" >"$LOG_FILE" 2>&1 &
else
  EVAL_RUN_GPU="$GPU_BUCKET" EVAL_RUN_DIR="$RUN_DIR" \
    nohup "$PYTHON_BIN" "$SCORE_SCRIPT" >"$LOG_FILE" 2>&1 &
fi
PID=$!
echo "$PID" > "$LOG_DIR/evaluate_vision_results.pid"

echo "Started with nohup (pid: $PID)"
echo "Run dir: $RUN_DIR"
echo "GPU bucket: $GPU_BUCKET"
echo "Log file: $LOG_FILE"
echo "PID file: $LOG_DIR/evaluate_vision_results.pid"
[[ -n "$DATASET_PATH" ]] && echo "Dataset: $DATASET_PATH"
echo "Tail logs: tail -f '$LOG_FILE'"
echo "Stop: kill $PID"
