#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_DIR/logs"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_context.py"

MODE="tmux"
SESSION_NAME="eval_context"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--mode tmux|nohup] [--session NAME]

Starts runpod_setup/evaluate_context.py in a persistent background run.
Defaults:
  --mode tmux
  --session eval_context
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

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/evaluate_context_${TS}.log"

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

  CMD="cd '$SCRIPT_DIR' && '$PYTHON_BIN' '$EVAL_SCRIPT' 2>&1 | tee '$LOG_FILE'"
  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "Started in tmux session: $SESSION_NAME"
  echo "Log file: $LOG_FILE"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Detach: Ctrl+b then d"
  exit 0
fi

nohup "$PYTHON_BIN" "$EVAL_SCRIPT" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/evaluate_context.pid"

echo "Started with nohup (pid: $PID)"
echo "Log file: $LOG_FILE"
echo "PID file: $LOG_DIR/evaluate_context.pid"
echo "Tail logs: tail -f '$LOG_FILE'"
echo "Stop: kill $PID"
