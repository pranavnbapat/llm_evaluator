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

Starts gpu_runtime/evaluate_context.py in a persistent background run.
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
  echo "Run gpu_runtime/setup.sh first."
  exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Missing evaluation script: $EVAL_SCRIPT"
  exit 1
fi

read_env_file_value() {
  local key="$1"
  local env_file="$SCRIPT_DIR/.env"
  if [[ ! -f "$env_file" ]]; then
    return 1
  fi
  local line
  line="$(grep -E "^[[:space:]]*${key}=" "$env_file" | tail -n1 || true)"
  if [[ -z "$line" ]]; then
    return 1
  fi
  line="${line#*=}"
  line="${line%%#*}"
  line="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
  printf '%s' "$line"
}

resolve_context_question_families() {
  local raw="${EVAL_CONTEXT_QUESTION_FAMILIES+x}"
  if [[ -n "$raw" ]]; then
    printf '%s' "$EVAL_CONTEXT_QUESTION_FAMILIES"
    return
  fi
  local from_file
  from_file="$(read_env_file_value EVAL_CONTEXT_QUESTION_FAMILIES || true)"
  if [[ -n "$from_file" ]]; then
    printf '%s' "$from_file"
    return
  fi
  printf '3'
}

resolve_context_languages() {
  local raw_set="${EVAL_CONTEXT_LANGUAGES+x}"
  local raw=""
  if [[ -n "$raw_set" ]]; then
    raw="$EVAL_CONTEXT_LANGUAGES"
  else
    raw="$(read_env_file_value EVAL_CONTEXT_LANGUAGES || true)"
  fi
  raw="$(printf '%s' "$raw" | tr -d '[:space:]')"
  if [[ -z "$raw" ]]; then
    printf 'EN'
    return
  fi

  local expanded=()
  local part
  IFS=',' read -r -a parts <<< "$raw"
  local eu_codes=(BG HR CS DA NL EN ET FI FR DE EL HU GA IT LV LT MT PL PT RO SK SL ES SV)
  for part in "${parts[@]}"; do
    part="${part^^}"
    [[ -z "$part" ]] && continue
    if [[ "$part" == "EU" ]]; then
      local code
      for code in "${eu_codes[@]}"; do
        if [[ ! " ${expanded[*]} " =~ " ${code} " ]]; then
          expanded+=("$code")
        fi
      done
      continue
    fi
    if [[ ! " ${expanded[*]} " =~ " ${part} " ]]; then
      expanded+=("$part")
    fi
  done
  if [[ ${#expanded[@]} -eq 0 ]]; then
    printf 'EN'
  else
    local joined
    joined="$(IFS=,; printf '%s' "${expanded[*]}")"
    printf '%s' "$joined"
  fi
}

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

CONTEXT_QUESTION_FAMILIES="$(resolve_context_question_families)"
CONTEXT_LANGUAGES="$(resolve_context_languages)"

GPU_BUCKET="$(detect_gpu_bucket)"
RUN_ID="${EVAL_RUN_ID:-$(date +%Y-%m-%d_%H%M%S)_context_eval}"
RUN_DIR="${EVAL_RUN_DIR:-$REPO_DIR/results/runs/$GPU_BUCKET/$RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
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

  CMD="cd '$SCRIPT_DIR' && EVAL_RUN_GPU='$GPU_BUCKET' EVAL_RUN_ID='$RUN_ID' EVAL_RUN_DIR='$RUN_DIR' EVAL_CONTEXT_QUESTION_FAMILIES='$CONTEXT_QUESTION_FAMILIES' EVAL_CONTEXT_LANGUAGES='$CONTEXT_LANGUAGES' '$PYTHON_BIN' '$EVAL_SCRIPT' 2>&1 | tee '$LOG_FILE'"
  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "Started in tmux session: $SESSION_NAME"
  echo "Run dir: $RUN_DIR"
  echo "GPU bucket: $GPU_BUCKET"
  echo "Run id: $RUN_ID"
  echo "Log file: $LOG_FILE"
  echo "Question families: $CONTEXT_QUESTION_FAMILIES"
  echo "Languages: $CONTEXT_LANGUAGES"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Detach: Ctrl+b then d"
  exit 0
fi

EVAL_RUN_GPU="$GPU_BUCKET" EVAL_RUN_ID="$RUN_ID" EVAL_RUN_DIR="$RUN_DIR" EVAL_CONTEXT_QUESTION_FAMILIES="$CONTEXT_QUESTION_FAMILIES" EVAL_CONTEXT_LANGUAGES="$CONTEXT_LANGUAGES" nohup "$PYTHON_BIN" "$EVAL_SCRIPT" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/evaluate_context.pid"

echo "Started with nohup (pid: $PID)"
echo "Run dir: $RUN_DIR"
echo "GPU bucket: $GPU_BUCKET"
echo "Run id: $RUN_ID"
echo "Log file: $LOG_FILE"
echo "Question families: $CONTEXT_QUESTION_FAMILIES"
echo "Languages: $CONTEXT_LANGUAGES"
echo "PID file: $LOG_DIR/evaluate_context.pid"
echo "Tail logs: tail -f '$LOG_FILE'"
echo "Stop: kill $PID"
