#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "❌ python not found. Set PYTHON_BIN or install python3."
  exit 1
fi

RUN_DIR=""
ALL_RUNS=0
FORCE=0

usage() {
  cat <<EOF
Run post-scoring insights pipeline.

Usage:
  bash gpu_runtime/run_post_scoring_insights.sh --run-dir <results/runs/<gpu>/<run_id>> [--force]
  bash gpu_runtime/run_post_scoring_insights.sh --all-runs [--force]

Notes:
  - --run-dir: process one run and then refresh that GPU's aggregate report.
  - --all-runs: process all runs and refresh aggregate report for each GPU bucket.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --all-runs)
      ALL_RUNS=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "❌ Unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ -n "$RUN_DIR" && "$ALL_RUNS" -eq 1 ]]; then
  echo "❌ Use either --run-dir or --all-runs, not both."
  exit 2
fi
if [[ -z "$RUN_DIR" && "$ALL_RUNS" -eq 0 ]]; then
  echo "❌ Missing mode: provide --run-dir or --all-runs."
  usage
  exit 2
fi

FORCE_ARG=""
if [[ "$FORCE" -eq 1 ]]; then
  FORCE_ARG="--force"
fi

run_for_one() {
  local rd="$1"
  if [[ ! -d "$rd" ]]; then
    echo "❌ Run dir not found: $rd"
    return 1
  fi
  local gpu
  gpu="$(basename "$(dirname "$rd")")"

  echo "==> Processing run: $rd"
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_charts.py" --run-dir "$rd" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_presentation_qa.py" --run-dir "$rd" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_token_budget.py" --run-dir "$rd" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_vram_docs.py" --run-dir "$rd" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/gpu_efficiency/generate_gpu_efficiency_report.py" --run-dir "$rd" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_gpu_insights_report.py" --gpu "$gpu"
}

if [[ "$ALL_RUNS" -eq 1 ]]; then
  echo "==> Processing all runs under: $REPO_DIR/results/runs"
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_charts.py" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_presentation_qa.py" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_token_budget.py" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_vram_docs.py" $FORCE_ARG
  "$PYTHON_BIN" "$REPO_DIR/insights/gpu_efficiency/generate_gpu_efficiency_report.py" $FORCE_ARG

  runs_root="$REPO_DIR/results/runs"
  if [[ -d "$runs_root" ]]; then
    for gpu_dir in "$runs_root"/*; do
      [[ -d "$gpu_dir" ]] || continue
      gpu="$(basename "$gpu_dir")"
      "$PYTHON_BIN" "$REPO_DIR/insights/generate_gpu_insights_report.py" --gpu "$gpu"
    done
  fi
else
  run_for_one "$RUN_DIR"
fi

echo "✅ Post-scoring insights pipeline complete."
