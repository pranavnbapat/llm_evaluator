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
GPU_FILTER=""
FORCE=0

usage() {
  cat <<EOF
Run post-scoring insights pipeline.

Usage:
  bash gpu_runtime/run_post_scoring_insights.sh --run-dir <results/runs/<gpu>/<run_id>> [--force]
  bash gpu_runtime/run_post_scoring_insights.sh --all-runs [--gpu <bucket>] [--force]

Notes:
  - --run-dir: process one run and refresh that GPU's aggregate report.
  - --all-runs: process every run; pair with --gpu to limit to one bucket.
  - Runs without scores/evaluation_scores_euf_context.db are skipped with a warning.
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
    --gpu)
      GPU_FILTER="$2"
      shift 2
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
if [[ -n "$GPU_FILTER" && "$ALL_RUNS" -eq 0 ]]; then
  echo "❌ --gpu only applies with --all-runs."
  exit 2
fi

FORCE_ARG=""
if [[ "$FORCE" -eq 1 ]]; then
  FORCE_ARG="--force"
fi

scores_present() {
  local rd="$1"
  [[ -f "$rd/scores/evaluation_scores_euf_context.db" ]]
}

run_for_one() {
  local rd="$1"
  if [[ ! -d "$rd" ]]; then
    echo "❌ Run dir not found: $rd"
    return 1
  fi
  if ! scores_present "$rd"; then
    echo "⚠️  Skipping $rd: no scores/evaluation_scores_euf_context.db"
    echo "   Run scoring first: bash gpu_runtime/run_evaluate_context_results_background.sh"
    return 0
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
  runs_root="$REPO_DIR/results/runs"
  if [[ ! -d "$runs_root" ]]; then
    echo "❌ No runs directory: $runs_root"
    exit 1
  fi

  if [[ -n "$GPU_FILTER" ]]; then
    gpu_dirs=("$runs_root/$GPU_FILTER")
    if [[ ! -d "${gpu_dirs[0]}" ]]; then
      echo "❌ GPU bucket not found: ${gpu_dirs[0]}"
      exit 1
    fi
    echo "==> Filtering --all-runs to bucket: $GPU_FILTER"
  else
    gpu_dirs=()
    for d in "$runs_root"/*; do
      [[ -d "$d" ]] && gpu_dirs+=("$d")
    done
  fi

  any_processed=0
  for gpu_dir in "${gpu_dirs[@]}"; do
    gpu="$(basename "$gpu_dir")"
    for run_dir in "$gpu_dir"/*; do
      [[ -d "$run_dir" ]] || continue
      if ! scores_present "$run_dir"; then
        echo "⚠️  Skipping $run_dir: no scores/evaluation_scores_euf_context.db"
        continue
      fi
      any_processed=1
      echo "==> Processing run: $run_dir"
      "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_charts.py" --run-dir "$run_dir" $FORCE_ARG
      "$PYTHON_BIN" "$REPO_DIR/insights/generate_presentation_qa.py" --run-dir "$run_dir" $FORCE_ARG
      "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_token_budget.py" --run-dir "$run_dir" $FORCE_ARG
      "$PYTHON_BIN" "$REPO_DIR/insights/generate_context_vram_docs.py" --run-dir "$run_dir" $FORCE_ARG
      "$PYTHON_BIN" "$REPO_DIR/insights/gpu_efficiency/generate_gpu_efficiency_report.py" --run-dir "$run_dir" $FORCE_ARG
    done
    "$PYTHON_BIN" "$REPO_DIR/insights/generate_gpu_insights_report.py" --gpu "$gpu"
  done

  if [[ $any_processed -eq 0 ]]; then
    echo "⚠️  No runs with scores DBs were found under the requested scope."
  fi
else
  run_for_one "$RUN_DIR"
fi

echo "✅ Post-scoring insights pipeline complete."
