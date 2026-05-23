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
ALL_RUNS=1
GPU_FILTER=""
OUT_XLSX=""

usage() {
  cat <<EOF
Build simplified model comparison workbooks from context-evaluation runs.

Usage:
  bash gpu_runtime/run_build_model_comparison_workbooks.sh
  bash gpu_runtime/run_build_model_comparison_workbooks.sh --gpu <bucket>
  bash gpu_runtime/run_build_model_comparison_workbooks.sh --run-dir <results/runs/<gpu>/<run_id>> [--out-xlsx <path>]

Defaults:
  - With no arguments, process every context-evaluation run under results/runs/*/*
  - With --gpu, process every context-evaluation run under results/runs/<gpu>/*
  - With --run-dir, process exactly one run

Output:
  - Default per-run output path:
      <run_dir>/insights/model_comparisons_for_RAG_simplified.xlsx
  - Template workbook source, when present:
      results/runs/<gpu>/model_comparisons_for_RAG_sheet2_filled_with_results.xlsx
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      ALL_RUNS=0
      shift 2
      ;;
    --gpu)
      GPU_FILTER="$2"
      shift 2
      ;;
    --out-xlsx)
      OUT_XLSX="$2"
      shift 2
      ;;
    --all-runs)
      ALL_RUNS=1
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

if [[ -n "$RUN_DIR" && -n "$GPU_FILTER" ]]; then
  echo "❌ Use either --run-dir or --gpu, not both."
  exit 2
fi

if [[ -n "$OUT_XLSX" && -z "$RUN_DIR" ]]; then
  echo "❌ --out-xlsx only applies with --run-dir."
  exit 2
fi

is_context_run() {
  local rd="$1"
  [[ -f "$rd/raw/evaluation_results_euf_context.db" && -f "$rd/scores/evaluation_scores_euf_context.db" ]]
}

run_for_one() {
  local rd="$1"
  if [[ ! -d "$rd" ]]; then
    echo "❌ Run dir not found: $rd"
    return 1
  fi
  if ! is_context_run "$rd"; then
    echo "⚠️  Skipping $rd: missing context raw/scores DBs"
    return 0
  fi

  local gpu_dir
  gpu_dir="$(dirname "$rd")"
  local template="$gpu_dir/model_comparisons_for_RAG_sheet2_filled_with_results.xlsx"
  local out="$rd/insights/model_comparisons_for_RAG_simplified.xlsx"
  if [[ -n "$OUT_XLSX" ]]; then
    out="$OUT_XLSX"
  fi

  mkdir -p "$(dirname "$out")"
  echo "==> Building workbook for: $rd"
  if [[ -f "$template" ]]; then
    "$PYTHON_BIN" "$REPO_DIR/gpu_runtime/build_model_comparison_workbook.py" \
      --run-dir "$rd" \
      --template-xlsx "$template" \
      --out-xlsx "$out"
  else
    "$PYTHON_BIN" "$REPO_DIR/gpu_runtime/build_model_comparison_workbook.py" \
      --run-dir "$rd" \
      --out-xlsx "$out"
  fi
}

if [[ -n "$RUN_DIR" ]]; then
  run_for_one "$RUN_DIR"
  echo "✅ Comparison workbook complete."
  exit 0
fi

runs_root="$REPO_DIR/results/runs"
if [[ ! -d "$runs_root" ]]; then
  echo "❌ No runs directory: $runs_root"
  exit 1
fi

gpu_dirs=()
if [[ -n "$GPU_FILTER" ]]; then
  gpu_dirs=("$runs_root/$GPU_FILTER")
  if [[ ! -d "${gpu_dirs[0]}" ]]; then
    echo "❌ GPU bucket not found: ${gpu_dirs[0]}"
    exit 1
  fi
else
  for d in "$runs_root"/*; do
    [[ -d "$d" ]] && gpu_dirs+=("$d")
  done
fi

processed=0
for gpu_dir in "${gpu_dirs[@]}"; do
  for run_dir in "$gpu_dir"/*; do
    [[ -d "$run_dir" ]] || continue
    if ! is_context_run "$run_dir"; then
      continue
    fi
    processed=1
    OUT_XLSX=""
    run_for_one "$run_dir"
  done
done

if [[ "$processed" -eq 0 ]]; then
  echo "⚠️  No context-evaluation runs with raw+scores DBs were found."
  exit 0
fi

echo "✅ Comparison workbook generation complete."
