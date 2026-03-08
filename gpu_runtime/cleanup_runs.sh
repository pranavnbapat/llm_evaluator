#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for results/runs/<gpu_bucket>/<run_id>.
# Default mode is dry-run. Use --apply to actually delete.
#
# Examples:
#   bash runpod_setup/cleanup_runs.sh --gpu-bucket a40 --prune-partial
#   bash runpod_setup/cleanup_runs.sh --gpu-bucket a40 --keep-latest 2
#   bash runpod_setup/cleanup_runs.sh --gpu-bucket a40 --prune-partial --keep-latest 2 --apply

GPU_BUCKET="a40"
RESULTS_ROOT="results/runs"
KEEP_LATEST=""
PRUNE_PARTIAL=0
APPLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --gpu-bucket <name>    GPU bucket directory under results/runs (default: a40)
  --results-root <path>  Base runs root (default: results/runs)
  --keep-latest <N>      Keep only latest N run directories by name, delete older ones
  --prune-partial        Delete runs missing raw/evaluation_results_euf_context.db
  --apply                Apply deletions (without this flag, dry-run only)
  -h, --help             Show this help

Notes:
  - Run directory naming is expected as sortable timestamps, e.g. 2026-03-07_104215_context_eval.
  - --keep-latest and --prune-partial can be combined.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-bucket)
      GPU_BUCKET="${2:-}"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --keep-latest)
      KEEP_LATEST="${2:-}"
      shift 2
      ;;
    --prune-partial)
      PRUNE_PARTIAL=1
      shift 1
      ;;
    --apply)
      APPLY=1
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

if [[ -n "$KEEP_LATEST" ]]; then
  if ! [[ "$KEEP_LATEST" =~ ^[0-9]+$ ]]; then
    echo "Error: --keep-latest must be a non-negative integer."
    exit 1
  fi
fi

RUNS_DIR="${RESULTS_ROOT}/${GPU_BUCKET}"
if [[ ! -d "$RUNS_DIR" ]]; then
  echo "No such runs directory: $RUNS_DIR"
  exit 0
fi

echo "Runs dir: $RUNS_DIR"
if [[ $APPLY -eq 0 ]]; then
  echo "Mode: dry-run (no files will be deleted). Use --apply to execute."
else
  echo "Mode: apply (deletions will be executed)."
fi

declare -a DELETE_LIST=()

mapfile -t RUN_DIRS < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "No run directories found."
  exit 0
fi

if [[ $PRUNE_PARTIAL -eq 1 ]]; then
  echo ""
  echo "Checking for partial runs (missing raw/evaluation_results_euf_context.db)..."
  for d in "${RUN_DIRS[@]}"; do
    db="$d/raw/evaluation_results_euf_context.db"
    if [[ ! -f "$db" ]]; then
      DELETE_LIST+=("$d")
      echo "  mark partial: $d"
    fi
  done
fi

if [[ -n "$KEEP_LATEST" ]]; then
  echo ""
  echo "Applying keep-latest policy: keep newest $KEEP_LATEST run(s)"
  mapfile -t SORTED_BASENAMES < <(for d in "${RUN_DIRS[@]}"; do basename "$d"; done | sort)

  if [[ "$KEEP_LATEST" -lt "${#SORTED_BASENAMES[@]}" ]]; then
    CUTOFF=$(( ${#SORTED_BASENAMES[@]} - KEEP_LATEST ))
    for ((i=0; i<CUTOFF; i++)); do
      d="${RUNS_DIR}/${SORTED_BASENAMES[$i]}"
      DELETE_LIST+=("$d")
      echo "  mark old:     $d"
    done
  else
    echo "  nothing to delete by keep-latest."
  fi
fi

# Deduplicate delete list
if [[ ${#DELETE_LIST[@]} -gt 0 ]]; then
  mapfile -t UNIQUE_DELETE < <(printf '%s\n' "${DELETE_LIST[@]}" | sort -u)
else
  UNIQUE_DELETE=()
fi

echo ""
if [[ ${#UNIQUE_DELETE[@]} -eq 0 ]]; then
  echo "Nothing to delete."
  exit 0
fi

echo "Planned deletions (${#UNIQUE_DELETE[@]}):"
for d in "${UNIQUE_DELETE[@]}"; do
  if [[ -d "$d" ]]; then
    du -sh "$d" 2>/dev/null || true
  else
    echo "  skip missing: $d"
  fi
done

if [[ $APPLY -eq 1 ]]; then
  echo ""
  echo "Deleting..."
  for d in "${UNIQUE_DELETE[@]}"; do
    if [[ -d "$d" ]]; then
      rm -rf "$d"
      echo "  deleted: $d"
    fi
  done
  echo "Done."
else
  echo ""
  echo "Dry-run complete. Re-run with --apply to delete."
fi
