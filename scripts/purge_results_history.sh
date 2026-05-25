#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: purge_results_history.sh [--repo-dir PATH] [--path PATH]... [--push] [--yes]

Rewrite git history to remove committed artifacts such as results/ from all commits.

Options:
  --repo-dir PATH   Repository to rewrite. Default: current directory.
  --path PATH       Path to purge from history. Repeatable. Default: results
  --push            After rewrite, force-push branches and tags to origin.
  --yes             Skip interactive confirmation.
  -h, --help        Show this help.

Examples:
  scripts/purge_results_history.sh
  scripts/purge_results_history.sh --path results --path gpu_runtime/results
  scripts/purge_results_history.sh --repo-dir /path/to/repo --push

Notes:
  - This rewrites commit history and changes commit hashes.
  - Everyone using the repository must re-sync after the force-push.
  - Old data may still exist in forks, clones, and remote caches until cleaned there too.
USAGE
}

REPO_DIR="."
DO_PUSH=0
ASSUME_YES=0
declare -a PURGE_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="${2:-}"
      shift 2
      ;;
    --path)
      PURGE_PATHS+=("${2:-}")
      shift 2
      ;;
    --push)
      DO_PUSH=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#PURGE_PATHS[@]} -eq 0 ]]; then
  PURGE_PATHS=("results")
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed." >&2
  exit 1
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is not installed." >&2
  echo "Install it first, for example: pip install git-filter-repo" >&2
  exit 1
fi

cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository: $REPO_DIR" >&2
  exit 1
fi

ABS_REPO_DIR="$(pwd)"
REMOTE_NAME="$(git remote 2>/dev/null | head -n1 || true)"

echo "Repository: $ABS_REPO_DIR"
echo "Paths to purge from history:"
for path in "${PURGE_PATHS[@]}"; do
  echo "  - $path"
done
if [[ $DO_PUSH -eq 1 ]]; then
  echo "Push mode: force-push branches and tags to origin after rewrite"
else
  echo "Push mode: disabled"
fi

if [[ $ASSUME_YES -ne 1 ]]; then
  echo
  echo "This operation rewrites git history and changes commit hashes."
  read -r -p "Continue? [y/N] " answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

declare -a FILTER_ARGS=()
for path in "${PURGE_PATHS[@]}"; do
  FILTER_ARGS+=(--path "$path")
done
FILTER_ARGS+=(--invert-paths --force)

echo
echo "Rewriting history..."
git filter-repo "${FILTER_ARGS[@]}"

echo
echo "History rewrite complete."
echo "Recommended verification:"
echo "  git log --stat -- ${PURGE_PATHS[0]}"

if [[ $DO_PUSH -eq 1 ]]; then
  if [[ -z "$REMOTE_NAME" ]]; then
    echo
    echo "No git remote is configured."
    echo "Add one first, for example:"
    echo "  git remote add origin <repo-url>"
    echo "Then force-push manually:"
    echo "  git push --force --all origin"
    echo "  git push --force --tags origin"
    exit 1
  fi
  echo
  echo "Force-pushing rewritten history to $REMOTE_NAME..."
  git push --force --all "$REMOTE_NAME"
  git push --force --tags "$REMOTE_NAME"
  echo "Force-push complete."
else
  echo
  if [[ -n "$REMOTE_NAME" ]]; then
    echo "Next steps:"
    echo "  git push --force --all $REMOTE_NAME"
    echo "  git push --force --tags $REMOTE_NAME"
  else
    echo "No git remote is configured."
    echo "Next steps:"
    echo "  git remote add origin <repo-url>"
    echo "  git push --force --all origin"
    echo "  git push --force --tags origin"
  fi
fi
