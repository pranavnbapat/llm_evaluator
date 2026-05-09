#!/bin/bash
# One-time Git identity + PAT credential setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "❌ git is not installed."
  exit 1
fi

DEFAULT_NAME="$(git config user.name || true)"
DEFAULT_EMAIL="$(git config user.email || true)"

read -r -p "Git user.name${DEFAULT_NAME:+ [${DEFAULT_NAME}]}: " INPUT_NAME
read -r -p "Git user.email${DEFAULT_EMAIL:+ [${DEFAULT_EMAIL}]}: " INPUT_EMAIL
GIT_NAME="${INPUT_NAME:-$DEFAULT_NAME}"
GIT_EMAIL="${INPUT_EMAIL:-$DEFAULT_EMAIL}"

if [[ -z "$GIT_NAME" || -z "$GIT_EMAIL" ]]; then
  echo "❌ Git user.name and user.email are required."
  exit 1
fi

git config user.name "$GIT_NAME"
git config user.email "$GIT_EMAIL"

REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$REMOTE_URL" ]]; then
  echo "❌ Could not detect origin remote."
  exit 1
fi

if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
  GH_USER="${BASH_REMATCH[1]}"
else
  read -r -p "GitHub username: " GH_USER
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Paste a GitHub PAT with repo write access (input hidden)."
  read -r -s -p "GITHUB_TOKEN: " GITHUB_TOKEN
  echo ""
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "❌ Empty token. Aborting."
  exit 1
fi

CRED_FILE="${HOME}/.git-credentials"
git config credential.helper "store --file ${CRED_FILE}"
printf "https://%s:%s@github.com\n" "$GH_USER" "$GITHUB_TOKEN" > "${CRED_FILE}"
chmod 600 "${CRED_FILE}"

echo "✅ Git identity configured for this repo."
echo "✅ PAT credential stored at ${CRED_FILE}."
echo "Now pushes should be non-interactive:"
echo "   git push"
