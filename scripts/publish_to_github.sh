#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/publish_to_github.sh <git-remote-url> [branch]" >&2
  exit 2
fi

remote_url="$1"
branch="${2:-main}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ -n "$(git status --short)" ]; then
  echo "Working tree is not clean. Commit or discard changes before publishing." >&2
  git status --short >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
if [ "$current_branch" != "$branch" ]; then
  git branch -M "$branch"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$remote_url"
else
  git remote add origin "$remote_url"
fi

git push -u origin "$branch"
