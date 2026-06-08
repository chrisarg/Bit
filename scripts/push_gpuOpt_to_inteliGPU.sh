#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="gpuOpt"
BRANCH_DST="inteliGPU"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"



if [[ "$CURRENT_BRANCH" != "$BRANCH_SRC" ]]; then
  echo "ERROR: must run from branch '$BRANCH_SRC'"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean; commit or stash changes first."
  git status --short
  exit 1
fi

git fetch origin "$BRANCH_DST"
git switch "$BRANCH_DST"
git pull origin "$BRANCH_DST"

git merge "$BRANCH_SRC" --no-ff -m "Merge  '$BRANCH_SRC' into '$BRANCH_DST'"

git switch "$CURRENT_BRANCH"
echo "Switched back to '$CURRENT_BRANCH'"