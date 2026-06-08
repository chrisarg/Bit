#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="gpuOpt"
BRANCH_DST="inteliGPU"


if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH_SRC" ]]; then
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

git switch "$BRANCH_SRC" -m "Switch back to '$BRANCH_SRC' after merging into '$BRANCH_DST'"