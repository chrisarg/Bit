#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="gpuOpt"
BRANCH_DST="inteliGPU"
CURRENT_BRANCH="$(git branch --show-current)"

cd "$(git rev-parse --show-toplevel)"

if ! git show-ref --verify --quiet "refs/heads/${BRANCH_SRC}"; then
  echo "ERROR: branch ${BRANCH_SRC} does not exist locally."
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/${BRANCH_DST}"; then
  echo "ERROR: branch ${BRANCH_DST} does not exist locally."
  exit 1
fi

if [[ "$CURRENT_BRANCH" != "$BRANCH_SRC" ]]; then
  echo "ERROR: must run from branch '$BRANCH_SRC'"
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean; commit or stash changes first."
  git status --short
  exit 1
fi

restore_branch() {
  status=$?

  if git rev-parse --verify --quiet MERGE_HEAD >/dev/null; then
    echo "Merge failed; aborting it before returning to '$CURRENT_BRANCH'." >&2
    git merge --abort >/dev/null 2>&1 || true
  fi

  git switch "$CURRENT_BRANCH" >/dev/null 2>&1 || true
  return "$status"
}
trap restore_branch EXIT

git fetch origin "$BRANCH_DST"
git switch "$BRANCH_DST"
git pull --ff-only origin "$BRANCH_DST"

git merge "$BRANCH_SRC" --no-ff -m "Merge '$BRANCH_SRC' into '$BRANCH_DST'"
git push origin "$BRANCH_DST"

git switch "$CURRENT_BRANCH"
echo "Switched back to '$CURRENT_BRANCH'"
trap - EXIT
