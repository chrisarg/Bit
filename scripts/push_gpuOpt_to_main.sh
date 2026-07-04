#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="gpuOpt"
BRANCH_DST="main"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
FILES=(
  Makefile
  README.md
  src/*
  benchmark/benchmark.c
  benchmark/openmp_bit_helpers.c
  benchmark/openmp_bit_helpers.h
  benchmark/openmp_bit.c
  benchmark/openmp_bit_nogpu.c
  tests/test_bit.c
  tests/test_offload.c
)

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
  git switch "$CURRENT_BRANCH" >/dev/null 2>&1 || true
}
trap restore_branch EXIT

git fetch origin "$BRANCH_DST"
git switch "$BRANCH_DST"
git pull --ff-only origin "$BRANCH_DST"

git checkout "$BRANCH_SRC" -- include "${FILES[@]}"

git add include "${FILES[@]}"
if git diff --cached --quiet; then
  echo "No selected-file changes to commit."
else
  git commit -m "Copy selected gpuOpt files into main"
  git push origin "$BRANCH_DST"
fi

git switch "$CURRENT_BRANCH"
echo "Switched back to '$CURRENT_BRANCH'"
trap - EXIT