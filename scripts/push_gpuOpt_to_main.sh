#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="gpuOpt"
BRANCH_DST="main"

FILES=(
  Makefile
  README.md
  src/bit.c
  benchmark/benchmark.c
  benchmark/openmp_bit_helpers.c
  benchmark/openmp_bit_helpers.h
  benchmark/openmp_bit.c
  benchmark/openmp_bit_nogpu.c
  tests/test_bit.c
  tests/test_offload.c
)

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

git checkout "$BRANCH_SRC" -- include "${FILES[@]}"

git add include "${FILES[@]}"
git commit -m "Cherry-pick selected gpuOpt files into main"
git push origin "$BRANCH_DST"