#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="main"
BRANCH_DST="inteliGPU"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

FILES=(
  Makefile
  README.md
  include/bit.h
  include/simde/*
  src/bit_internal.h
  src/bit.c
  src/bit_gpu.c
  benchmark/benchmark.c
  benchmark/openmp_bit_helpers.c
  benchmark/openmp_bit_helpers.h
  benchmark/openmp_bit.c
  benchmark/openmp_bit_nogpu.c
  benchmark/openmp_bit_container.c
  tests/test_bit.c
  tests/test_offload.c
)

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

CHECKOUT_PATHS=(include "${FILES[@]}")
missing=()
for path in "${CHECKOUT_PATHS[@]}"; do
  if ! git cat-file -e "${BRANCH_SRC}:${path}" 2>/dev/null; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "ERROR: the following paths are not present in ${BRANCH_SRC}:"
  printf "  %s\n" "${missing[@]}"
  exit 1
fi

restore_branch() {
  git switch "$CURRENT_BRANCH" >/dev/null 2>&1 || true
}
trap restore_branch EXIT

git fetch origin "$BRANCH_DST"
git switch "$BRANCH_DST"
git pull --ff-only origin "$BRANCH_DST"

git checkout "$BRANCH_SRC" -- "${CHECKOUT_PATHS[@]}"

git add -- "${CHECKOUT_PATHS[@]}"
if git diff --cached --quiet; then
  echo "No selected-file changes to commit."
else
  git commit -m "Copy selected main files into inteliGPU"
  git push origin "$BRANCH_DST"
fi

git switch "$CURRENT_BRANCH"
echo "Switched back to '$CURRENT_BRANCH'"
trap - EXIT