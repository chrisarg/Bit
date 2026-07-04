#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="main"
BRANCH_DST="gpuOpt"

DIRS=(
  include
)

FILES=(
  Makefile
  README.md
  include/bit.h
  include/bit_internal.h
  src/bit.c
  src/bit_internal.c
  benchmark/benchmark.c
  benchmark/openmp_bit_helpers.c
  benchmark/openmp_bit_helpers.h
  benchmark/openmp_bit.c
  benchmark/openmp_bit_nogpu.c
  tests/test_bit.c
  tests/test_offload.c
)

cd "$(git rev-parse --show-toplevel)"

if ! git show-ref --verify --quiet "refs/heads/${BRANCH_SRC}"; then
  echo "ERROR: branch ${BRANCH_SRC} does not exist locally."
  exit 1
fi

git fetch origin "${BRANCH_DST}"
git switch "${BRANCH_DST}"
git pull origin "${BRANCH_DST}"

missing=()
checkout_paths=()

for path in "${DIRS[@]}" "${FILES[@]}"; do
  if git ls-tree --name-only -r "${BRANCH_SRC}" -- "$path" >/dev/null 2>&1; then
    checkout_paths+=("$path")
  else
    missing+=("$path")
  fi
done

if (( ${#missing[@]} )); then
  echo "ERROR: the following paths are not present in ${BRANCH_SRC}:"
  printf "  %s\n" "${missing[@]}"
  exit 1
fi

git checkout "${BRANCH_SRC}" -- "${checkout_paths[@]}"
git add "${checkout_paths[@]}"
git commit -m "Copy selected gpuOpt files into main"
git push origin "${BRANCH_DST}"