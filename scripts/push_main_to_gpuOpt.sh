#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="main"
BRANCH_DST="gpuOpt"
CURRENT_BRANCH="$(git branch --show-current)"

cd "$(git rev-parse --show-toplevel)"

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

mapfile -t SYNC_PATHS < <(
  git ls-tree -r --name-only "$BRANCH_SRC" -- "${CHECKOUT_PATHS[@]}"
)
if (( ${#SYNC_PATHS[@]} == 0 )); then
  echo "ERROR: the allowlist contains no tracked files in ${BRANCH_SRC}."
  exit 1
fi

restore_branch() {
  status=$?

  if (( status != 0 )) &&
     [[ "$(git branch --show-current)" == "$BRANCH_DST" ]] &&
     [[ -n "${DESTINATION_START:-}" ]] &&
     [[ "$(git rev-parse HEAD)" == "$DESTINATION_START" ]]; then
    git restore --source=HEAD --staged --worktree -- "${SYNC_PATHS[@]}" \
      >/dev/null 2>&1 || true
  fi

  git switch "$CURRENT_BRANCH" >/dev/null 2>&1 || true
  return "$status"
}
trap restore_branch EXIT

git fetch origin "$BRANCH_DST"
git switch "$BRANCH_DST"
git pull --ff-only origin "$BRANCH_DST"
DESTINATION_START="$(git rev-parse HEAD)"

git restore --source="$BRANCH_SRC" --staged --worktree -- "${SYNC_PATHS[@]}"

if git diff --cached --quiet; then
  echo "No selected-file changes to commit."
else
  git commit -m "Copy selected main files into gpuOpt"
  git push origin "$BRANCH_DST"
fi

git switch "$CURRENT_BRANCH"
echo "Switched back to '$CURRENT_BRANCH'"
trap - EXIT