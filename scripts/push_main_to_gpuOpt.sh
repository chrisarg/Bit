#!/usr/bin/env bash
set -euo pipefail

BRANCH_SRC="main"
BRANCH_DST="gpuOpt"
CURRENT_BRANCH="$(git branch --show-current)"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

cd "$(git rev-parse --show-toplevel)"

# Individual required files. Each path is used exactly as listed, and the
# script aborts before switching branches if any path is missing from main.
EXACT_PATHS=(
  Makefile
  README.md
  include/bit.h
  src/bit_internal.h
  src/bit.c
  src/bit_gpu.c
  src/gpu_layout.h
  src/gpu_layout_fsm.c
  src/gpu_layout_fsm.h
  src/gpu_layout_kernels.c
  src/gpu_layout_kernels.h
  src/gpu_layout_registry.c
  src/gpu_layout_registry.h
  benchmark/benchmark.c
  benchmark/openmp_bit_helpers.c
  benchmark/openmp_bit_helpers.h
  benchmark/openmp_bit.c
  benchmark/openmp_bit_nogpu.c
  benchmark/openmp_bit_container.c
  benchmark/cpu_param_sweep.c
  tests/test_bit.c
  tests/test_offload.c
  scripts/generate_bug_report.sh
)

# Directories whose complete tracked file trees, including subdirectories,
# are synchronized. This is appropriate for header trees that move as a unit.
RECURSIVE_DIRS=(include)

# Directories that contribute only tracked files immediately beneath them;
# nested files are filtered out when SYNC_PATHS is built below.
DIRECT_FILE_DIRS=()

# Directories that should be removed in their entirety during synchronization
CLEANUP_PATHS=()

# gpuOpt owns Makefile_bench.mak, GPU benchmark sources, GPU sweep tooling,
# FAISS comparisons, and benchmark_GPU_params. Those paths are deliberately
# outside this curated shared-file sync and remain untouched in gpuOpt.

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

if (( ! DRY_RUN )) && [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean; commit or stash changes first."
  git status --short
  exit 1
fi

missing=()
for path in "${EXACT_PATHS[@]}"; do
  if ! git cat-file -e "${BRANCH_SRC}:${path}" 2>/dev/null; then
    missing+=("$path")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "ERROR: the following paths are not present in ${BRANCH_SRC}:"
  printf "  %s\n" "${missing[@]}"
  exit 1
fi

SYNC_PATHS=("${EXACT_PATHS[@]}")
for directory in "${RECURSIVE_DIRS[@]}"; do
  mapfile -t paths < <(git ls-tree -r --name-only "$BRANCH_SRC" -- "$directory")
  if (( ${#paths[@]} == 0 )); then
    echo "ERROR: no tracked source files found under ${directory}."
    exit 1
  fi
  SYNC_PATHS+=("${paths[@]}")
done

echo "Shared paths copied from ${BRANCH_SRC} to ${BRANCH_DST}:"
printf "  %s\n" "${SYNC_PATHS[@]}"
echo "gpuOpt-owned paths removed from ${BRANCH_DST} when present:"
printf "  %s\n" "${CLEANUP_PATHS[@]}"

if (( DRY_RUN )); then
  echo "Changes between ${BRANCH_DST} and ${BRANCH_SRC} within the copy scope:"
  git diff --name-status "$BRANCH_DST" "$BRANCH_SRC" -- "${SYNC_PATHS[@]}"
  echo "Cleanup paths currently tracked by ${BRANCH_DST}:"
  for path in "${CLEANUP_PATHS[@]}"; do
    git cat-file -e "${BRANCH_DST}:${path}" 2>/dev/null && printf "  %s\n" "$path"
  done
  exit 0
fi

for directory in "${DIRECT_FILE_DIRS[@]}"; do
  paths=()
  while IFS= read -r path; do
    relative_path="${path#"${directory}/"}"
    if [[ "$relative_path" != */* ]]; then
      paths+=("$path")
    fi
  done < <(git ls-tree -r --name-only "$BRANCH_SRC" -- "$directory")
  if (( ${#paths[@]} == 0 )); then
    echo "ERROR: no tracked source files found directly under ${directory}."
    exit 1
  fi
  SYNC_PATHS+=("${paths[@]}")
done

restore_branch() {
  status=$?

  if (( status != 0 )) &&
     [[ "$(git branch --show-current)" == "$BRANCH_DST" ]] &&
     [[ -n "${DESTINATION_START:-}" ]] &&
     [[ "$(git rev-parse HEAD)" == "$DESTINATION_START" ]]; then
    git restore --source=HEAD --staged --worktree -- "${SYNC_PATHS[@]}" \
      >/dev/null 2>&1 || true
    for path in "${CLEANUP_PATHS[@]}"; do
      if git cat-file -e "HEAD:${path}" 2>/dev/null; then
        git restore --source=HEAD --staged --worktree -- "$path" \
          >/dev/null 2>&1 || true
      fi
    done
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
git rm --ignore-unmatch -- "${CLEANUP_PATHS[@]}"

if git diff --cached --quiet; then
  echo "No selected-file changes to commit."
else
  git commit -m "Copy selected main files into gpuOpt"
  git push origin "$BRANCH_DST"
fi

git switch "$CURRENT_BRANCH"
echo "Switched back to '$CURRENT_BRANCH'"
trap - EXIT