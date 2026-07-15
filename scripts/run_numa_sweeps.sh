#!/usr/bin/env bash
# Run comparable single- and dual-socket CPU tuning sweeps on the dual-socket
# Xeon E5-2697 v4 system. Run this script from any directory.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

command -v numactl >/dev/null || {
  echo "numactl is required but was not found in PATH." >&2
  exit 1
}

# Updated to include the complete telemetry mesh for cross-architecture sweeps
profiles=summary,cache-l1,cache-l2,cache-l3-dram,cache-stalls,buffers-pending,buffers-store,execution-uops,execution-ports,frontend,frequency,vectorization,tlb,uncore-numa,power-rapl
common_env=(
  "ARCH_TAG=x86-64-intel-xeon-e5-2697-v4"
  "LIBPOPCNT_MODES=0,1"
  "REPS=5"
  "PERF_REPS=3"
  "PERF_PROFILES=$profiles"
  "ELEVATE=always"
)

run_sweep() {
  local label=$1
  local numa_policy=$2
  local numa_cmd=$3
  shift 3

  echo "=== Starting $label ==="
  env "${common_env[@]}" \
    "RUN_LABEL=$label" \
    "NUMA_POLICY=$numa_policy" \
    "NUMA_CMD=$numa_cmd" \
    "$@" \
    ./scripts/sweep_cpu_tuning.pl
}

# Use each socket's 18 physical cores with all allocations local to that node.
run_sweep socket0-local 'cpunodebind=0, membind=0' 'numactl --cpunodebind=0 --membind=0' \
  OMP_PLACES=cores OMP_PROC_BIND=close CORES=0-17 THREADS=18 RUN_LABEL=AVX512

run_sweep socket1-local 'cpunodebind=1, membind=1' 'numactl --cpunodebind=1 --membind=1' \
  OMP_PLACES=cores OMP_PROC_BIND=close CORES=18-35 THREADS=18 RUN_LABEL=AVX512

# Re-establish a dual-socket first-touch baseline with OpenMP core binding.
# Empty string passed for numa_cmd as default OS policy applies.
run_sweep dual-first-touch-spread 'default first-touch policy' '' \
  OMP_PLACES=cores OMP_PROC_BIND=spread CORES=0-35 THREADS=36 RUN_LABEL=AVX512

# Spread threads across sockets and interleave newly allocated pages by node.
run_sweep dual-interleave 'interleave=0,1' 'numactl --interleave=0,1' \
  OMP_PLACES=cores OMP_PROC_BIND=spread CORES=0-35 THREADS=36