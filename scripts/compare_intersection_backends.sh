#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 <bitset_bits> <num_bits> <num_ref_bits> <gpu_id> <iters> <openmp_avg_ns> <openmp_gbps> <openmp_max>" >&2
  exit 2
fi

BITSET_BITS="$1"
NUM_BITS="$2"
NUM_REF_BITS="$3"
GPU_ID="$4"
ITERS="$5"
OPENMP_AVG_NS="$6"
OPENMP_GBPS="$7"
OPENMP_MAX="$8"

OPENMP_NS_PER_PAIR="$(awk -v ns="$OPENMP_AVG_NS" -v a="$NUM_BITS" -v b="$NUM_REF_BITS" 'BEGIN { if (a*b == 0) { print "nan" } else { printf "%.6f", ns/(a*b) } }')"

NELEM="$NUM_BITS"
if (( NUM_REF_BITS < NUM_BITS )); then
  NELEM="$NUM_REF_BITS"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_RADEON_DIR="/home/chrisarg/software-dev/test_radeon"
CUDA_MAIN_DIR="$TEST_RADEON_DIR/test_nvidia"
HIP_MAIN_DIR="$TEST_RADEON_DIR/test_amd"
CUDA_BIN="$ROOT_DIR/build/intersection_cuda_bench"
HIP_BIN="$ROOT_DIR/build/intersection_hip_bench"
BACKEND_OVERRIDE="${BIT_COMPARE_BACKEND:-AUTO}"

if [[ ! -d "$TEST_RADEON_DIR" ]]; then
  echo "COMPARE_WARNING: test_radeon repo not found at $TEST_RADEON_DIR" >&2
  exit 1
fi

BACKEND=""
case "$BACKEND_OVERRIDE" in
  AUTO|auto)
    if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
      GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
      if [[ "$GPU_COUNT" != "0" ]]; then
        BACKEND="CUDA"
      fi
    fi

    if [[ -z "$BACKEND" ]] && command -v hipcc >/dev/null 2>&1; then
      BACKEND="HIP"
    fi
    ;;
  CUDA|cuda)
    BACKEND="CUDA"
    ;;
  HIP|hip)
    BACKEND="HIP"
    ;;
  *)
    echo "COMPARE_WARNING: invalid BIT_COMPARE_BACKEND='$BACKEND_OVERRIDE' (use AUTO, CUDA, or HIP)" >&2
    exit 1
    ;;
esac

if [[ -z "$BACKEND" ]]; then
  echo "COMPARE_WARNING: neither CUDA nor HIP backend is available" >&2
  exit 1
fi

echo "COMPARE_BACKEND_SELECTED=$BACKEND"

device_arg="$GPU_ID"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  device_arg="0"
fi

if [[ "$BACKEND" == "CUDA" ]]; then
  (
    cd "$CUDA_MAIN_DIR"
    nvcc -O3 -std=c++17 main.cu -o "$CUDA_BIN"
  )
  BACKEND_CMD=("$CUDA_BIN" 1000 --device "$device_arg" --bitset_size "$BITSET_BITS" --nelem "$NELEM")
else
  (
    cd "$HIP_MAIN_DIR"
    hipcc -O3 -std=c++17 main.cpp -o "$HIP_BIN"
  )
  BACKEND_CMD=("$HIP_BIN" 1000 --device "$device_arg" --bitset_size "$BITSET_BITS" --nelem "$NELEM")
fi

BACKEND_OUTPUT="$("${BACKEND_CMD[@]}")"

WWG_LINE="$(printf '%s\n' "$BACKEND_OUTPUT" | grep 'POP_SUMMARY,width=UINT64,method=WWG' | tail -n 1 || true)"
BUILTIN_LINE="$(printf '%s\n' "$BACKEND_OUTPUT" | grep 'POP_SUMMARY,width=UINT64,method=builtin' | tail -n 1 || true)"

if [[ -z "$WWG_LINE" || -z "$BUILTIN_LINE" ]]; then
  echo "COMPARE_WARNING: failed to parse backend POP_SUMMARY lines" >&2
  printf '%s\n' "$BACKEND_OUTPUT" | tail -n 40 >&2
  exit 1
fi

extract_field() {
  local line="$1"
  local key="$2"
  printf '%s\n' "$line" | awk -F',' -v k="$key" '
  {
    for (i = 1; i <= NF; i++) {
      split($i, a, "=")
      if (a[1] == k) {
        print a[2]
        exit
      }
    }
  }'
}

WWG_GBPS="$(extract_field "$WWG_LINE" gbps)"
WWG_NS="$(extract_field "$WWG_LINE" ns_per_elem)"
WWG_DISAGREE="$(extract_field "$WWG_LINE" disagree)"

BUILTIN_GBPS="$(extract_field "$BUILTIN_LINE" gbps)"
BUILTIN_NS="$(extract_field "$BUILTIN_LINE" ns_per_elem)"
BUILTIN_DISAGREE="$(extract_field "$BUILTIN_LINE" disagree)"

printf '\nBackend comparison (intersection, UINT64 path):\n'
printf '| %-18s | %-12s | %-12s | %-10s | %-8s |\n' "Method" "GB/s" "ns/elem" "disagree" "notes"
printf '|---|---:|---:|---:|---|\n'
printf '| %-18s | %-12.6f | %-12s | %-10s | %-8s |\n' "OpenMP" "$OPENMP_GBPS" "$OPENMP_NS_PER_PAIR" "n/a" "max=$OPENMP_MAX"
printf '| %-18s | %-12s | %-12s | %-10s | %-8s |\n' "$BACKEND-WWG" "$WWG_GBPS" "$WWG_NS" "$WWG_DISAGREE" "UINT64"
printf '| %-18s | %-12s | %-12s | %-10s | %-8s |\n' "$BACKEND-builtin" "$BUILTIN_GBPS" "$BUILTIN_NS" "$BUILTIN_DISAGREE" "UINT64"
printf 'OPENMP_NORMALIZATION: avg_ns_per_iteration=%.6f over %s x %s pairs -> ns/pair=%s\n' "$OPENMP_AVG_NS" "$NUM_BITS" "$NUM_REF_BITS" "$OPENMP_NS_PER_PAIR"

if [[ "$WWG_DISAGREE" != "0" || "$BUILTIN_DISAGREE" != "0" ]]; then
  echo "COMPARE_WARNING: backend disagreement detected (non-zero disagree count)" >&2
  exit 1
fi

echo "COMPARE_STATUS=OK"
