/*
     Partial reimplementation of the bit.c for benchmarking the GPU
   implementation define bottlenecks, compute vs memory bound using the same
   benchmark as the openmp_bit.c file.
   */
#pragma once
#include "bit.h" // Contains your public API declarations
#include "openmp_bit_nocpu_defs.h"

#include "omp.h"
#include <assert.h>  // For assert() validation
#include <limits.h>  // For INT_MAX
#include <stdbool.h> // For bool type (is_Bit_T_allocated)
#include <stdint.h>  // For uintptr_t and UINT64_C macros
#include <stdio.h>   // For printf (if needed for debugging)
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset
#include <time.h>    // For struct timespec

static uint64_t count_WWG_GPU(unsigned long long x);
static uint64_t tree_adder_GPU(unsigned long long v);
#define STRINGIFY(x) #x // Macro to convert a macro argument to a string

#define BPQW (sizeof(unsigned long long) * 8) // bits per qword
#define BPB (sizeof(unsigned char) * 8)       // bits per byte
#define nqwords(len)                                                           \
  ((((len) + BPQW - 1) & (~(BPQW - 1))) / BPQW)          // ceil(len/QBPW)
#define nbytes(len) ((((len) + 8 - 1) & (~(8 - 1))) / 8) // ceil(len/QBPW)

#define T Bit_T
#define T_DB Bit_DB_T

#ifdef XOROSO
#endif
#define POPCOUNT(x) count_WWG_GPU((x))

/* OpenMP GPU implementation selection:
 *   TEAM_PARALLEL_SIMD
 *   FLAT_COLLAPSE
 */
#if !defined(OPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD) &&                            \
    !defined(OPENMP_GPU_IMPL_FLAT_COLLAPSE)
#define OPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD
#endif

#define SETOP_DB_CHECKS(bit, bits)                                             \
  assert(bit &&bits);                                                          \
  assert(bit->length == bits->length);

#define SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords, \
                       num_targets, n)                                         \
  uint64_t *bit_qwords = (uint64_t *)bit->qwords;                              \
  uint64_t *bits_qwords = (uint64_t *)bits->qwords;                            \
  int bit_size_in_qwords = bit->size_in_qwords;                                \
  int num_targets = bit->nelem;                                                \
  int n = bits->nelem;

#if defined(OPENMP_GPU_IMPL_FLAT_COLLAPSE)
#define setop_count_db_gpu_instrument(bit, bits, counts, op, opts, instr)      \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
  SETOP_INIT_GPU(bit, bits, counts, opts)                                      \
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);                          \
  OMP_GPU_FLAT(opts.device_id)                                                 \
  for (int k = 0; k < num_targets; k++) {                                      \
    for (unsigned int i = 0; i < n; i++) {                                     \
      int total_sum_for_i = 0;                                                 \
      uint64_t shift_k = k * bit_size_in_qwords;                               \
      uint64_t shift_i = i * bit_size_in_qwords;                               \
      for (unsigned int j = 0; j < bit_size_in_qwords; j++) {                  \
        unsigned long long x =                                                 \
            bit_qwords[shift_k + j] op bits_qwords[shift_i + j];               \
        total_sum_for_i += (uint32_t)POPCOUNT_GPU(x);                          \
      }                                                                        \
      counts[k * n + i] = total_sum_for_i;                                     \
    }                                                                          \
  }                                                                            \
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);                            \
  _Pragma(STRINGIFY(omp target exit data map(                                  \
      from : counts [0:num_targets * n]))) if (opts.release_1st_operand) {     \
    SETOP_FINALIZE_GPU(release, bit->qwords, 0,                                \
                       bit_size_in_qwords * num_targets, opts.device_id)       \
  }                                                                            \
  if (opts.release_2nd_operand) {                                              \
    SETOP_FINALIZE_GPU(release, bits->qwords, 0, bit_size_in_qwords * n,       \
                       opts.device_id)                                         \
  }                                                                            \
  if (opts.release_counts) {                                                   \
    SETOP_FINALIZE_GPU(release, counts, 0, num_targets * n, opts.device_id)    \
  }
#endif
#if defined(OPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD)
#define setop_count_db_gpu_instrument(bit, bits, counts, op, opts, instr)      \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
  SETOP_INIT_GPU(bit, bits, counts, opts)                                      \
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);                          \
  OMP_GPU_TEAMS(num_targets, opts.device_id)                                   \
  for (int k = 0; k < num_targets; k++) {                                      \
    const uint64_t *bit_row = bit_qwords + (uint64_t)k * bit_size_in_qwords;   \
    int *counts_row = counts + (size_t)k * n;                                  \
    OMP_GPU_PARALLEL(n) {                                                      \
      OMP_GPU_FOR_NOWAIT                                                       \
      for (unsigned int i = 0; i < n; i++) {                                   \
        const uint64_t *bits_row =                                             \
            bits_qwords + (uint64_t)i * bit_size_in_qwords;                    \
        int total_sum_for_i = 0;                                               \
        OMP_GPU_SIMD_REDUCTION(+, total_sum_for_i)                             \
        for (unsigned int j = 0; j < bit_size_in_qwords; j++) {                \
          unsigned long long x = bit_row[j] op bits_row[j];                    \
          total_sum_for_i += POPCOUNT_GPU(x);                    \
        }                                                                      \
        counts_row[i] = total_sum_for_i;                                       \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);                            \
  _Pragma(STRINGIFY(omp target exit data map(                                  \
      from : counts [0:num_targets * n]))) if (opts.release_1st_operand) {     \
    SETOP_FINALIZE_GPU(release, bit->qwords, 0,                                \
                       bit_size_in_qwords * num_targets, opts.device_id)       \
  }                                                                            \
  if (opts.release_2nd_operand) {                                              \
    SETOP_FINALIZE_GPU(release, bits->qwords, 0, bit_size_in_qwords * n,       \
                       opts.device_id)                                         \
  }                                                                            \
  if (opts.release_counts) {                                                   \
    SETOP_FINALIZE_GPU(release, counts, 0, num_targets * n, opts.device_id)    \
  }
#endif

#define UPDATE_GPU_ARRAY(dir, array, index1, index2, dev_id)                   \
  _Pragma(                                                                     \
      STRINGIFY(omp target update dir(array [index1:index2]) device(dev_id)))

#define TARGET_GPU_ARRAY(point, dir, array, index1, index2, dev_id)            \
  _Pragma(STRINGIFY(omp target point data map(dir : array [index1:index2])     \
                        device(dev_id)))

#define SETOP_INIT_GPU(bit, bits, counts, opts)                                \
  const int _setop_dev_id = (opts).device_id;                                  \
  const int _setop_upd_1st = (opts).upd_1st_operand;                           \
  const int _setop_upd_2nd = (opts).upd_2nd_operand;                           \
  unsigned long long *_setop_bit_qwords = (bit)->qwords;                       \
  unsigned long long *_setop_bits_qwords = (bits)->qwords;                     \
  int *_setop_counts = (counts);                                               \
  const size_t _setop_bit_span = (size_t)(bit)->size_in_qwords * (bit)->nelem; \
  const size_t _setop_bits_span =                                              \
      (size_t)(bits)->size_in_qwords * (bits)->nelem;                          \
  const size_t _setop_counts_span = (size_t)(bit)->nelem * (bits)->nelem;      \
  if (omp_target_is_present(_setop_bit_qwords, _setop_dev_id)) {               \
    if (_setop_upd_1st) {                                                      \
      UPDATE_GPU_ARRAY(to, _setop_bit_qwords, 0, _setop_bit_span,              \
                       _setop_dev_id)                                          \
    }                                                                          \
  } else {                                                                     \
    TARGET_GPU_ARRAY(enter, to, _setop_bit_qwords, 0, _setop_bit_span,         \
                     _setop_dev_id)                                            \
  }                                                                            \
  if (omp_target_is_present(_setop_bits_qwords, _setop_dev_id)) {              \
    if (_setop_upd_2nd) {                                                      \
      UPDATE_GPU_ARRAY(to, _setop_bits_qwords, 0, _setop_bits_span,            \
                       _setop_dev_id)                                          \
    }                                                                          \
  } else {                                                                     \
    TARGET_GPU_ARRAY(enter, to, _setop_bits_qwords, 0, _setop_bits_span,       \
                     _setop_dev_id)                                            \
  }                                                                            \
  if (!omp_target_is_present(_setop_counts, _setop_dev_id)) {                  \
    TARGET_GPU_ARRAY(enter, to, _setop_counts, 0, _setop_counts_span,          \
                     _setop_dev_id)                                            \
  }

// Flat 2-level collapse: exposes num_targets*n work items to the runtime,
// letting it choose team/thread counts for optimal GPU occupancy.
// Replaces the old 3-level: OMP_GPU_TEAMS -> OMP_GPU_PARALLEL ->
// OMP_GPU_FOR_NOWAIT
#define OMP_GPU_FLAT(dev_id)                                                   \
  _Pragma(STRINGIFY(omp target teams distribute parallel for                   \
                        collapse(2) schedule(static) device(dev_id)))

// Legacy macros retained for reference / alternative experimentation
#define OMP_GPU_TEAMS(num_targets, dev_id)                                     \
  _Pragma(STRINGIFY(omp target teams distribute num_teams(num_targets)         \
                        device(dev_id)))

#define OMP_GPU_PARALLEL(n) _Pragma(STRINGIFY(omp parallel num_threads(n)))

#define OMP_GPU_SIMD_REDUCTION(reduction_type, reduction_var)                  \
  _Pragma(STRINGIFY(omp simd reduction(reduction_type : reduction_var)))

#define OMP_GPU_FOR_NOWAIT \
  _Pragma(STRINGIFY(omp for nowait schedule(static)))

#define SETOP_FINALIZE_GPU(action, buffer, index1, index2, dev_id)             \
  if (omp_target_is_present(buffer, dev_id)) {                                 \
    _Pragma(STRINGIFY(omp target exit data map(                                \
        action : buffer [index1:index2]) device(dev_id)))                      \
  }

// device kernels
#define C1_WWG UINT64_C(0X5555555555555555)
#define C2_WWG UINT64_C(0x3333333333333333)
#define C3_WWG UINT64_C(0x0F0F0F0F0F0F0F0F)
#define C4_WWG UINT64_C(0x0101010101010101)
static inline uint64_t count_WWG_GPU(unsigned long long x) {
  x -= (x >> 1) & C1_WWG;
  x = ((x >> 2) & C2_WWG) + (x & C2_WWG);
  x = (x + (x >> 4)) & C3_WWG;
  x *= C4_WWG;

  return (x >> 56);
}

// Tree adder implementation found in https://metacpan.org/pod/Bit::Fast
#define C1_TRADD UINT64_C(0xAAAAAAAAAAAAAAAA)
#define C2_TRADD UINT64_C(0xCCCCCCCCCCCCCCCC)
#define C3_TRADD UINT64_C(0xF0F0F0F0F0F0F0F0)
#define C4_TRADD UINT64_C(0xFF00FF00FF00FF00)
#define C5_TRADD UINT64_C(0x00FF00FF00FF00FF)
#define C6_TRADD UINT64_C(0xFF00FF00FF00FF00)
#define C7_TRADD UINT64_C(0x0000FFFF0000FFFF)
#define C8_TRADD UINT64_C(0xFFFF0000FFFF0000)
#define C9_TRADD UINT64_C(0x00000000FFFFFFFF)
#define C10_TRADD UINT64_C(0xFFFFFFFF00000000)
static inline uint64_t tree_adder_GPU(unsigned long long v) {
  v = (v & C1_WWG) + ((v & C1_TRADD) >> 1);
  v = (v & C2_WWG) + ((v & C2_TRADD) >> 2);
  v = (v & C3_WWG) + ((v & C3_TRADD) >> 4);
  v = (v & C5_TRADD) + ((v & C6_TRADD) >> 8);
  v = (v & C7_TRADD) + ((v & C8_TRADD) >> 16);
  v = (v & C9_TRADD) + ((v & C10_TRADD) >> 32);
  return v;
}
#pragma omp declare target(count_WWG_GPU)
#pragma omp declare target(tree_adder_GPU)

#pragma omp declare target
static inline uint64_t _inter_count_scalar_gpu(const uint64_t *restrict a,
                                               const uint64_t *restrict b,
                                               unsigned int n) {
  uint64_t cnt = 0;
  for (unsigned int i = 0; i < n; i++) {
#ifdef USE_BUILTIN_POPCOUNT
    cnt += (uint64_t)__builtin_popcountll(a[i] & b[i]);
#else
    cnt += (uint64_t)POPCOUNT(a[i] & b[i]);
#endif
  }
  return cnt;
}
#pragma omp end declare target

// POPCOUNT_GPU selects the popcount implementation used in GPU kernels.
// Default: tree_adder (portable software popcount, ~10 ALU ops).
// Define USE_BUILTIN_POPCOUNT to use the compiler intrinsic instead, which
// maps to a single popc instruction on NVIDIA (PTX popc.b64) and AMD GPUs.
#ifdef USE_BUILTIN_POPCOUNT
#define POPCOUNT_GPU(x) __builtin_popcountll(x)
#else
#define POPCOUNT_GPU(x) count_WWG_GPU(x)
#endif

static void _BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                         SETOP_COUNT_OPTS opts,
                                         GPU_Instrumentation *instr) {
  setop_count_db_gpu_instrument(bit, bits, counts, &, opts, instr);
}

/*


#define setop_count_db_gpu_instrument(bit, bits, counts, op, opts, instr)      \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
  SETOP_INIT_GPU(bit, bits, counts, opts)                                      \
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);                          \
  OMP_GPU_TEAMS(num_targets, opts.device_id)                                   \
  for (int k = 0; k < num_targets; k++) {                                      \
    uint64_t shift_k = k * bit_size_in_qwords;                                 \
    OMP_GPU_PARALLEL(n) {                                                      \
      OMP_GPU_FOR_NOWAIT                                                       \
      for (unsigned int i = 0; i < n; i++) {                                   \
        uint64_t shift_i = i * bit_size_in_qwords;                             \
        int total_sum_for_i = 0;                                               \
        OMP_GPU_SIMD_REDUCTION(+, total_sum_for_i)                             \
        for (unsigned int j = 0; j < bit_size_in_qwords; j++) {                \
          unsigned long long x =                                               \
              bit_qwords[shift_k + j] op bits_qwords[shift_i + j];             \
          total_sum_for_i += (uint32_t)POPCOUNT_GPU(x);                        \
        }                                                                      \
        counts[k * n + i] = total_sum_for_i;                                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);                            \
  _Pragma(STRINGIFY(omp target exit data map(                                  \
      from : counts [0:num_targets * n]))) if (opts.release_1st_operand) {     \
    SETOP_FINALIZE_GPU(release, bit->qwords, 0,                                \
                       bit_size_in_qwords * num_targets, opts.device_id)       \
  }                                                                            \
  if (opts.release_2nd_operand) {                                              \
    SETOP_FINALIZE_GPU(release, bits->qwords, 0, bit_size_in_qwords * n,       \
                       opts.device_id)                                         \
  }                                                                            \
  if (opts.release_counts) {                                                   \
    SETOP_FINALIZE_GPU(release, counts, 0, num_targets * n, opts.device_id)    \
  }




     */
