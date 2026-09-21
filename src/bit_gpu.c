/*
    OpenMP GPU offloading set operations for the Bit library.

    * Author : Christos Argyropoulos
    * Created : July 5th 2026
    * Copyright : (c) 2025 - 2026
    * License : BSD-2
*/


#include <assert.h>  // assert() validation
#include <limits.h>  // INT_MAX
#include <stdbool.h> // bool type
#include <stdint.h>  // uintptr_t and UINT64_C macros
#include <stdlib.h>  // calloc, free
#include <string.h>  // memcpy
#include "bit.h"     // Public API declarations
#include "bit_internal.h"
#include "gpu_layout_registry.h"
#include "omp.h"     // OpenMP parallelization


#ifndef USE_LIBPOPCNT
#define USE_LIBPOPCNT 1
#endif

#if USE_LIBPOPCNT
#include "libpopcnt.h"
#endif

#define T Bit_T
#define T_DB Bit_DB_T



// Make popcount functions available on GPU device targets
#pragma omp declare target(count_WWG)
#pragma omp declare target(tree_adder)


// GPU popcount alias
#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER) ||                    \
    defined(__llvm__) || defined(__GNUC__)

#// Fallback for GCC < 10 which lack __has_builtin but support popcount
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_popcountll) || (defined(__GNUC__) && !defined(__clang__))
#define POPCOUNT_GPU(x) __builtin_popcountll((x))
#else
#define POPCOUNT_GPU(x) count_WWG((x))
#endif

#endif



uint64_t *BitDB_inter_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
#ifndef NOGPU
  BitDB_inter_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_inter_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, uint64_t *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, &, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _AND, opts);
#endif
}

uint64_t *BitDB_union_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
#ifndef NOGPU
  BitDB_union_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_union_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_union_count_store_gpu(T_DB bit, T_DB bits, uint64_t *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, |, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _OR, opts);
#endif
}

uint64_t *BitDB_diff_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
#ifndef NOGPU
  BitDB_diff_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_diff_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_diff_count_store_gpu(T_DB bit, T_DB bits, uint64_t *counts,
                                SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, ^, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _XOR, opts);
#endif
}

uint64_t *BitDB_minus_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
#ifndef NOGPU
  BitDB_minus_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_minus_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_minus_count_store_gpu(T_DB bit, T_DB bits, uint64_t *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, &~, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _AND_NOT, opts);
#endif
}


#ifndef NOGPU
#include <stdio.h> // Ensure stdio is available for printf

void _Bit_gpu_configuration(void) {
    // Print the OpenMP GPU Implementation Strategy
    #if defined(OPENMP_GPU_IMPL_TRANSPOSED_TEAM_PARALLEL_SIMD)
        printf(" %-20s : %s\n", "OpenMP GPU Impl", "TRANSPOSED_TEAM_PARALLEL_SIMD");
    #elif defined(OPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD)
        printf(" %-20s : %s\n", "OpenMP GPU Impl", "TEAM_PARALLEL_SIMD");
    #else
        printf(" %-20s : %s\n", "OpenMP GPU Impl", "UNKNOWN");
    #endif

    // You can add your future formatted GPU configurations here
    // printf(" %-20s : %d\n", "Another GPU Config", SOME_MACRO);
}
#endif