/*
    OpenMP GPU offloading set operations for the Bit library.

    * Author : Christos Argyropoulos
    * Created : July 5th 2026
    * Copyright : (c) 2025 - 2026
    * License : BSD-2
*/

#include "bit.h"               // Public API declarations
#include "omp.h"               // OpenMP parallelization
#include <assert.h>            // assert() validation
#include <limits.h>            // INT_MAX
#include <stdbool.h>           // bool type
#include <stdint.h>            // uintptr_t and UINT64_C macros
#include <stdlib.h>            // calloc, free
#include <string.h>            // memcpy

#ifndef USE_LIBPOPCNT
#define USE_LIBPOPCNT 1
#endif

#if USE_LIBPOPCNT
#include "libpopcnt.h"
#endif

#define T Bit_T
#define T_DB Bit_DB_T

#include "bit_internal.h"

// Make popcount functions available on GPU device targets
#pragma omp declare target(count_WWG)
#pragma omp declare target(tree_adder)
// GPU popcount alias
#define POPCOUNT_GPU count_WWG

/* --- 11d. GPU set operations (allocate and return counts buffer) --- */

int *BitDB_inter_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_inter_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_inter_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, &, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _AND, opts);
#endif
}

int *BitDB_union_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_union_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_union_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_union_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, |, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _OR, opts);
#endif
}

int *BitDB_diff_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_diff_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_diff_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_diff_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, ^, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _XOR, opts);
#endif
}

int *BitDB_minus_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_minus_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_minus_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_minus_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, &~, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _AND_NOT, opts);
#endif
}
