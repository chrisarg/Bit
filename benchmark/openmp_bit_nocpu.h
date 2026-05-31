/*
  Contains helper functions for openmp_bit_nocpu
*/
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
#include "bit.h" // Contains your public API declarations
#include "openmp_bit_nocpu_GPU.h" 
// TRACK_INSERTED: do not include openmp_bit_nocpu_GPU.h here; include it in openmp_bit_nocpu.c only.

#define T Bit_T
typedef struct T *T;

#define T_DB Bit_DB_T
typedef struct T_DB *T_DB;

// definitions of functions
int64_t timeDiff(struct timespec* timeA_p, struct timespec* timeB_p);


int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
          timeB_p->tv_nsec);
}

void summarize_results(char *test, int64_t timeElapsed, int iteration,
                       int result, float speedup) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("GPU iteration: %3d \t", iteration);
  printf("Result: %d\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}


int *BitDB_inter_count_gpu_instrument(T_DB bit, T_DB bits,
                                      SETOP_COUNT_OPTS opts,
                                      GPU_Instrumentation *instr);
void BitDB_inter_count_store_gpu_instrument(T_DB bit, T_DB bits, int *buffer,
                                            SETOP_COUNT_OPTS opts,
                                            GPU_Instrumentation *instr);
void _BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, int *buffer,
                                  SETOP_COUNT_OPTS opts,
                                  GPU_Instrumentation *instr);

extern int *BitDB_inter_count_gpu_instrument(T_DB bit, T_DB bits,
                                             SETOP_COUNT_OPTS opts,
                                             GPU_Instrumentation *instr) {

  size_t nelem = (size_t)BitDB_nelem(bit) * BitDB_nelem(bits);
  int *counts = (int *)calloc(nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_inter_count_store_gpu_instrument(bit, bits, counts, opts, instr);
  return counts;
}

void BitDB_inter_count_store_gpu_instrument(T_DB bit, T_DB bits, int *counts,
                                            SETOP_COUNT_OPTS opts,
                                            GPU_Instrumentation *instr) {
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);
  _BitDB_inter_count_store_gpu(bit, bits, counts, opts, instr);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);
}


int database_match_GPU(Bit_DB_T db1, Bit_DB_T db2, SETOP_COUNT_OPTS opts) {
  int max = 0, current = 0, *results;
  results = BitDB_inter_count_gpu(db1, db2, opts);
  size_t nelem = (size_t)BitDB_nelem(db2) * BitDB_nelem(db1);
  for (size_t i = 0; i < nelem; i++) {
    current = results[i];
    if (current > max) {
      max = current;
    }
  }
  return max;
}

int database_match_GPU_instrument(Bit_DB_T db1, Bit_DB_T db2,
                                  SETOP_COUNT_OPTS opts,
                                  GPU_Instrumentation *instr) {
  int max = 0, current = 0, *results;
  clock_gettime(CLOCK_MONOTONIC, &instr->start_PCIe_time);
  results = BitDB_inter_count_gpu_instrument(db1, db2, opts, instr);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_PCIe_time);
  clock_gettime(CLOCK_MONOTONIC, &instr->start_CPU_overhead);
  size_t nelem = (size_t)BitDB_nelem(db2) * BitDB_nelem(db1);
  for (size_t i = 0; i < nelem; i++) {
    current = results[i];
    if (current > max) {
      max = current;
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &instr->end_CPU_overhead);
  return max;
}

/*
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
     */