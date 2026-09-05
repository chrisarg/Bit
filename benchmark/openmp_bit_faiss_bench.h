/*
  Shared helpers for the FAISS-comparison benchmarks
  (openmp_bit_cpu_FAISS_comp and openmp_bit_gpu_FAISS_comp).

  This header deliberately depends ONLY on the public Bit library interface
  (bit.h), the private topk selection API (topk.h) and the benchmark helper
  routines (openmp_bit_helpers.h). It must NOT include openmp_bit_nocpu_defs.h
  or any other header that re-declares the library's private concrete types.

  All small utility functions are defined `static inline` so that including
  this header in multiple translation units (and linking them alongside other
  benchmark objects that define same-named extern functions such as timeDiff
  and summarize_results) cannot produce duplicate-symbol link errors.
*/
#ifndef OPENMP_BIT_FAISS_BENCH_H
#define OPENMP_BIT_FAISS_BENCH_H

#define _POSIX_C_SOURCE 199309L

#include "bit.h"
#include "openmp_bit_helpers.h"
#include "topk.h"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Shared, trimmed instrumentation: identical shape for the CPU and the GPU
 * benchmarks. start_time/end_time bracket the setop count kernel (host or
 * device); start_CPU_overhead/end_CPU_overhead bracket the host-side or
 * device-side top-k selection. */
typedef struct {
  struct timespec start_time;
  struct timespec end_time;
  struct timespec start_CPU_overhead;
  struct timespec end_CPU_overhead;
} Bench_Instrumentation;

typedef struct FilteredResults {
  int max;
  int *top_scores;
  int *top_ids;
} FilteredResults;

static inline int parse_positive_size(const char *text, size_t *value) {
  char *end = NULL;
  unsigned long long parsed;

  if (text == NULL || value == NULL || text[0] == '-') {
    return 0;
  }

  errno = 0;
  parsed = strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' || parsed == 0 ||
      parsed > SIZE_MAX) {
    return 0;
  }

  *value = (size_t)parsed;
  return 1;
}

static inline int parse_positive_int(const char *text, int *value) {
  size_t parsed;
  if (!parse_positive_size(text, &parsed) || parsed > INT_MAX) {
    return 0;
  }
  *value = (int)parsed;
  return 1;
}

static inline int64_t timeDiff(struct timespec *timeA_p,
                               struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
          timeB_p->tv_nsec);
}

static inline void summarize_results(const char *test, int64_t timeElapsed,
                                     int iteration, int result,
                                     float speedup) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("CPU iteration: %3d \t", iteration);
  printf("Result: %d\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}

#endif // OPENMP_BIT_FAISS_BENCH_H
