/*
  Contains helper function declarations for openmp_bit_nocpu.
*/
#ifndef OPENMP_BIT_NOCPU_H
#define OPENMP_BIT_NOCPU_H

#include "openmp_bit_nocpu_defs.h"
#include "omp.h"
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bit.h"
#include "openmp_bit_helpers.h"

/*
 * Note: openmp_bit_nocpu_GPU.h should be included by openmp_bit_nocpu.c only.
 */

#define T Bit_T
typedef struct T *T;

#define T_DB Bit_DB_T
typedef struct T_DB *T_DB;

int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p);

void summarize_results(const char *test, int64_t timeElapsed, int iteration,
                       int result, float speedup);

int *BitDB_inter_count_gpu_instrument(T_DB bit, T_DB bits,
                                      SETOP_COUNT_OPTS opts,
                                      GPU_Instrumentation *instr);
void BitDB_inter_count_store_gpu_instrument(T_DB bit, T_DB bits, int *buffer,
                                            SETOP_COUNT_OPTS opts,
                                            GPU_Instrumentation *instr);

int database_match_GPU(Bit_DB_T db1, Bit_DB_T db2, SETOP_COUNT_OPTS opts);
int database_match_GPU_instrument(Bit_DB_T db1, Bit_DB_T db2,
                                  SETOP_COUNT_OPTS opts,
                                  GPU_Instrumentation *instr);

/* CPU helper declarations are provided by openmp_bit_helpers.h */

#endif // OPENMP_BIT_NOCPU_H
