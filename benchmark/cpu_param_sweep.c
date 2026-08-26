/*
 * Universal CPU Parameter Sweep Target
 * Designed for external orchestrators (Cartesian parameter sweeps).
 * Executes exactly ONE thread configuration per run.
 */
#define _POSIX_C_SOURCE 199309L

#include "bit.h"
#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_THREADS 1024
#define MIN_SIZE 128
#define POPCOUNT(x) (int)count_WWG((x))

#define CONTAINER unsigned long long
typedef CONTAINER *bitcontainer;

#ifndef CPU_TILE
#define Q_BLOCK 32
#define R_BLOCK 32
#else
#define Q_BLOCK CPU_TILE
#define R_BLOCK CPU_TILE
#endif

static inline unsigned long long count_WWG(unsigned long long x);
int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p);
int database_match_omp(Bit_T *bit, Bit_T *bitsets, int num_of_bits,
                       int num_of_ref_bits, int threads);
int database_match_container_omp(Bit_DB_T db1, Bit_DB_T db2, int threads);
void summarize_results(char *test, int64_t timeElapsed, int num_of_threads,
                       int result);

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference "
            "bitsets> <exact threads>\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  int size = atoi(argv[1]);
  int num_of_bits = atoi(argv[2]);
  int num_of_ref_bits = atoi(argv[3]);
  int exact_threads = atoi(argv[4]);

  if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 ||
      exact_threads <= 0) {
    fprintf(stderr, "Error: All arguments must be positive integers.\n");
    return EXIT_FAILURE;
  }

  if (exact_threads > MAX_THREADS) {
    fprintf(stderr, "Warning: exact threads capped to %d\n", MAX_THREADS);
    exact_threads = MAX_THREADS;
  }
  if (size < MIN_SIZE) {
    size = MIN_SIZE;
  }

  // Allocate and initialize bitsets
  Bit_T *bits = malloc(num_of_bits * sizeof(Bit_T));
  Bit_T *bitsets = malloc(num_of_ref_bits * sizeof(Bit_T));
  for (int i = 0; i < num_of_bits; i++) {
    bits[i] = Bit_new(size);
    Bit_set(bits[i], size / 2, size - 1);
  }
  for (int i = 0; i < num_of_ref_bits; i++) {
    bitsets[i] = Bit_new(size);
    Bit_set(bitsets[i], size / 2, size - 1);
  }

  Bit_set(bits[0], size / 2 - 1, size / 2 + 5);
  Bit_set(bitsets[0], size / 2, size / 2 + 5);

  Bit_DB_T db1 = BitDB_new(size, num_of_bits);
  Bit_DB_T db2 = BitDB_new(size, num_of_ref_bits);
  for (int i = 0; i < num_of_bits; i++)
    BitDB_put_at(db1, i, bits[i]);
  for (int i = 0; i < num_of_ref_bits; i++)
    BitDB_put_at(db2, i, bitsets[i]);

  struct timespec start_time, end_time;
  int64_t duration;
  int max_res;

  // ---------------------------------------------------------
  // OpenMP Burn-in Phase
  // ---------------------------------------------------------
  // Volatile sinks prevent Dead Code Elimination (DCE) while
  // forcing the OS to wake up and pin the OpenMP thread pool.
  volatile int burn1 = database_match_omp(bits, bitsets, num_of_bits,
                                          num_of_ref_bits, exact_threads);
  volatile int burn2 = database_match_container_omp(db1, db2, exact_threads);
  (void)burn1; // Suppress unused variable warnings
  (void)burn2;

  // ---------------------------------------------------------
  // Timed Execution Phase
  // ---------------------------------------------------------

  // 1. Standard OpenMP Match
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max_res = database_match_omp(bits, bitsets, num_of_bits, num_of_ref_bits,
                               exact_threads);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  summarize_results(" Multi-threaded - OpenMP", duration, exact_threads,
                    max_res);

  // 2. Containerized OpenMP Match
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max_res = database_match_container_omp(db1, db2, exact_threads);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  summarize_results("Container - Multi-threaded - OpenMP", duration,
                    exact_threads, max_res);

  return EXIT_SUCCESS;
}

void summarize_results(char *test, int64_t timeElapsed, int num_of_threads,
                       int result) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("Number of threads: %3d \t", num_of_threads);
  printf("Result: %d\n", result);
}

int database_match_omp(Bit_T *bit, Bit_T *bitsets, int num_of_bits,
                       int num_of_ref_bits, int threads) {
  int max = 0, current = 0;
  size_t workload = (size_t)num_of_bits * (size_t)num_of_ref_bits;
  int *counts = (int *)calloc(workload, sizeof(int));
  if (counts == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for counts array in %s\n",
            __func__);
    exit(EXIT_FAILURE);
  }

  omp_set_num_threads(threads);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < num_of_bits; i += Q_BLOCK) {
    for (int j = 0; j < num_of_ref_bits; j += R_BLOCK) {
      int i_max = (i + Q_BLOCK < num_of_bits) ? i + Q_BLOCK : num_of_bits;
      int j_max =
          (j + R_BLOCK < num_of_ref_bits) ? j + R_BLOCK : num_of_ref_bits;
      for (int qi = i; qi < i_max; qi++) {
        for (int rj = j; rj < j_max; rj++) {
          counts[qi * num_of_ref_bits + rj] =
              Bit_inter_count(bit[qi], bitsets[rj]);
        }
      }
    }
  }

  for (size_t i = 0; i < workload; i++) {
    current = counts[i];
    if (current > max)
      max = current;
  }
  free(counts);
  return max;
}

int database_match_container_omp(Bit_DB_T db1, Bit_DB_T db2, int num_threads) {
  int max = 0, current = 0, *results;
  results = BitDB_inter_count_cpu(
      db1, db2, (SETOP_COUNT_OPTS){.num_cpu_threads = num_threads});
  size_t nelem = (size_t)BitDB_nelem(db2) * BitDB_nelem(db1);
  for (size_t i = 0; i < nelem; i++) {
    current = results[i];
    if (current > max)
      max = current;
  }
  free(results);
  return (int)max;
}

int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
          timeB_p->tv_nsec);
}