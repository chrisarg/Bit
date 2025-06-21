/*

OpenMP enabled benchmarks

*/
#define _POSIX_C_SOURCE 199309L

#include "bit.h"
#include <assert.h> // For assert() validation
#include <omp.h>    // For OpenMP parallelization
#include <stdint.h> // For int64_t type
#include <stdio.h>  // For printf/fprintf functions
#include <stdlib.h> // For malloc, atoi, and EXIT_FAILURE
#include <string.h> // For memcpy
#include <time.h>   // For timespec structure and clock_gettime
#define MAX_THREADS 1024
#define MIN_SIZE 128
#define POPCOUNT(x) (int)count_WWG((x))

#define CONTAINER unsigned long long
typedef CONTAINER *bitcontainer;
static inline unsigned long long count_WWG(unsigned long long x);
int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p);
int database_match(Bit_T *bit, Bit_T *bitsets, int num_of_bits,
                   int num_of_ref_bits);
int database_match_omp(Bit_T *bit, Bit_T *bitsets, int num_of_bits,
                       int num_of_ref_bits, int threads);
int database_match_container_omp(Bit_T_DB db1, Bit_T_DB db2, int threads);
int database_match_GPU(Bit_T_DB db1, Bit_T_DB db2, SETOP_COUNT_OPTS opts);
void summarize_results(char *test, int64_t timeElapsed, int num_of_threads,
                       int result, float speedup);

int main(int argc, char *argv[]) {
  if (argc != 5) {
    fprintf(stderr,
            "Usage: %s <size> <number of reference bitsets> <max threads>\n",
            argv[0]);
    fprintf(stderr, "Example: %s 1024 1000 1000000 4\n", argv[0]);
    fprintf(stderr,
            "This will create 100 bitsets size 1024, do an intersection count"
            " against another 1000000 bitsets, and run the test for 1-4 "
            "threads.\n");
    fprintf(stderr, "Please ensure that the size is a positive integer.\n");
    return EXIT_FAILURE;
  }

  int size = atoi(argv[1]);
  int num_of_bits = atoi(argv[2]);
  int num_of_ref_bits = atoi(argv[3]);
  int max_threads = atoi(argv[4]);

  if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 ||
      max_threads <= 0) {
    fprintf(stderr, "Error: size, number of bits, number of ref bits, and max "
                    "threads must be "
                    "positive integers.\n");
    return EXIT_FAILURE;
  }

  assert(max_threads <= MAX_THREADS);
  assert(size >= MIN_SIZE);

  printf("Starting OMP and SIMD benchmarks\n");
  // allocate the bitsets
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
  printf("Finished allocating bitsets \n");

  Bit_T_DB db1 = BitDB_new(size, num_of_bits);
  Bit_T_DB db2 = BitDB_new(size, num_of_ref_bits);
  for (int i = 0; i < num_of_bits; i++)
    BitDB_put_at(db1, i, bits[i]);

  for (int i = 0; i < num_of_ref_bits; i++)
    BitDB_put_at(db2, i, bitsets[i]);

  printf("Finished allocating BitDB\n");
  int64_t timings[2 * MAX_THREADS + 6];
  int64_t results[2 * MAX_THREADS + 6];
  int max = 0;
  struct timespec start_time, end_time;

  max = database_match(bits, bitsets, num_of_bits,
                       num_of_ref_bits); // to warm up the processor

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match(bits, bitsets, num_of_bits, num_of_ref_bits);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  int64_t duration = timeDiff(&end_time, &start_time);
  timings[0] = duration;
  results[0] = max;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match(bits, bitsets, num_of_bits, num_of_ref_bits);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[1] = duration;
  results[1] = max;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match(bits, bitsets, num_of_bits, num_of_ref_bits);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[2] = duration;
  results[2] = max;

  puts("Finished single-threaded match");
  for (int i = 1; i <= max_threads; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    max = database_match_omp(bits, bitsets, num_of_bits, num_of_ref_bits, i);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    duration = timeDiff(&end_time, &start_time);
    timings[2 + i] = duration;
    results[2 + i] = max;
  }
  puts("Finished multi-threaded match with OpenMP");
  for (int i = max_threads + 1; i <= 2 * max_threads; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    max = database_match_container_omp(db1, db2, i - max_threads);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    duration = timeDiff(&end_time, &start_time);
    timings[2 + i] = duration;
    results[2 + i] = max;
  }
  puts("Finished multithreaded match with Bit");
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match_GPU(db1, db2,
                           (SETOP_COUNT_OPTS){.device_id = 0,
                                              .upd_1st_operand = false,
                                              .upd_2nd_operand = false});
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[2 + 2 * max_threads + 1] = duration;
  results[2 + 2 * max_threads + 1] = max;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match_GPU(db1, db2,
                           (SETOP_COUNT_OPTS){.device_id = 0,
                                              .upd_1st_operand = true,
                                              .upd_2nd_operand = false});
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[2 + 2 * max_threads + 2] = duration;
  results[2 + 2 * max_threads + 2] = max;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match_GPU(db1, db2,
                           (SETOP_COUNT_OPTS){.device_id = 0,
                                              .upd_1st_operand = true,
                                              .upd_2nd_operand = false,
                                              .release_1st_operand = true,
                                              .release_2nd_operand = true,
                                              .release_counts = true});
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[2 + 2 * max_threads + 3] = duration;
  results[2 + 2 * max_threads + 3] = max;

  // Print results
  puts("Results:");
  summarize_results("Single-threaded - Serial - Rep1", timings[0], 1,
                    results[0], 1.0f);
  summarize_results("Single-threaded - Serial - Rep2", timings[1], 1,
                    results[1], (float)timings[0] / timings[1]);
  summarize_results("Single-threaded - Serial - Rep3", timings[2], 1,
                    results[2], (float)timings[0] / timings[2]);
  for (int i = 1; i <= max_threads; i++) {
    summarize_results(" Multi-threaded - OpenMP", timings[2 + i], i,
                      results[2 + i], (float)timings[0] / timings[2 + i]);
  }
  for (int i = max_threads + 1; i <= 2 * max_threads; i++) {
    summarize_results("Container - Multi-threaded - OpenMP", timings[2 + i],
                      i - max_threads, results[2 + i],
                      (float)timings[0] / timings[2 + i]);
  }
  for (int i = 2 * max_threads + 1; i <= 2 * max_threads + 3; i++) {
    summarize_results("Container - GPU - OpenMP", timings[2 + i], -1,
                      results[2 + i], (float)timings[0] / timings[2 + i]);
  }
}

void summarize_results(char *test, int64_t timeElapsed, int num_of_threads,
                       int result, float speedup) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  num_of_threads > 0 ? printf("Number of threads: %3d \t", num_of_threads)
                     : printf("Number of threads: GPU \t");
  printf("Result: %d\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}

int database_match(Bit_T *bit, Bit_T *bitsets, int num_of_bits,
                   int num_of_ref_bits) {
  // Perform the intersection count
  int max = 0, current = 0;
  size_t workload = (size_t)num_of_bits * (size_t)num_of_ref_bits;
  int *counts = (int *)calloc(workload, sizeof(int));
  assert(counts != NULL);
  for (int i = 0; i < num_of_bits; i++) {
    for (int j = 0; j < num_of_ref_bits; j++) {
      counts[i * num_of_ref_bits + j] = Bit_inter_count(bit[i], bitsets[j]);
    }
  }
  for (size_t i = 0; i < workload; i++) {
    current = counts[i];
    if (current > max) {
      max = current;
    }
  }
  free(counts);
  return max;
}

int database_match_omp(Bit_T *bit, Bit_T *bitsets, int num_of_bits,
                       int num_of_ref_bits, int threads) {
  // Perform the intersection count in parallel
  int max = 0, current = 0;
  size_t workload = (size_t)num_of_bits * (size_t)num_of_ref_bits;
  int *counts = (int *)calloc(workload, sizeof(int));
  assert(counts != NULL);
  omp_set_num_threads(threads);
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_of_bits; i++) {
    for (int j = 0; j < num_of_ref_bits; j++) {
      counts[i * num_of_ref_bits + j] = Bit_inter_count(bit[i], bitsets[j]);
    }
  }

  for (size_t i = 0; i < workload; i++) {
    current = counts[i];
    if (current > max) {
      max = current;
    }
  }
  free(counts);
  return max;
}

int database_match_container_omp(Bit_T_DB db1, Bit_T_DB db2, int num_threads) {
  int max = 0, current = 0, *results;
  results = BitDB_inter_count_cpu(
      db1, db2, (SETOP_COUNT_OPTS){.num_cpu_threads = num_threads});
  size_t nelem = (size_t)BitDB_nelem(db2) * BitDB_nelem(db1);
  for (size_t i = 0; i < nelem; i++) {
    current = results[i];
    if (current > max) {
      max = current;
    }
  }
  free(results);
  return (int)max;
}

int database_match_GPU(Bit_T_DB db1, Bit_T_DB db2, SETOP_COUNT_OPTS opts) {
  int max = 0, current = 0, *results;
  results = BitDB_inter_count_gpu(db1, db2, opts);
  size_t nelem = (size_t)BitDB_nelem(db2) * BitDB_nelem(db1);
  for (size_t i = 0; i < nelem; i++) {
    current = results[i];
    if (current > max) {
      max = current;
    }
  }
  // free(results);
  return (int)max;
}
int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
          timeB_p->tv_nsec);
}

static inline unsigned long long count_WWG(unsigned long long x) {
#define C1_WWG UINT64_C(0X5555555555555555)
#define C2_WWG UINT64_C(0x3333333333333333)
#define C3_WWG UINT64_C(0x0F0F0F0F0F0F0F0F)
#define C4_WWG UINT64_C(0x0101010101010101)

  x -= (x >> 1) & C1_WWG;
  x = ((x >> 2) & C2_WWG) + (x & C2_WWG);
  x = (x + (x >> 4)) & C3_WWG;
  x *= C4_WWG;

  return (x >> 56);
}
