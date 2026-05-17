/*

OpenMP GPU-only benchmark

*/
#define _POSIX_C_SOURCE 199309L

#include "bit.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_GPU_ITERATIONS 1024
#define MIN_SIZE 128

int64_t timeDiff(struct timespec* timeA_p, struct timespec* timeB_p);
int database_match_GPU(Bit_DB_T db1, Bit_DB_T db2, SETOP_COUNT_OPTS opts);
void summarize_results(char* test, int64_t timeElapsed, int iteration,
  int result, float speedup);

int main(int argc, char* argv []) {
  if (argc != 5 && argc != 6) {
    fprintf(stderr,
      "Usage: %s <size> <number of bitsets> <number of reference bitsets> <gpu iterations> [<gpu_id>]\n",
      argv[0]);
    fprintf(stderr, "Example: %s 1024 1000 1000000 10 0\n", argv[0]);
    fprintf(stderr,
      "This will create 1000 bitsets of size 1024 and run 10 GPU-only "
      "containerized intersection-count iterations on GPU 0.\n");
    return EXIT_FAILURE;
  }

  int size = atoi(argv[1]);
  int num_of_bits = atoi(argv[2]);
  int num_of_ref_bits = atoi(argv[3]);
  int gpu_iterations = atoi(argv[4]);
  int gpu_id = 0;
  if (argc == 6) {
    gpu_id = atoi(argv[5]);
  }

  if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 ||
    gpu_iterations <= 0) {
    fprintf(stderr, "Error: size, number of bits, number of ref bits, and GPU "
      "iterations must be positive integers.\n");
    return EXIT_FAILURE;
  }

  if (gpu_iterations > MAX_GPU_ITERATIONS) {
    fprintf(stderr, "Warning: gpu iterations capped to %d\n",
      MAX_GPU_ITERATIONS);
    gpu_iterations = MAX_GPU_ITERATIONS;
  }
  if (size < MIN_SIZE) {
    fprintf(stderr, "Warning: size increased to %d\n", MIN_SIZE);
    size = MIN_SIZE;
  }

#ifndef NDEBUG
  printf("Debug mode is enabled.\n");
#else
  printf("Debug mode is disabled.\n");
#endif

  printf("Starting GPU-only benchmark\n");

  Bit_T* bits = malloc(num_of_bits * sizeof(Bit_T));
  Bit_T* bitsets = malloc(num_of_ref_bits * sizeof(Bit_T));
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
  for (int i = 0; i < num_of_bits; i++) {
    BitDB_put_at(db1, i, bits[i]);
  }
  for (int i = 0; i < num_of_ref_bits; i++) {
    BitDB_put_at(db2, i, bitsets[i]);
  }

  int64_t timings[MAX_GPU_ITERATIONS + 1];
  int results[MAX_GPU_ITERATIONS + 1];
  struct timespec start_time, end_time;

  int max = database_match_GPU(db1, db2,
    (SETOP_COUNT_OPTS) {
    .device_id = gpu_id,
      .upd_1st_operand = true,
      .upd_2nd_operand = true
  });

  for (int i = 1; i <= gpu_iterations; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    max = database_match_GPU(db1, db2,
      (SETOP_COUNT_OPTS) {
      .device_id = gpu_id,
        .upd_1st_operand = false,
        .upd_2nd_operand = false
    });
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    timings[i] = timeDiff(&end_time, &start_time);
    results[i] = max;
  }

  database_match_GPU(db1, db2,
    (SETOP_COUNT_OPTS) {
    .device_id = gpu_id,
      .upd_1st_operand = false,
      .upd_2nd_operand = false,
      .release_1st_operand = true,
      .release_2nd_operand = true,
      .release_counts = true
  });

  puts("Results:");
  for (int i = 1; i <= gpu_iterations; i++) {
    summarize_results("Container - GPU - OpenMP", timings[i], i, results[i],
      (float)timings[1] / timings[i]);
  }

  return 0;
}

void summarize_results(char* test, int64_t timeElapsed, int iteration,
  int result, float speedup) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("GPU iteration: %3d \t", iteration);
  printf("Result: %d\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}

int database_match_GPU(Bit_DB_T db1, Bit_DB_T db2, SETOP_COUNT_OPTS opts) {
  int max = 0, current = 0, * results;
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

int64_t timeDiff(struct timespec* timeA_p, struct timespec* timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
    timeB_p->tv_nsec);
}
