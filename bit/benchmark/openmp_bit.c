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
#include <time.h>   // For timespec structure and clock_gettime

#define MAX_THREADS 1024
#define MIN_SIZE 128

int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p);
int database_match(Bit_T bit, Bit_T *bitsets, int num_of_ref_bits);
int database_match_omp(Bit_T bit, Bit_T *bitsets, int num_of_ref_bits,
                       int threads);
void summarize_results(char *test, int64_t timeElapsed, int num_of_threads,
                       int result, float speedup);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <size> <number of reference bitsets> <max threads>\n",
            argv[0]);
    fprintf(stderr, "Example: %s 1024 1000000 4\n", argv[0]);
    fprintf(stderr,
            "This will create a bitset of size 1024, do an intersection count"
            " against another 1000000 bitsets, and run the test for 1-4 "
            "threads.\n");
    fprintf(stderr, "Please ensure that the size is a positive integer.\n");
    return EXIT_FAILURE;
  }

  int size = atoi(argv[1]);
  int num_of_ref_bits = atoi(argv[2]);
  int max_threads = atoi(argv[3]);

  if (size <= 0 || num_of_ref_bits <= 0 || max_threads <= 0) {
    fprintf(stderr, "Error: size, number of ref bits, and max threads must be "
                    "positive integers.\n");
    return EXIT_FAILURE;
  }

  assert(max_threads <= MAX_THREADS);
  assert(size >= MIN_SIZE);

  // one bitset
  Bit_T bit1 = Bit_new(size);
  Bit_set(bit1, 1, size / 2 + 5);

  printf("Starting OMP and SIMD benchmarks\n");
  // allocate the reference bitsets
  Bit_T *bitsets = malloc(num_of_ref_bits * sizeof(Bit_T));
  for (int i = 0; i < num_of_ref_bits; i++) {
    bitsets[i] = Bit_new(size);
  }
  Bit_set(bitsets[0], size / 2, size / 2 + 5);

  printf("Finished allocating bitsets\n");
  int64_t timings[MAX_THREADS];
  int64_t results[MAX_THREADS];
  int max = 0;
  struct timespec start_time, end_time;
  max = database_match(bit1, bitsets,
                       num_of_ref_bits); // to warm up the processor

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match(bit1, bitsets, num_of_ref_bits);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  int64_t duration = timeDiff(&end_time, &start_time);
  timings[0] = duration;
  results[0] = max;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match(bit1, bitsets, num_of_ref_bits);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[1] = duration;
  results[1] = max;

  clock_gettime(CLOCK_MONOTONIC, &start_time);
  max = database_match(bit1, bitsets, num_of_ref_bits);
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  duration = timeDiff(&end_time, &start_time);
  timings[2] = duration;
  results[2] = max;

  printf("Finished single-threaded match\n");
  for (int i = 1; i <= max_threads; i++) {
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    max = database_match_omp(bit1, bitsets, num_of_ref_bits, i);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    duration = timeDiff(&end_time, &start_time);
    timings[2 + i] = duration;
    results[2 + i] = max;
  }
  // Print results
  printf("Results:\n");
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
}

void summarize_results(char *test, int64_t timeElapsed, int num_of_threads,
                       int result, float speedup) {
  printf("Total time for %35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("Number of threads: %3d \t", num_of_threads);
  printf("Result: %d\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}

int database_match(Bit_T bit, Bit_T *bitsets, int num_of_ref_bits) {
  // Perform the intersection count
  int max = 0;
  for (int i = 0; i < num_of_ref_bits; i++) {
    int current = Bit_inter_count(bit, bitsets[i]);
    if (current > max) {
      max = current;
    }
  }
  return max;
}

int database_match_omp(Bit_T bit, Bit_T *bitsets, int num_of_ref_bits,
                       int threads) {
  // Perform the intersection count in parallel
  int max = 0;
  omp_set_num_threads(threads);
#pragma omp parallel for reduction(max : max) schedule(guided)
  for (int i = 0; i < num_of_ref_bits; i++) {
    int current = Bit_inter_count(bit, bitsets[i]);
    if (current > max) {
      max = current; // Update max only if current is greater
    }
  }
  return max;
}

int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
          timeB_p->tv_nsec);
}
