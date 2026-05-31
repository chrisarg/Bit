/*

OpenMP GPU-only benchmark
Contains instrumented code to assess PCIe transfer overheads and GPU-only
execution times for containerized intersection counts. The test creates a
database of bitsets, performs intersection counts on the GPU, and reports
timings and speedup factors.

*/
#define _POSIX_C_SOURCE 199309L
#include "openmp_bit_nocpu.h"


#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_GPU_ITERATIONS 1024
#define MIN_SIZE 128

int main(int argc, char *argv[]) {
  if (argc != 5 && argc != 6) {
    fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference "
            "bitsets> <gpu iterations> [<gpu_id>]\n",
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
  for (int i = 0; i < num_of_bits; i++) {
    BitDB_put_at(db1, i, bits[i]);
  }
  for (int i = 0; i < num_of_ref_bits; i++) {
    BitDB_put_at(db2, i, bitsets[i]);
  }

  int64_t timings[MAX_GPU_ITERATIONS + 1];
  int64_t PCIe_timings[MAX_GPU_ITERATIONS + 1];
  int64_t CPU_overhead_timings[MAX_GPU_ITERATIONS + 1];
  int results[MAX_GPU_ITERATIONS + 1];
  GPU_Instrumentation instr;
  // burn-in iteration to mitigate cold-start overheads
  int max = database_match_GPU_instrument(db1, db2,
                               (SETOP_COUNT_OPTS){.device_id = gpu_id,
                                                  .upd_1st_operand = true,
                                                  .upd_2nd_operand = true},
                               &instr);

  puts("Completed burn-in iteration to warm up GPU and PCIe paths");

for (int i = 1; i <= gpu_iterations; i++) {
  max = database_match_GPU_instrument(
      db1, db2,
      (SETOP_COUNT_OPTS){.device_id = gpu_id,
                         .upd_1st_operand = true,
                         .upd_2nd_operand = false},
      &instr);
  timings[i] = timeDiff(&instr.end_time, &instr.start_time);
  PCIe_timings[i] = timeDiff(&instr.end_PCIe_time, &instr.start_PCIe_time);
  CPU_overhead_timings[i] =
      timeDiff(&instr.end_CPU_overhead, &instr.start_CPU_overhead);
  results[i] = max;
}

database_match_GPU_instrument(db1, db2,
                              (SETOP_COUNT_OPTS){.device_id = gpu_id,
                                                 .upd_1st_operand = false,
                                                 .upd_2nd_operand = false,
                                                 .release_1st_operand = true,
                                                 .release_2nd_operand = true,
                                                 .release_counts = true},
                              &instr);

// scaling factors for averaging across iterations
double avg_algorithm_time = 0.0;
for (int i = 1; i <= gpu_iterations; i++) {
  avg_algorithm_time += timings[i];
}
avg_algorithm_time /= gpu_iterations;
double avg_total_operation_time = 0.0;
for (int i = 1; i <= gpu_iterations; i++) {
  avg_total_operation_time += PCIe_timings[i];
}
avg_total_operation_time /= gpu_iterations;

// Calculate per-iteration data movement sizes (in bytes)
// db1 is uploaded every iteration, results are downloaded every iteration
// db2 remains resident on GPU and is NOT counted
double db1_bytes_per_iter = (double)Bit_buffer_size(size) * num_of_bits;
double results_bytes_per_iter =
    (double)num_of_bits * num_of_ref_bits * sizeof(int);
double db2_bytes_resident = (double)Bit_buffer_size(size) * num_of_ref_bits;

// Per-iteration payload: db1 upload + results download (db2 excluded as
// resident)
double payload_per_iteration =
    (db1_bytes_per_iter + results_bytes_per_iter) / (1024 * 1024 * 1024);

puts("GPU Algorithm Timing:");
for (int i = 1; i <= gpu_iterations; i++) {
  summarize_results("Container - GPU - OpenMP", timings[i], i, results[i],
                    (float)timings[1] / timings[i]);
}

puts("GPU Algorithm + PCIe Timings:");
for (int i = 1; i <= gpu_iterations; i++) {
  summarize_results("Container - GPU - OpenMP", PCIe_timings[i], i,
                    results[i], (float)PCIe_timings[1] / PCIe_timings[i]);
}

puts("CPU Overhead Timings:");
for (int i = 1; i <= gpu_iterations; i++) {
  summarize_results("Container - GPU - OpenMP", CPU_overhead_timings[i], i,
                    results[i],
                    (float)CPU_overhead_timings[1] / CPU_overhead_timings[i]);
}

puts("\nPer-Iteration Data Movement Breakdown:");
printf("  db1 (queries) uploaded:   %.6lf GB\n",
       db1_bytes_per_iter / (1024 * 1024 * 1024));
printf("  results downloaded:       %.6lf GB\n",
       results_bytes_per_iter / (1024 * 1024 * 1024));
printf("  db2 (reference, resident): %.6lf GB (NOT transferred)\n",
       db2_bytes_resident / (1024 * 1024 * 1024));
printf("  Total per-iteration:      %.6lf GB\n", payload_per_iteration);

puts("\nEstimated Throughput (iterations 1-N, steady-state):");
printf("GPU compute throughput:      %.3lf GB/s\n",
       payload_per_iteration / ((float)avg_algorithm_time / 1E9));
printf("Total operation throughput:  %.3lf GB/s\n",
       payload_per_iteration / ((float)avg_total_operation_time / 1E9));
puts("\nNote: Total operation throughput includes GPU compute time, data "
     "staging,");
puts("      and PCIe transfers combined, representing user-perceived "
     "performance.");

double openmp_compute_gbps =
    payload_per_iteration / ((float)avg_algorithm_time / 1E9);
printf("OPENMP_SUMMARY,method=OpenMP-Intersection,bitset_bits=%d,nelem=%d,"
       "iterations=%d,avg_ns=%.3lf,gbps=%.6lf,max=%d\n",
       size, num_of_bits, gpu_iterations, avg_algorithm_time,
       openmp_compute_gbps, results[gpu_iterations]);

  return 0;
}
