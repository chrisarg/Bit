/*

OpenMP GPU-only benchmark
Contains instrumented code to assess PCIe transfer overheads and GPU-only
execution times for containerized intersection counts. The test creates a
database of bitsets, performs intersection counts on the GPU, and reports
timings and speedup factors.

*/
#define _POSIX_C_SOURCE 199309L

#include "openmp_bit_nocpu.h"
#include "openmp_bit_helpers.h"

#include "openmp_bit_nocpu_GPU.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_GPU_ITERATIONS 1024
#define MIN_SIZE 128

int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 +
          timeA_p->tv_nsec - timeB_p->tv_nsec);
}

void summarize_results(const char *test, int64_t timeElapsed, int iteration,
                       int result, float speedup) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("GPU iteration: %3d \t", iteration);
  printf("Result: %d\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}

int *BitDB_inter_count_gpu_instrument(T_DB bit, T_DB bits,
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
  free(results);
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
  free(results);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_CPU_overhead);
  return max;
}

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

#ifdef OPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD
  printf("Using OpenMP GPU implementation: TEAM_PARALLEL_SIMD\n");
#endif
#ifdef OPENMP_GPU_IMPL_TRANSPOSED_TEAM_PARALLEL_SIMD
  printf("Using OpenMP GPU implementation: TRANSPOSED_TEAM_PARALLEL_SIMD\n");
#endif
#ifdef USE_BUILTIN_POPCOUNT
  printf("Using OpenMP GPU popcount: builtin\n");
#else
  printf("Using OpenMP GPU popcount: WWG\n");
#endif
  printf("Starting GPU-only benchmark\n");

  const size_t words_per_bitset = (size + 63) / 64;
  size_t queries_words = words_per_bitset * (size_t)num_of_bits;
  size_t refs_words = words_per_bitset * (size_t)num_of_ref_bits;
  size_t results_words = (size_t)num_of_bits * (size_t)num_of_ref_bits;

  uint64_t *h_queries = malloc(queries_words * sizeof(uint64_t));
  uint64_t *h_refs = malloc(refs_words * sizeof(uint64_t));
  uint32_t *cpu_results = malloc(results_words * sizeof(uint32_t));
  assert(h_queries && h_refs && cpu_results);

  puts("Generating random bitsets...");
  uint64_t seed = 0xDEADBEEF;
  for (size_t i = 0; i < queries_words; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    h_queries[i] = seed;
  }
  for (size_t i = 0; i < refs_words; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    h_refs[i] = seed;
  }

  puts("Computing CPU reference results...");
  compute_cpu_popcount_reference(h_queries, h_refs, size, num_of_bits,
                                 num_of_ref_bits, cpu_results);

  puts("Loading random bitvectors into Bitsets...");
  Bit_T *bits = malloc(num_of_bits * sizeof(Bit_T));
  Bit_T *bitsets = malloc(num_of_ref_bits * sizeof(Bit_T));
  assert(bits && bitsets);
  for (int i = 0; i < num_of_bits; i++) {
    bits[i] = Bit_load(size, h_queries + (size_t)i * words_per_bitset);
  }
  for (int i = 0; i < num_of_ref_bits; i++) {
    bitsets[i] = Bit_load(size, h_refs + (size_t)i * words_per_bitset);
  }


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
  int64_t GPU_transpose_timings[MAX_GPU_ITERATIONS + 1];
  int results[MAX_GPU_ITERATIONS + 1];
  GPU_Instrumentation instr;
  // burn-in iteration to mitigate cold-start overheads
  int max =
      database_match_GPU_instrument(db1, db2,
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
    GPU_transpose_timings[i] =
        timeDiff(&instr.end_GPU_transpose_time, &instr.start_GPU_transpose_time);
    results[i] = max;
  }

  size_t agreements = 0;
  size_t disagreements = 0;
  uint32_t verify_max = 0;
  int *gpu_counts = BitDB_inter_count_gpu_instrument(
      db1, db2,
      (SETOP_COUNT_OPTS){.device_id = gpu_id,
                         .upd_1st_operand = false,
                         .upd_2nd_operand = false},
      &instr);
  compare_gpu_to_cpu_results(gpu_counts, cpu_results, num_of_bits,
                             num_of_ref_bits, &agreements, &disagreements,
                             &verify_max);
  free(gpu_counts);

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
  double stddev_algorithm_time = 0.0;
  compute_int64_mean_stddev(&timings[1], gpu_iterations,
                            &avg_algorithm_time, &stddev_algorithm_time);
  double avg_total_operation_time = 0.0;
  double stddev_total_operation_time = 0.0;
  compute_int64_mean_stddev(&PCIe_timings[1], gpu_iterations,
                            &avg_total_operation_time,
                            &stddev_total_operation_time);

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

  puts("GPU Transpose Timings:");
  for (int i = 1; i <= gpu_iterations; i++) {
    summarize_results("Container - GPU - OpenMP", GPU_transpose_timings[i], i,
                      results[i], (float)GPU_transpose_timings[1] /
                                      GPU_transpose_timings[i]);
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

  printf("  agreements: %zu\n", agreements);
  printf("  disagreements: %zu\n", disagreements);
  if (disagreements > 0) {
    printf("  WARNING: GPU results disagree with CPU reference\n");
  }

  double compute_gbps[MAX_GPU_ITERATIONS];
  double total_gbps[MAX_GPU_ITERATIONS];
  for (int i = 1; i <= gpu_iterations; i++) {
    compute_gbps[i - 1] = payload_per_iteration / ((double)timings[i] / 1E9);
    total_gbps[i - 1] = payload_per_iteration / ((double)PCIe_timings[i] / 1E9);
  }
  double avg_compute_gbps = 0.0;
  double stddev_compute_gbps = 0.0;
  compute_mean_stddev(compute_gbps, gpu_iterations,
                      &avg_compute_gbps, &stddev_compute_gbps);
  double avg_total_gbps = 0.0;
  double stddev_total_gbps = 0.0;
  compute_mean_stddev(total_gbps, gpu_iterations,
                      &avg_total_gbps, &stddev_total_gbps);

    puts("\nEstimated Throughput (iterations 1-N, steady-state):");
    printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n",
      avg_algorithm_time, stddev_algorithm_time);
    printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
      avg_compute_gbps, stddev_compute_gbps);
    printf("Total operation time: mean=%.3f ns, stddev=%.3f ns\n",
      avg_total_operation_time, stddev_total_operation_time);
    printf("Total operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
      avg_total_gbps, stddev_total_gbps);
  puts("\nNote: Total operation throughput includes GPU compute time, data "
       "staging,");
  puts("      and PCIe transfers combined, representing user-perceived "
       "performance.");

    printf("OPENMP_SUMMARY,method=OpenMP-Intersection-%s,bitset_bits=%d,nelem=%d,"
      "iterations=%d,avg_ns=%.3lf,stddev_ns=%.3lf,gbps=%.6lf,gbps_stddev=%.6lf,max=%d\n",
#ifdef USE_BUILTIN_POPCOUNT
         "builtin",
#else
         "WWG",
#endif
      size, num_of_bits, gpu_iterations, avg_algorithm_time,
      stddev_algorithm_time,
      avg_compute_gbps,
      stddev_compute_gbps,
      results[gpu_iterations]);

  free(cpu_results);

  return 0;
}
