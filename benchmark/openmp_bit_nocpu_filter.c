/*

OpenMP GPU-only benchmark
Contains instrumented code to assess PCIe transfer overheads and GPU-only
execution times for containerized intersection counts. The test creates a
database of bitsets, performs intersection counts on the GPU, and reports
timings and speedup factors.

*/
#define _POSIX_C_SOURCE 199309L


#include "openmp_bit_helpers.h"
#include "openmp_bit_nocpu_filter.h"
#include "openmp_bit_nocpu_filter_GPU.h"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_GPU_ITERATIONS 1024
#define MIN_SIZE 128

static int parse_positive_size(const char *text, size_t *value) {
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

int *BitDB_diff_count_gpu_instrument(T_DB bit, T_DB bits,
                                      SETOP_COUNT_OPTS opts,
                                      GPU_Instrumentation *instr) {
  size_t nelem = (size_t)BitDB_nelem(bit) * BitDB_nelem(bits);
  int *counts = (int *)calloc(nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_diff_count_store_gpu_instrument(bit, bits, counts, opts, instr);
  return counts;
}

void BitDB_diff_count_store_gpu_instrument(T_DB bit, T_DB bits, int *counts,
                                            SETOP_COUNT_OPTS opts,
                                            GPU_Instrumentation *instr) {
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);
  _BitDB_diff_count_store_gpu(bit, bits, counts, opts, instr);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);
}

int database_match_GPU(Bit_DB_T db1, Bit_DB_T db2, SETOP_COUNT_OPTS opts) {
  int max = 0, current = 0, *results;
  results = BitDB_diff_count_gpu(db1, db2, opts);
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

/*
  HEAP IMPLEMENTATION
  

*/

static int candidate_better(int lhs_score, int lhs_id,
                            int rhs_score, int rhs_id) {
  /* Lower distance is better */
  return lhs_score < rhs_score ||
         (lhs_score == rhs_score && lhs_id < rhs_id);
}

static int candidate_worse(int lhs_score, int lhs_id,
                           int rhs_score, int rhs_id) {
  /* Higher distance is worse (belongs at the root to be discarded) */
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_id > rhs_id);
}

static void candidate_swap(int *scores, int *ids,
                           size_t lhs, size_t rhs) {
  int score = scores[lhs];
  int id = ids[lhs];

  scores[lhs] = scores[rhs];
  ids[lhs] = ids[rhs];
  scores[rhs] = score;
  ids[rhs] = id;
}

static void heap_sift_down(int *scores, int *ids,
                           size_t heap_size, size_t root) {
  for (;;) {
    size_t worst = root;
    size_t left = 2 * root + 1;
    size_t right = left + 1;

    if (left < heap_size &&
        candidate_worse(scores[left], ids[left],
                        scores[worst], ids[worst])) {
      worst = left;
    }

    if (right < heap_size &&
        candidate_worse(scores[right], ids[right],
                        scores[worst], ids[worst])) {
      worst = right;
    }

    if (worst == root) {
      return;
    }

    candidate_swap(scores, ids, root, worst);
    root = worst;
  }
}

static void select_topk_sorted(const int *scores,
                               size_t num_refs,
                               size_t top_k,
                               int *top_scores,
                               int *top_ids) {
  assert(scores && top_scores && top_ids);
  assert(top_k > 0 && top_k <= num_refs);
  assert(num_refs <= INT_MAX);

  for (size_t ref = 0; ref < top_k; ++ref) {
    top_scores[ref] = scores[ref];
    top_ids[ref] = (int)ref;
  }

  /* Construct a heap with the worst retained candidate at its root. */
  for (size_t parent = top_k / 2; parent > 0; --parent) {
    heap_sift_down(top_scores, top_ids, top_k, parent - 1);
  }

  for (size_t ref = top_k; ref < num_refs; ++ref) {
    if (candidate_better(scores[ref], (int)ref,
                         top_scores[0], top_ids[0])) {
      top_scores[0] = scores[ref];
      top_ids[0] = (int)ref;
      heap_sift_down(top_scores, top_ids, top_k, 0);
    }
  }

  /* Min-heap sort produces best-to-worst order. */
  for (size_t heap_size = top_k; heap_size > 1; --heap_size) {
    candidate_swap(top_scores, top_ids, 0, heap_size - 1);
    heap_sift_down(top_scores, top_ids, heap_size - 1, 0);
  }
}

static int database_match_GPU_filter_instrument(Bit_DB_T db1, Bit_DB_T db2,
                                                size_t top_k,
                                                SETOP_COUNT_OPTS opts,
                                                GPU_Instrumentation *instr) {
  const size_t num_queries = BitDB_nelem(db1);
  const size_t num_refs = BitDB_nelem(db2);

  assert(top_k > 0 && top_k <= num_refs);
  assert(num_queries <= SIZE_MAX / top_k);

  int *results;
  int *top_scores = malloc(num_queries * top_k * sizeof(*top_scores));
  int *top_ids = malloc(num_queries * top_k * sizeof(*top_ids));
  assert(top_scores && top_ids);

  clock_gettime(CLOCK_MONOTONIC, &instr->start_PCIe_time);
  results = BitDB_diff_count_gpu_instrument(db1, db2, opts, instr);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_PCIe_time);

  clock_gettime(CLOCK_MONOTONIC, &instr->start_CPU_overhead);

#pragma omp parallel for schedule(static)
  for (size_t query = 0; query < num_queries; ++query) {
    select_topk_sorted(
        results + query * num_refs,
        num_refs,
        top_k,
        top_scores + query * top_k,
        top_ids + query * top_k);
  }

  free(results);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_CPU_overhead);

int min_score = INT_MAX;

#pragma omp parallel for reduction(min : min_score)
for (size_t candidate = 0; candidate < num_queries * top_k; ++candidate) {
  if (top_scores[candidate] < min_score) {
    min_score = top_scores[candidate];
  }
}

  free(top_ids);
  free(top_scores);
  return min_score;
}

int main(int argc, char *argv[]) {
  if (argc != 6 && argc != 7) {
    fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference "
            "bitsets> <top-k> <gpu iterations> [<gpu_id>]\n",
            argv[0]);
    fprintf(stderr, "Example: %s 1024 1000 1000000 256 10 0\n", argv[0]);
    fprintf(stderr,
            "This will create 1000 bitsets of size 1024 and run 10 GPU-only "
            "containerized intersection-count iterations on GPU 0 and provide the top 256 results.\n");
    return EXIT_FAILURE;
  }

     printf(" %-20s : %d\n", "OpenMP version", _OPENMP);

  int num_devices = omp_get_num_devices();
    printf("OpenMP detected %d capable GPU devices.\n", num_devices);


  int size = atoi(argv[1]);
  int num_of_bits = atoi(argv[2]);
  int num_of_ref_bits = atoi(argv[3]);
  size_t top_k = 0;
  int gpu_iterations = atoi(argv[5]);
  int gpu_id = 0;
  if (!parse_positive_size(argv[4], &top_k)) {
    fprintf(stderr, "Error: top-k must be a positive integer.\n");
    return EXIT_FAILURE;
  }
  if (argc == 7) {
    gpu_id = atoi(argv[6]);
  }

  if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 ||
      gpu_iterations <= 0) {
    fprintf(stderr, "Error: size, number of bits, number of ref bits, top-k, and GPU "
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
  if (top_k > (size_t)num_of_ref_bits) {
    fprintf(stderr, "Warning: top-k decreased to %d\n", num_of_ref_bits);
    top_k = (size_t)num_of_ref_bits;
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
#ifdef OPENMP_GPU_IMPL_SHARED_TILE_ILP
  printf("Using OpenMP GPU implementation: SHARED_TILE_ILP\n");
#endif
#ifdef USE_BUILTIN_POPCOUNT
  printf("Using OpenMP GPU popcount: builtin\n");
#else
  printf("Using OpenMP GPU popcount: WWG\n");
#endif
  printf("CPU heap selection: top %zu candidates per query\n", top_k);
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

  puts("\t\t ... first 5 queries and references");
  for (size_t i = 0; i < 5 && i < num_of_bits; ++i) {
    printf("Query %zu: 0x%016llx\n", i, (unsigned long long)h_queries[i]);
  }
  for (size_t i = 0; i < 5 && i < num_of_ref_bits; ++i) {
    printf("Ref %zu: 0x%016llx\n", i, (unsigned long long)h_refs[i]);
  }

  puts("Computing CPU reference results...");
  compute_cpu_popcount_xor_reference(h_queries, h_refs, size, num_of_bits,
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
  int64_t filter_timings[MAX_GPU_ITERATIONS + 1];
  int results[MAX_GPU_ITERATIONS + 1];
  int64_t end_to_end_timings[MAX_GPU_ITERATIONS + 1];
  GPU_Instrumentation instr = {0};
  // burn-in iteration to mitigate cold-start overheads
  int max =
      database_match_GPU_filter_instrument(
        db1, db2, top_k,
        (SETOP_COUNT_OPTS){.device_id = gpu_id,
                 .upd_1st_operand = true,
                 .upd_2nd_operand = true},
        &instr);

  puts("Completed burn-in iteration to warm up GPU and PCIe paths");

  for (int i = 1; i <= gpu_iterations; i++) {
    max = database_match_GPU_filter_instrument(
      db1, db2, top_k,
        (SETOP_COUNT_OPTS){.device_id = gpu_id,
                           .upd_1st_operand = true,
                           .upd_2nd_operand = false},
        &instr);
    timings[i] = timeDiff(&instr.end_time, &instr.start_time);
    filter_timings[i] =
        timeDiff(&instr.end_CPU_overhead, &instr.start_CPU_overhead);
    end_to_end_timings[i] = timings[i]  +
                            filter_timings[i];
    results[i] = max;
  }

  size_t agreements = 0;
  size_t disagreements = 0;
  uint32_t verify_max = 0;
  int *gpu_counts = BitDB_diff_count_gpu_instrument(
      db1, db2,
      (SETOP_COUNT_OPTS){.device_id = gpu_id,
                         .upd_1st_operand = false,
                         .upd_2nd_operand = false},
      &instr);
  compare_gpu_to_cpu_results(gpu_counts, cpu_results, num_of_bits,
                             num_of_ref_bits, &agreements, &disagreements,
                             &verify_max);
  free(gpu_counts);

  database_match_GPU_filter_instrument(
      db1, db2, top_k,
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
  double avg_filter_operation_time = 0.0;
  double stddev_filter_operation_time = 0.0;
  compute_int64_mean_stddev(&filter_timings[1], gpu_iterations,
                            &avg_filter_operation_time,
                            &stddev_filter_operation_time);
  double avg_end_to_end_time = 0.0;
  double stddev_end_to_end_time = 0.0;
  compute_int64_mean_stddev(&end_to_end_timings[1], gpu_iterations,
                            &avg_end_to_end_time,
                            &stddev_end_to_end_time);

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


  puts("Filter Timings:");
  for (int i = 1; i <= gpu_iterations; i++) {
    summarize_results("Container - GPU - OpenMP", filter_timings[i], i,
                      results[i],
                      (float)filter_timings[1] / filter_timings[i]);
  }

  puts("End-to-End GPU + Filter Timings (Component Sum):");
  for (int i = 1; i <= gpu_iterations; i++) {
    summarize_results("Container - GPU - OpenMP Filter Total",
                      end_to_end_timings[i], i, results[i],
                      (float)end_to_end_timings[1] / end_to_end_timings[i]);
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
  double filter_gbps[MAX_GPU_ITERATIONS];
  double end_to_end_gbps[MAX_GPU_ITERATIONS];
  for (int i = 1; i <= gpu_iterations; i++) {
    compute_gbps[i - 1] = payload_per_iteration / ((double)timings[i] / 1E9);
    filter_gbps[i - 1] = payload_per_iteration / ((double)filter_timings[i] / 1E9);
    end_to_end_gbps[i - 1] =
        payload_per_iteration / ((double)end_to_end_timings[i] / 1E9);
  }
  double avg_compute_gbps = 0.0;
  double stddev_compute_gbps = 0.0;
  compute_mean_stddev(compute_gbps, gpu_iterations,
                      &avg_compute_gbps, &stddev_compute_gbps);
  double avg_filter_gbps = 0.0;
  double stddev_filter_gbps = 0.0;
  compute_mean_stddev(filter_gbps, gpu_iterations,
                      &avg_filter_gbps, &stddev_filter_gbps);
  double avg_end_to_end_gbps = 0.0;
  double stddev_end_to_end_gbps = 0.0;
  compute_mean_stddev(end_to_end_gbps, gpu_iterations,
                      &avg_end_to_end_gbps, &stddev_end_to_end_gbps);
  double avg_end_to_end_searches_per_sec =
      avg_end_to_end_time > 0.0 ? 1E9 / avg_end_to_end_time : 0.0;

    puts("\nEstimated Throughput (iterations 1-N, steady-state):");
    printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n",
      avg_algorithm_time, stddev_algorithm_time);
    printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
      avg_compute_gbps, stddev_compute_gbps);
    printf("Filter operation time: mean=%.3f ns, stddev=%.3f ns\n",
      avg_filter_operation_time, stddev_filter_operation_time);
    printf("Filter operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
      avg_filter_gbps, stddev_filter_gbps);
    printf("End-to-end time: mean=%.3f ns, stddev=%.3f ns\n",
      avg_end_to_end_time, stddev_end_to_end_time);
    printf("End-to-end throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
      avg_end_to_end_gbps, stddev_end_to_end_gbps);
  puts("\nNote: Total operation throughput includes GPU compute time, data "
       "staging,");
  puts("      and PCIe transfers combined, representing user-perceived "
       "performance.");

// 1. Extract the method string to avoid ugly inline macros inside the printf
#ifdef USE_BUILTIN_POPCOUNT
    const char* method_suffix = "builtin";
#else
    const char* method_suffix = "WWG";
#endif

    // 2. Beautifully format the OpenMP Summary output
    printf("\n"
           "================ OPENMP SUMMARY ================\n"
           "Method                     : OpenMP-Intersection-%s\n"
           "Bitset Bits                : %d\n"
           "Elements                   : %d\n"
           "Iterations                 : %d\n"
           "Avg Time (ns)              : %.3lf\n"
           "StdDev Time (ns)           : %.3lf\n"
           "Throughput (GB/s)          : %.6lf\n"
           "Throughput StdDev (GB/s)   : %.6lf\n"
           "Max Results                : %d\n"
           "Filter Avg Time (ns)       : %.3lf\n"
           "Filter StdDev Time (ns)    : %.3lf\n"
           "Filter Throughput (GB/s)   : %.6lf\n"
           "Filter Throughput StdDev   : %.6lf\n"
           "================================================\n",
           method_suffix, size, num_of_bits, gpu_iterations,
           avg_algorithm_time, stddev_algorithm_time, avg_compute_gbps,
           stddev_compute_gbps, results[gpu_iterations], avg_end_to_end_time,
           stddev_end_to_end_time, avg_end_to_end_gbps, stddev_end_to_end_gbps);

    // 3. Beautifully format the Search Summary output
    printf("\n"
           "================ SEARCH SUMMARY ================\n"
           "Backend                    : OpenMP\n"
           "Method                     : OpenMP-Intersection-%s\n"
           "Device                     : GPU_%d\n"
           "Score Type                 : intersection_count\n"
           "Score Order                : max\n"
           "Selection                  : CPU_topk_heap\n"
           "Timing Scope               : component_sum\n"
           "Bitset Bits                : %d\n"
           "Num Queries                : %d\n"
           "Num Refs                   : %d\n"
           "Top K                      : %zu\n"
           "Iterations                 : %d\n"
           "E2E Avg Time (ns)          : %.3lf\n"
           "E2E StdDev Time (ns)       : %.3lf\n"
           "E2E Searches/sec           : %.6lf\n"
           "Best Score                 : %d\n"
           "================================================\n",
           method_suffix, gpu_id, size, num_of_bits, num_of_ref_bits,
           top_k, gpu_iterations, avg_end_to_end_time, stddev_end_to_end_time,
           avg_end_to_end_searches_per_sec, results[gpu_iterations]);

  free(cpu_results);

  return 0;
}
