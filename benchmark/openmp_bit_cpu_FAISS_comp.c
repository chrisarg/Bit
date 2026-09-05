/*

OpenMP CPU-only benchmark
Contains instrumented code to assess CPU execution times for containerized
Hamming-distance counts and CPU top-k selection. The test creates a database
of bitsets, performs XOR popcount counts on the CPU, selects the top-k
candidates per query, and reports timings comparable to the FAISS CPU
benchmark script.

*/
#define _POSIX_C_SOURCE 199309L

#include "openmp_bit_faiss_bench.h"

#define T Bit_T
#define T_DB Bit_DB_T

#define MAX_ITERATIONS 1024
#define MAX_THREADS 1024
#define MIN_SIZE 128

typedef Bench_Instrumentation CPU_Instrumentation;

void BitDB_diff_count_store_cpu_instrument(T_DB bit, T_DB bits, int *counts,
                                           SETOP_COUNT_OPTS opts,
                                           CPU_Instrumentation *instr) {
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);
  BitDB_diff_count_store_cpu(bit, bits, counts, opts);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);
}

int *BitDB_diff_count_cpu_instrument(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts,
                                     CPU_Instrumentation *instr) {
  size_t nelem = (size_t)BitDB_nelem(bit) * BitDB_nelem(bits);
  int *counts = (int *)calloc(nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_diff_count_store_cpu_instrument(bit, bits, counts, opts, instr);
  return counts;
}

FilteredResults database_match_cpu_filter_instrument(Bit_DB_T db1, Bit_DB_T db2,
                                                     size_t top_k,
                                                     SETOP_COUNT_OPTS opts,
                                                     CPU_Instrumentation *instr,
                                                     bool return_filtered) {
  const size_t num_queries = BitDB_nelem(db1);
  const size_t num_refs = BitDB_nelem(db2);
  assert(top_k > 0 && top_k <= num_refs);
  assert(num_queries <= SIZE_MAX / top_k);

  int *top_scores = malloc(num_queries * top_k * sizeof(*top_scores));
  int *top_ids = malloc(num_queries * top_k * sizeof(*top_ids));
  assert(top_scores && top_ids);

  int *results = BitDB_diff_count_cpu_instrument(db1, db2, opts, instr);

  clock_gettime(CLOCK_MONOTONIC, &instr->start_CPU_overhead);
  topk_int_omp_cpu(results, num_queries, num_refs, top_k, top_scores, top_ids,
                   0);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_CPU_overhead);

  int min_score = INT_MAX;
#pragma omp parallel for reduction(min : min_score)
  for (size_t candidate = 0; candidate < num_queries * top_k; ++candidate) {
    if (top_scores[candidate] < min_score) {
      min_score = top_scores[candidate];
    }
  }

  free(results);

  FilteredResults filtered_results;
  if (return_filtered) {
    filtered_results = (FilteredResults){
        .max = min_score, .top_scores = top_scores, .top_ids = top_ids};
    return filtered_results;
  }
  filtered_results =
      (FilteredResults){.max = min_score, .top_scores = NULL, .top_ids = NULL};

  free(top_ids);
  free(top_scores);

  return filtered_results;
}

int main(int argc, char *argv[]) {
  if (argc != 6 && argc != 7) {
    fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference "
            "bitsets> <top-k> <iterations> [threads]\n",
            argv[0]);
    fprintf(stderr, "Example: %s 1024 1000 1000000 256 10 32\n", argv[0]);
    fprintf(stderr,
            "This will create 1000 bitsets of size 1024 and run 10 CPU-only "
            "containerized Hamming-distance iterations using 32 threads (or "
            "all available cores if omitted) and provide the top 256 results.\n");
    return EXIT_FAILURE;
  }

  printf(" %-20s : %d\n", "OpenMP version", _OPENMP);

  int size = atoi(argv[1]);
  int num_of_bits = atoi(argv[2]);
  int num_of_ref_bits = atoi(argv[3]);
  size_t top_k = 0;
  int iterations = atoi(argv[5]);
  int num_threads = 0;

  if (!parse_positive_size(argv[4], &top_k)) {
    fprintf(stderr, "Error: top-k must be a positive integer.\n");
    return EXIT_FAILURE;
  }
  if (argc == 7) {
    if (!parse_positive_int(argv[6], &num_threads)) {
      fprintf(stderr, "Error: threads must be a positive integer.\n");
      return EXIT_FAILURE;
    }
  } else {
    num_threads = omp_get_max_threads();
  }

  if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 || iterations <= 0) {
    fprintf(stderr,
            "Error: size, number of bits, number of ref bits, top-k, and "
            "iterations must be positive integers.\n");
    return EXIT_FAILURE;
  }

  if (iterations > MAX_ITERATIONS) {
    fprintf(stderr, "Warning: iterations capped to %d\n", MAX_ITERATIONS);
    iterations = MAX_ITERATIONS;
  }
  if (size < MIN_SIZE) {
    fprintf(stderr, "Warning: size increased to %d\n", MIN_SIZE);
    size = MIN_SIZE;
  }
  if (top_k > (size_t)num_of_ref_bits) {
    fprintf(stderr, "Warning: top-k decreased to %d\n", num_of_ref_bits);
    top_k = (size_t)num_of_ref_bits;
  }
  if (num_threads > MAX_THREADS) {
    fprintf(stderr, "Warning: threads capped to %d\n", MAX_THREADS);
    num_threads = MAX_THREADS;
  }

#ifndef NDEBUG
  printf("Debug mode is enabled.\n");
#else
  printf("Debug mode is disabled.\n");
#endif

  printf("CPU threads used: %d\n", num_threads);
  printf("CPU heap selection: top %zu candidates per query\n", top_k);
  printf("Starting CPU-only benchmark\n");

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
  for (size_t i = 0; i < 5 && i < (size_t)num_of_bits; ++i) {
    printf("Query %zu: 0x%016llx\n", i, (unsigned long long)h_queries[i]);
  }
  for (size_t i = 0; i < 5 && i < (size_t)num_of_ref_bits; ++i) {
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

  int64_t timings[MAX_ITERATIONS + 1];
  int64_t filter_timings[MAX_ITERATIONS + 1];
  int results[MAX_ITERATIONS + 1];
  int64_t end_to_end_timings[MAX_ITERATIONS + 1];
  CPU_Instrumentation instr = {0};

  SETOP_COUNT_OPTS opts = (SETOP_COUNT_OPTS){.num_cpu_threads = num_threads};

  FilteredResults filtered_results = database_match_cpu_filter_instrument(
      db1, db2, top_k, opts, &instr, false);

  puts("Completed burn-in iteration to warm up CPU and memory paths");
  for (int i = 1; i <= iterations; i++) {
    filtered_results = database_match_cpu_filter_instrument(
        db1, db2, top_k, opts, &instr, i < iterations ? false : true);
    timings[i] = timeDiff(&instr.end_time, &instr.start_time);
    filter_timings[i] =
        timeDiff(&instr.end_CPU_overhead, &instr.start_CPU_overhead);
    end_to_end_timings[i] = timings[i] + filter_timings[i];
    results[i] = filtered_results.max;
  }

  size_t agreements = 0;
  size_t disagreements = 0;
  uint32_t verify_max = 0;
  int *library_counts = BitDB_diff_count_cpu_instrument(
      db1, db2, (SETOP_COUNT_OPTS){.num_cpu_threads = num_threads}, &instr);
  compare_gpu_to_cpu_results(library_counts, cpu_results, num_of_bits,
                             num_of_ref_bits, &agreements, &disagreements,
                             &verify_max);
  free(library_counts);

  double avg_algorithm_time = 0.0;
  double stddev_algorithm_time = 0.0;
  compute_int64_mean_stddev(&timings[1], iterations, &avg_algorithm_time,
                            &stddev_algorithm_time);
  double avg_filter_operation_time = 0.0;
  double stddev_filter_operation_time = 0.0;
  compute_int64_mean_stddev(&filter_timings[1], iterations,
                            &avg_filter_operation_time,
                            &stddev_filter_operation_time);
  double avg_end_to_end_time = 0.0;
  double stddev_end_to_end_time = 0.0;
  compute_int64_mean_stddev(&end_to_end_timings[1], iterations,
                            &avg_end_to_end_time, &stddev_end_to_end_time);

  puts("CPU Algorithm Timing:");
  for (int i = 1; i <= iterations; i++) {
    summarize_results("Container - CPU - OpenMP", timings[i], i, results[i],
                      (float)timings[1] / timings[i]);
  }

  puts("Filter Timings:");
  for (int i = 1; i <= iterations; i++) {
    summarize_results("Container - CPU - OpenMP", filter_timings[i], i,
                      results[i], (float)filter_timings[1] / filter_timings[i]);
  }

  puts("End-to-End CPU + Filter Timings (Component Sum):");
  for (int i = 1; i <= iterations; i++) {
    summarize_results("Container - CPU - OpenMP Filter Total",
                      end_to_end_timings[i], i, results[i],
                      (float)end_to_end_timings[1] / end_to_end_timings[i]);
  }

  printf("  agreements: %zu\n", agreements);
  printf("  disagreements: %zu\n", disagreements);
  if (disagreements > 0) {
    printf("  WARNING: CPU library results disagree with CPU reference\n");
  }

  puts(
      "Printing filtered results and scores (five scores, first five queries)");
  for (int i = 0; i < 5 && i < num_of_bits; i++) {
    printf("|Query %d:\t|", i);
    for (size_t j = 0; j < 5 && j < top_k; j++) {
      printf("  Score: %5d, ID: %5d |",
             filtered_results.top_scores[i * (int)top_k + (int)j],
             filtered_results.top_ids[i * (int)top_k + (int)j]);
    }
    printf("\n");
  }

  double avg_end_to_end_searches_per_sec =
      avg_end_to_end_time > 0.0 ? 1E9 / avg_end_to_end_time : 0.0;

  puts("\nEstimated Throughput (iterations 1-N, steady-state):");
  printf("CPU compute time: mean=%.3f ns, stddev=%.3f ns\n", avg_algorithm_time,
         stddev_algorithm_time);
  printf("Filter operation time: mean=%.3f ns, stddev=%.3f ns\n",
         avg_filter_operation_time, stddev_filter_operation_time);
  printf("End-to-end time: mean=%.3f ns, stddev=%.3f ns\n", avg_end_to_end_time,
         stddev_end_to_end_time);
  puts("\nNote: Total operation throughput includes CPU compute time and "
       "host-side top-k selection,");
  puts("      representing user-perceived performance.");

#if defined(USE_BUILTIN_POPCOUNT)
  const char *method_suffix = "builtin";
#elif defined(USE_LIBPOPCNT) && USE_LIBPOPCNT
  const char *method_suffix = "libpopcnt";
#else
  const char *method_suffix = "SIMDe";
#endif

  printf("\n"
         "================ OPENMP SUMMARY ================\n"
         "Method                     : OpenMP-CPU-Intersection-%s\n"
         "Bitset Bits                : %d\n"
         "Elements                   : %d\n"
         "Iterations                 : %d\n"
         "Avg Time (ns)              : %.3lf\n"
         "StdDev Time (ns)           : %.3lf\n"
         "Max Results                : %d\n"
         "Filter Avg Time (ns)       : %.3lf\n"
         "Filter StdDev Time (ns)    : %.3lf\n"
         "================================================\n",
         method_suffix, size, num_of_bits, iterations, avg_algorithm_time,
         stddev_algorithm_time, results[iterations], avg_filter_operation_time,
         stddev_filter_operation_time);

  printf("\n"
         "================ SEARCH SUMMARY ================\n"
         "Backend                    : OpenMP\n"
         "Method                     : OpenMP-CPU-Intersection-%s\n"
         "Device                     : CPU\n"
         "Score Type                 : hamming_distance\n"
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
         method_suffix, size, num_of_bits, num_of_ref_bits, top_k, iterations,
         avg_end_to_end_time, stddev_end_to_end_time,
         avg_end_to_end_searches_per_sec, results[iterations]);

  free(cpu_results);

  return 0;
}
