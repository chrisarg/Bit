/*

OpenMP GPU-only FAISS-comparison benchmark
Contains instrumented code to assess PCIe transfer overheads and GPU-only
execution times for containerized Hamming-distance counts, followed by
device-side top-k selection. The test creates a database of bitsets, performs
XOR popcount counts on the GPU through the PUBLIC Bit library API, selects the
top-k candidates per query on the device, and reports timings comparable to
the FAISS GPU benchmark script (scripts/faiss_gpu_benchmark.py).

This translation unit intentionally consumes ONLY the public bit.h interface:
all GPU kernel layout/offload machinery lives inside libbit.
*/
#define _POSIX_C_SOURCE 199309L

#include "openmp_bit_faiss_bench.h"

#define T Bit_T
#define T_DB Bit_DB_T

#define MAX_GPU_ITERATIONS 1024
#define MIN_SIZE 128

typedef Bench_Instrumentation GPU_Instrumentation;

void BitDB_diff_count_store_gpu_instrument(T_DB bit, T_DB bits, int *counts,
                                           SETOP_COUNT_OPTS opts,
                                           GPU_Instrumentation *instr) {
  clock_gettime(CLOCK_MONOTONIC, &instr->start_time);
  BitDB_diff_count_store_gpu(bit, bits, counts, opts);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_time);
}

int *BitDB_diff_count_gpu_instrument(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts,
                                     GPU_Instrumentation *instr) {
  size_t nelem = (size_t)BitDB_nelem(bit) * BitDB_nelem(bits);
  int *counts = (int *)calloc(nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_diff_count_store_gpu_instrument(bit, bits, counts, opts, instr);
  return counts;
}

FilteredResults database_match_GPU_filter_instrument(Bit_DB_T db1, Bit_DB_T db2,
                                                     size_t top_k,
                                                     SETOP_COUNT_OPTS opts,
                                                     GPU_Instrumentation *instr,
                                                     bool return_filtered) {
  const size_t num_queries = BitDB_nelem(db1);
  const size_t num_refs = BitDB_nelem(db2);
  size_t nelem = num_queries * num_refs;
  assert(top_k > 0 && top_k <= num_refs);
  assert(num_queries <= SIZE_MAX / top_k);

  int *top_scores = malloc(num_queries * top_k * sizeof(*top_scores));
  int *top_ids = malloc(num_queries * top_k * sizeof(*top_ids));
  assert(top_scores && top_ids);

  int *results = BitDB_diff_count_gpu_instrument(db1, db2, opts, instr);

  /* The counts buffer is resident on the device (defer_counts_transfer);
   * obtain its device address for the device-side top-k selection. */
  int *device_results = NULL;
#pragma omp target data map(alloc : results[0 : nelem])                        \
    use_device_ptr(results) device(opts.device_id)
  {
    device_results = results;
  }
  assert(device_results != NULL &&
         "Error: failed to obtain device pointer for the counts buffer");

  clock_gettime(CLOCK_MONOTONIC, &instr->start_CPU_overhead);
  topk_int_omp_gpu(device_results, num_queries, num_refs, top_k, top_scores,
                   top_ids, opts.device_id);
  clock_gettime(CLOCK_MONOTONIC, &instr->end_CPU_overhead);

  int min_score = INT_MAX;
#pragma omp parallel for reduction(min : min_score)
  for (size_t candidate = 0; candidate < num_queries * top_k; ++candidate) {
    if (top_scores[candidate] < min_score) {
      min_score = top_scores[candidate];
    }
  }

#pragma omp target exit data map(delete : results[0 : nelem])                  \
    device(opts.device_id)
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
            "bitsets> <top-k> <gpu iterations> [<gpu_id>]\n",
            argv[0]);
    fprintf(stderr, "Example: %s 1024 1000 1000000 256 10 0\n", argv[0]);
    fprintf(stderr,
            "This will create 1000 bitsets of size 1024 and run 10 GPU-only "
            "containerized intersection-count iterations on GPU 0 and provide "
            "the top 256 results.\n");
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
    fprintf(stderr,
            "Error: size, number of bits, number of ref bits, top-k, and GPU "
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

#if defined(OPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD)
  printf("Using OpenMP GPU implementation: TEAM_PARALLEL_SIMD\n");
#elif defined(OPENMP_GPU_IMPL_TRANSPOSED_TEAM_PARALLEL_SIMD)
  printf("Using OpenMP GPU implementation: TRANSPOSED_TEAM_PARALLEL_SIMD\n");
#endif
#ifdef USE_BUILTIN_POPCOUNT
  printf("Using OpenMP GPU popcount: builtin\n");
#else
  printf("Using OpenMP GPU popcount: WWG\n");
#endif
  printf("GPU heap selection: top %zu candidates per query\n", top_k);
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

  int64_t timings[MAX_GPU_ITERATIONS + 1];
  int64_t filter_timings[MAX_GPU_ITERATIONS + 1];
  int results[MAX_GPU_ITERATIONS + 1];
  int64_t end_to_end_timings[MAX_GPU_ITERATIONS + 1];
  GPU_Instrumentation instr = {0};

  /* The GPU benchmark always pairs the device count kernel with the device
   * top-k selection: db1 is re-uploaded each iteration, db2 stays resident,
   * and the counts buffer remains on the device for the device top-k. */
  SETOP_COUNT_OPTS opts = (SETOP_COUNT_OPTS){.device_id = gpu_id,
                                             .upd_1st_operand = true,
                                             .upd_2nd_operand = false,
                                             .release_1st_operand = true,
                                             .defer_counts_transfer = true,
                                             .release_counts = false};

  FilteredResults filtered_results = database_match_GPU_filter_instrument(
      db1, db2, top_k, opts, &instr, false);

  puts("Completed burn-in iteration to warm up GPU and PCIe paths");
  for (int i = 1; i <= gpu_iterations; i++) {
    filtered_results = database_match_GPU_filter_instrument(
        db1, db2, top_k, opts, &instr, i < gpu_iterations ? false : true);
    timings[i] = timeDiff(&instr.end_time, &instr.start_time);
    filter_timings[i] =
        timeDiff(&instr.end_CPU_overhead, &instr.start_CPU_overhead);
    end_to_end_timings[i] = timings[i] + filter_timings[i];
    results[i] = filtered_results.max;
  }

  size_t agreements = 0;
  size_t disagreements = 0;
  uint32_t verify_max = 0;
  int *gpu_counts = BitDB_diff_count_gpu_instrument(
      db1, db2,
      (SETOP_COUNT_OPTS){.device_id = gpu_id,
                         .upd_1st_operand = false,
                         .upd_2nd_operand = false,
                         .release_1st_operand = true,
                         .release_2nd_operand = true,
                         .defer_counts_transfer = false,
                         .release_counts = false},
      &instr);
  compare_gpu_to_cpu_results(gpu_counts, cpu_results, num_of_bits,
                             num_of_ref_bits, &agreements, &disagreements,
                             &verify_max);
  free(gpu_counts);

  // scaling factors for averaging across iterations
  double avg_algorithm_time = 0.0;
  double stddev_algorithm_time = 0.0;
  compute_int64_mean_stddev(&timings[1], gpu_iterations, &avg_algorithm_time,
                            &stddev_algorithm_time);
  double avg_filter_operation_time = 0.0;
  double stddev_filter_operation_time = 0.0;
  compute_int64_mean_stddev(&filter_timings[1], gpu_iterations,
                            &avg_filter_operation_time,
                            &stddev_filter_operation_time);
  double avg_end_to_end_time = 0.0;
  double stddev_end_to_end_time = 0.0;
  compute_int64_mean_stddev(&end_to_end_timings[1], gpu_iterations,
                            &avg_end_to_end_time, &stddev_end_to_end_time);

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
                      results[i], (float)filter_timings[1] / filter_timings[i]);
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

  puts(
      "Printing filtered results and scores (five scores, first five queries)");
  for (int i = 0; i < 5 && i < num_of_bits; i++) {
    printf("|Query %d:\t|", i);
    for (int j = 0; j < 5 && j < (int)top_k; j++) {
      printf("  Score: %5d, ID: %5d |",
             filtered_results.top_scores[i * (int)top_k + j],
             filtered_results.top_ids[i * (int)top_k + j]);
    }
    printf("\n");
  }

  double compute_gbps[MAX_GPU_ITERATIONS];
  double filter_gbps[MAX_GPU_ITERATIONS];
  double end_to_end_gbps[MAX_GPU_ITERATIONS];
  for (int i = 1; i <= gpu_iterations; i++) {
    compute_gbps[i - 1] = payload_per_iteration / ((double)timings[i] / 1E9);
    filter_gbps[i - 1] =
        payload_per_iteration / ((double)filter_timings[i] / 1E9);
    end_to_end_gbps[i - 1] =
        payload_per_iteration / ((double)end_to_end_timings[i] / 1E9);
  }
  double avg_compute_gbps = 0.0;
  double stddev_compute_gbps = 0.0;
  compute_mean_stddev(compute_gbps, gpu_iterations, &avg_compute_gbps,
                      &stddev_compute_gbps);
  double avg_filter_gbps = 0.0;
  double stddev_filter_gbps = 0.0;
  compute_mean_stddev(filter_gbps, gpu_iterations, &avg_filter_gbps,
                      &stddev_filter_gbps);
  double avg_end_to_end_gbps = 0.0;
  double stddev_end_to_end_gbps = 0.0;
  compute_mean_stddev(end_to_end_gbps, gpu_iterations, &avg_end_to_end_gbps,
                      &stddev_end_to_end_gbps);
  double avg_end_to_end_searches_per_sec =
      avg_end_to_end_time > 0.0 ? 1E9 / avg_end_to_end_time : 0.0;

  puts("\nEstimated Throughput (iterations 1-N, steady-state):");
  printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n", avg_algorithm_time,
         stddev_algorithm_time);
  printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
         avg_compute_gbps, stddev_compute_gbps);
  printf("Filter operation time: mean=%.3f ns, stddev=%.3f ns\n",
         avg_filter_operation_time, stddev_filter_operation_time);
  printf("Filter operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
         avg_filter_gbps, stddev_filter_gbps);
  printf("End-to-end time: mean=%.3f ns, stddev=%.3f ns\n", avg_end_to_end_time,
         stddev_end_to_end_time);
  printf("End-to-end throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
         avg_end_to_end_gbps, stddev_end_to_end_gbps);
  puts("\nNote: Total operation throughput includes GPU compute time, data "
       "staging,");
  puts("      and PCIe transfers combined, representing user-perceived "
       "performance.");

#if defined(USE_BUILTIN_POPCOUNT)
  const char *method_suffix = "builtin";
#elif defined(USE_LIBPOPCNT) && USE_LIBPOPCNT
  const char *method_suffix = "libpopcnt";
#else
  const char *method_suffix = "SIMDe";
#endif

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
         method_suffix, size, num_of_bits, gpu_iterations, avg_algorithm_time,
         stddev_algorithm_time, avg_compute_gbps, stddev_compute_gbps,
         results[gpu_iterations], avg_end_to_end_time, stddev_end_to_end_time,
         avg_end_to_end_gbps, stddev_end_to_end_gbps);

  printf("\n"
         "================ SEARCH SUMMARY ================\n"
         "Backend                    : OpenMP\n"
         "Method                     : OpenMP-Intersection-%s\n"
         "Device                     : GPU_%d\n"
         "Score Type                 : hamming_distance\n"
         "Score Order                : max\n"
         "Selection                  : GPU_topk_heap\n"
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
         method_suffix, gpu_id, size, num_of_bits, num_of_ref_bits, top_k,
         gpu_iterations, avg_end_to_end_time, stddev_end_to_end_time,
         avg_end_to_end_searches_per_sec, results[gpu_iterations]);

  free(filtered_results.top_scores);
  free(filtered_results.top_ids);
  free(cpu_results);

  return 0;
}
