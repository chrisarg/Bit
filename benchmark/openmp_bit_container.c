/* Isolated CPU Bit_DB intersection-count benchmark. */
#define _POSIX_C_SOURCE 199309L

#include "bit.h"
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int64_t elapsed_ns(const struct timespec *start,
                          const struct timespec *end) {
  return (int64_t)(end->tv_sec - start->tv_sec) * INT64_C(1000000000) +
         end->tv_nsec - start->tv_nsec;
}

static int parse_positive(const char *text, const char *name) {
  char *end = NULL;
  long value = strtol(text, &end, 10);

  if (*text == '\0' || *end != '\0' || value <= 0 || value > INT_MAX) {
    fprintf(stderr, "%s must be an integer in [1, %d]\n", name, INT_MAX);
    exit(EXIT_FAILURE);
  }
  return (int)value;
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    fprintf(stderr,
            "Usage: %s <bits> <left-bitsets> <right-bitsets> <threads> "
            "<repetitions>\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  const int bit_length = parse_positive(argv[1], "bits");
  const int left_count = parse_positive(argv[2], "left-bitsets");
  const int right_count = parse_positive(argv[3], "right-bitsets");
  const int threads = parse_positive(argv[4], "threads");
  const int repetitions = parse_positive(argv[5], "repetitions");
  const size_t result_count = (size_t)left_count * (size_t)right_count;

  if (result_count > SIZE_MAX / sizeof(int)) {
    fputs("Result matrix is too large\n", stderr);
    return EXIT_FAILURE;
  }

  Bit_DB_T left = BitDB_new(bit_length, left_count);
  Bit_DB_T right = BitDB_new(bit_length, right_count);
  Bit_T template = Bit_new(bit_length);
  Bit_set(template, bit_length / 2, bit_length - 1);

  for (int i = 0; i < left_count; ++i) {
    BitDB_put_at(left, i, template);
  }
  for (int j = 0; j < right_count; ++j) {
    BitDB_put_at(right, j, template);
  }
  Bit_free(&template);

  int *results = malloc(result_count * sizeof(*results));
  if (results == NULL) {
    fputs("Unable to allocate result matrix\n", stderr);
    BitDB_free(&left);
    BitDB_free(&right);
    return EXIT_FAILURE;
  }

  const SETOP_COUNT_OPTS opts = {.num_cpu_threads = threads};
  print_Bit_configuration();

  /* Establish mappings, page placement, and OpenMP worker state. */
  BitDB_inter_count_store_cpu(left, right, results, opts);

  int64_t total_ns = 0;
  int64_t best_ns = INT64_MAX;
  for (int run = 0; run < repetitions; ++run) {
    struct timespec start;
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    BitDB_inter_count_store_cpu(left, right, results, opts);
    clock_gettime(CLOCK_MONOTONIC, &end);

    const int64_t duration = elapsed_ns(&start, &end);
    total_ns += duration;
    if (duration < best_ns) {
      best_ns = duration;
    }
    printf("run %d: %" PRId64 " ns\n", run + 1, duration);
  }

  uint64_t checksum = 0;
  for (size_t i = 0; i < result_count; ++i) {
    checksum += (uint32_t)results[i];
  }

  const double qword_reductions =
      (double)left_count * right_count * ((bit_length + 63) / 64);
  const double average_ns = (double)total_ns / repetitions;
  printf("best:    %" PRId64 " ns (%.3f Gqword-pairs/s)\n", best_ns,
         qword_reductions / best_ns);
  printf("average: %.0f ns (%.3f Gqword-pairs/s)\n", average_ns,
         qword_reductions / average_ns);
  printf("checksum: %" PRIu64 "\n", checksum);

  free(results);
  BitDB_free(&left);
  BitDB_free(&right);
  return EXIT_SUCCESS;
}
