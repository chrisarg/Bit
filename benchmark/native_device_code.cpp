// Include CUDA/HIP runtime and standard types first.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Use the builtin popcount only when requested; otherwise use a portable
// WWG software implementation for both CUDA and HIP.
// The burn-in warmup is retained for benchmark stability. The GPU code path
// itself now uses the shared `gpu_kernels.h` kernel implementation.
#if defined(USE_BUILTIN_POPCOUNT)
#define POPCOUNT_METHOD_LABEL "builtin"
#else
#define POPCOUNT_METHOD_LABEL "WWG"
#endif

#ifndef GPU_TILE_DIM
#define GPU_TILE_DIM 32
#endif
#ifndef GPU_BLOCK_ROWS
#define GPU_BLOCK_ROWS 8
#endif
// Typical working parameter ranges for the benchmark:
//   num_queries: 10k - 100k
//   num_refs:    2k  - 16k
//   bit_size_in_qwords: 1024 - 4096
// A 128-qword shared tile gives 8-32 inner iterations for J in this range,
// which reduces loop overhead while keeping per-block shared memory small.
#define QUERY_WORD_TILE 128

#ifndef GPU_TILE_J
#define GPU_TILE_J 2048
#endif

#ifndef GPU_ILP
#define GPU_ILP 16
#endif

#include <algorithm>
#include <ctime>
#include <numeric>
#include <random>
#include <sys/stat.h>
#include <vector>

#include "openmp_bit_helpers.h"

#if defined(__HIPCC__)
typedef hipEvent_t gpu_event_t;
#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define GPU_MALLOC hipMalloc
#define GPU_MEMCPY hipMemcpy
#define GPU_MEMCPY_H2D hipMemcpyHostToDevice
#define GPU_MEMCPY_D2H hipMemcpyDeviceToHost
#define GPU_FREE hipFree
#define GPU_EVENT_CREATE(e) hipEventCreate(e)
#define GPU_EVENT_DESTROY(e) hipEventDestroy(e)
#define GPU_EVENT_RECORD(e, s) hipEventRecord(e, s)
#define GPU_EVENT_SYNC(e) hipEventSynchronize(e)
#define GPU_EVENT_ELAPSED_TIME(ms, start, end)                                 \
  hipEventElapsedTime(ms, start, end)
#define GPU_DEVICE_SYNC hipDeviceSynchronize()
#define GPU_GET_LAST_ERROR hipGetLastError()
#define GPU_SET_DEVICE hipSetDevice
#define BACKEND_NAME "HIP"
#define BACKEND_SUMMARY_LABEL "HIP_SUMMARY"
#else
typedef cudaEvent_t gpu_event_t;
#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define GPU_MALLOC cudaMalloc
#define GPU_MEMCPY cudaMemcpy
#define GPU_MEMCPY_H2D cudaMemcpyHostToDevice
#define GPU_MEMCPY_D2H cudaMemcpyDeviceToHost
#define GPU_FREE cudaFree
#define GPU_EVENT_CREATE(e) cudaEventCreate(e)
#define GPU_EVENT_DESTROY(e) cudaEventDestroy(e)
#define GPU_EVENT_RECORD(e, s) cudaEventRecord(e, s)
#define GPU_EVENT_SYNC(e) cudaEventSynchronize(e)
#define GPU_EVENT_ELAPSED_TIME(ms, start, end)                                 \
  cudaEventElapsedTime(ms, start, end)
#define GPU_DEVICE_SYNC cudaDeviceSynchronize()
#define GPU_GET_LAST_ERROR cudaGetLastError()
#define GPU_SET_DEVICE cudaSetDevice
#define BACKEND_NAME "CUDA"
#define BACKEND_SUMMARY_LABEL "CUDA_SUMMARY"
#endif

#include "gpu_kernels.h"

static inline int64_t time_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

void summarize_results(const char *test, int64_t timeElapsed, int iteration,
                       uint32_t result, float speedup) {
  printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
  printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
  printf("GPU iteration: %3d \t", iteration);
  printf("Result: %u\t", result);
  printf("Speedup factor: %.2f\n", speedup);
}

static bool csv_path_is_set(char **csv_path) {
  const char *path = getenv("GPU_CSV_OUTPUT");
  if (!path || path[0] == '\0') {
    *csv_path = nullptr;
    return false;
  }
  *csv_path = const_cast<char *>(path);
  return true;
}

static void append_csv_header(FILE *csv_file) {
  fprintf(csv_file, "GPU_TILE_J,GPU_ILP,number of bits in bitsets,number of "
                    "queries,number of reference sequences,iteration "
                    "count,timing type,timing (ns),iterations per second\n");
}

static void write_csv_row(FILE *csv_file, int tile_j, int ilp,
                          size_t bitset_bits, size_t num_queries,
                          size_t num_refs, int iteration,
                          const char *timing_type,
                          unsigned long long timing_ns) {
  const double ips = (timing_ns > 0) ? (1000000000.0 / (double)timing_ns) : 0.0;
  fprintf(csv_file, "%d,%d,%zu,%zu,%zu,%d,%s,%llu,%.3f\n", tile_j, ilp,
    bitset_bits, num_queries, num_refs, iteration, timing_type,
    timing_ns, ips);
}

static FILE *open_csv_output(const char *path) {
  FILE *csv = fopen(path, "a+");
  if (!csv) {
    fprintf(stderr, "Error: Unable to open CSV output file '%s'\n", path);
    return nullptr;
  }
  if (fseek(csv, 0, SEEK_END) == 0) {
    long length = ftell(csv);
    if (length == 0) {
      append_csv_header(csv);
    }
  }
  return csv;
}

static inline int64_t ms_to_ns(double ms) {
  return static_cast<int64_t>(ms * 1.0e6 + 0.5);
}

static inline size_t compute_words_per_bitset(size_t bitset_bits,
                                              unsigned int word_bits) {
  return (bitset_bits + word_bits - 1) / word_bits;
}

static void compare_gpu_to_cpu_results_wrapper(
    const uint32_t *gpu_results, const uint32_t *cpu_results,
    size_t num_queries, size_t num_refs, size_t *agreements,
    size_t *disagreements, uint32_t *max_result) {
  if (cpu_results == nullptr) {
    *agreements = 0;
    *disagreements = 0;
    return;
  }

  *agreements = 0;
  *disagreements = 0;
  *max_result = 0;
  const size_t nelem = num_queries * num_refs;
  for (size_t idx = 0; idx < nelem; ++idx) {
    const uint32_t current = gpu_results[idx];
    if (current > *max_result) {
      *max_result = current;
    }
    if (gpu_results[idx] == cpu_results[idx]) {
      ++*agreements;
    } else {
      if (*disagreements < 16) {
        fprintf(stderr, "Mismatch[%zu] gpu=0x%08x cpu=0x%08x\n", idx,
                gpu_results[idx], cpu_results[idx]);
      }
      ++*disagreements;
    }
  }
}

template <typename T, typename Engine>
static void fill_random_bitsets(T *bitsets, size_t num_bitsets,
                                size_t words_per_bitset, Engine &rng) {
  std::uniform_int_distribution<T> distribution(0);
  const size_t total_words = num_bitsets * words_per_bitset;
  for (size_t i = 0; i < total_words; ++i) {
    bitsets[i] = distribution(rng);
  }
}

static bool parse_word_bits(const char *arg, unsigned int *word_bits) {
  if (strcmp(arg, "32") == 0) {
    *word_bits = 32u;
    return true;
  }
  if (strcmp(arg, "64") == 0) {
    *word_bits = 64u;
    return true;
  }
  return false;
}

struct NativeBenchmarkResult {
  uint32_t *gpu_results;
  size_t num_pairs;
  double transpose_time_ns;
  double kernel_time_ns;
  double total_time_ns;
  double cpu_overhead_ns;
  double compute_gbps;
  double total_gbps;
  uint64_t checksum;
  size_t agreements;
  size_t disagreements;
  uint32_t max_result;
  std::vector<double> per_iter_transpose_ms;
  std::vector<double> per_iter_kernel_ms;
  std::vector<double> per_iter_total_ms;
  std::vector<double> per_iter_cpu_overhead_ms;
  std::vector<double> per_iter_d2h_ms;
  std::vector<uint32_t> per_iter_results;
};

template <typename T>
__global__ void transpose_kernel(const T *__restrict__ in, T *__restrict__ out,
                                 int width, int height) {
  __shared__ T tile[GPU_TILE_DIM][GPU_TILE_DIM + 1];

  int x = blockIdx.x * GPU_TILE_DIM + threadIdx.x;
  int y = blockIdx.y * GPU_TILE_DIM + threadIdx.y;

  for (int j = 0; j < GPU_TILE_DIM; j += GPU_BLOCK_ROWS) {
    if (x < width && (y + j) < height) {
      tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
  }

  __syncthreads();

  int x_t = blockIdx.y * GPU_TILE_DIM + threadIdx.x;
  int y_t = blockIdx.x * GPU_TILE_DIM + threadIdx.y;

  for (int j = 0; j < GPU_TILE_DIM; j += GPU_BLOCK_ROWS) {
    if (x_t < height && (y_t + j) < width) {
      out[(y_t + j) * height + x_t] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

template <typename T>
void run_setop_gpu(const T *d_bit_qwords, const T *d_bits_qwords_T,
                   int *d_counts, int K, int N, int J) {
  launch_setop_coarsened<T>(d_bit_qwords, d_bits_qwords_T, d_counts, K, N, J);
  GPU_CHECK(GPU_DEVICE_SYNC);
  GPU_CHECK(GPU_GET_LAST_ERROR);
}

template <typename T>
NativeBenchmarkResult
benchmark_native_gpu(const T *h_queries, const T *h_refs,
                     const uint32_t *cpu_results, size_t bitset_bits,
                     size_t num_queries, size_t num_refs, int gpu_iterations,
                     int gpu_id, unsigned int word_bits) {
  NativeBenchmarkResult result = {};
  const size_t words_per_bitset =
      compute_words_per_bitset(bitset_bits, word_bits);
  const size_t queries_bytes = words_per_bitset * num_queries * sizeof(T);
  const size_t refs_bytes = words_per_bitset * num_refs * sizeof(T);
  const size_t results_bytes = num_queries * num_refs * sizeof(uint32_t);

  T *d_queries = nullptr;
  T *d_refs = nullptr;
  T *d_refs_T = nullptr;
  int *d_results = nullptr;
  GPU_CHECK(GPU_MALLOC(&d_queries, queries_bytes));
  GPU_CHECK(GPU_MALLOC(&d_refs, refs_bytes));
  GPU_CHECK(GPU_MALLOC(&d_refs_T, refs_bytes));
  GPU_CHECK(GPU_MALLOC(&d_results, results_bytes));

  GPU_CHECK(GPU_MEMCPY(d_refs, h_refs, refs_bytes, GPU_MEMCPY_H2D));

  gpu_event_t transpose_start = nullptr;
  gpu_event_t transpose_stop = nullptr;
  GPU_CHECK(GPU_EVENT_CREATE(&transpose_start));
  GPU_CHECK(GPU_EVENT_CREATE(&transpose_stop));
  GPU_CHECK(GPU_EVENT_RECORD(transpose_start, 0));
  {
    dim3 grid_dim_T((words_per_bitset + GPU_TILE_DIM - 1) / GPU_TILE_DIM,
                    ((unsigned int)num_refs + GPU_TILE_DIM - 1) / GPU_TILE_DIM, 1);
    dim3 block_dim_T(GPU_TILE_DIM, GPU_BLOCK_ROWS, 1);
    transpose_kernel<T><<<grid_dim_T, block_dim_T>>>(
        d_refs, d_refs_T, static_cast<int>(words_per_bitset),
        static_cast<int>(num_refs));
  }
  GPU_CHECK(GPU_EVENT_RECORD(transpose_stop, 0));
  GPU_CHECK(GPU_EVENT_SYNC(transpose_stop));
  float transpose_ms = 0.0f;
  GPU_CHECK(
      GPU_EVENT_ELAPSED_TIME(&transpose_ms, transpose_start, transpose_stop));
  result.transpose_time_ns = transpose_ms * 1.0e6;
  result.per_iter_transpose_ms.push_back(transpose_ms);

  GPU_CHECK(GPU_EVENT_DESTROY(transpose_start));
  GPU_CHECK(GPU_EVENT_DESTROY(transpose_stop));

  GPU_CHECK(GPU_MEMCPY(d_queries, h_queries, queries_bytes, GPU_MEMCPY_H2D));
  run_setop_gpu<T>(d_queries, d_refs_T, d_results,
                   static_cast<int>(num_queries), static_cast<int>(num_refs),
                   static_cast<int>(words_per_bitset));
  uint32_t *burnin_results = (uint32_t *)malloc(results_bytes);
  assert(burnin_results);
  GPU_CHECK(
      GPU_MEMCPY(burnin_results, d_results, results_bytes, GPU_MEMCPY_D2H));
  free(burnin_results);
  puts("Completed burn-in iteration to warm up GPU and PCIe paths");

  char *csv_path = nullptr;
  FILE *csv_file = nullptr;
  if (csv_path_is_set(&csv_path)) {
    csv_file = open_csv_output(csv_path);
    if (!csv_file) {
      fprintf(stderr,
              "Warning: CSV output requested but file could not be opened.\n");
    }
  }

  std::vector<double> kernel_times;
  std::vector<double> total_times;
  std::vector<double> cpu_overhead_times;
  std::vector<double> d2h_times;
  std::vector<uint32_t> iteration_results;
  kernel_times.reserve(gpu_iterations);
  total_times.reserve(gpu_iterations);
  cpu_overhead_times.reserve(gpu_iterations);
  d2h_times.reserve(gpu_iterations);
  iteration_results.reserve(gpu_iterations);

  result.gpu_results = (uint32_t *)malloc(results_bytes);
  assert(result.gpu_results);
  memset(result.gpu_results, 0, results_bytes);

  for (int repeat = 0; repeat < gpu_iterations; ++repeat) {
    int64_t total_start = time_ns();
    GPU_CHECK(GPU_MEMCPY(d_queries, h_queries, queries_bytes, GPU_MEMCPY_H2D));
    gpu_event_t kernel_start = nullptr;
    gpu_event_t kernel_stop = nullptr;
    GPU_CHECK(GPU_EVENT_CREATE(&kernel_start));
    GPU_CHECK(GPU_EVENT_CREATE(&kernel_stop));
    GPU_CHECK(GPU_EVENT_RECORD(kernel_start, 0));
    run_setop_gpu<T>(d_queries, d_refs_T, d_results,
                     static_cast<int>(num_queries), static_cast<int>(num_refs),
                     static_cast<int>(words_per_bitset));
    GPU_CHECK(GPU_EVENT_RECORD(kernel_stop, 0));
    GPU_CHECK(GPU_EVENT_SYNC(kernel_stop));
    int64_t d2h_start = time_ns();
    GPU_CHECK(GPU_MEMCPY(result.gpu_results, d_results, results_bytes,
                         GPU_MEMCPY_D2H));
    int64_t d2h_end = time_ns();
    int64_t cpu_scan_start = time_ns();
    uint32_t max_val = 0;
    uint32_t current = 0;
    const size_t nelem = num_queries * num_refs;
    for (size_t idx = 0; idx < nelem; ++idx) {
      current = result.gpu_results[idx];
      if (current > max_val) {
        max_val = current;
      }
    }
    int64_t cpu_scan_end = time_ns();
    int64_t total_end = time_ns();

    float kernel_ms = 0.0f;
    GPU_CHECK(GPU_EVENT_ELAPSED_TIME(&kernel_ms, kernel_start, kernel_stop));
    GPU_CHECK(GPU_EVENT_DESTROY(kernel_start));
    GPU_CHECK(GPU_EVENT_DESTROY(kernel_stop));

    const double total_ns = (double)(total_end - total_start);
    const double cpu_ns = (double)(cpu_scan_end - cpu_scan_start);
    const double d2h_ns = (double)(d2h_end - d2h_start);
    total_times.push_back(total_ns / 1.0e6);
    kernel_times.push_back(kernel_ms);
    cpu_overhead_times.push_back(cpu_ns / 1.0e6);
    d2h_times.push_back(d2h_ns / 1.0e6);
    iteration_results.push_back((uint32_t)max_val);

    if (csv_file) {
      const unsigned long long kernel_ns =
          (unsigned long long)(kernel_ms * 1.0e6 + 0.5);
      const unsigned long long total_ns_ull =
          (unsigned long long)(total_ns + 0.5);
      const unsigned long long cpu_ns_ull = (unsigned long long)(cpu_ns + 0.5);
      const unsigned long long pcie_ns =
          (total_ns_ull > kernel_ns + cpu_ns_ull)
              ? (total_ns_ull - kernel_ns - cpu_ns_ull)
              : 0ULL;
      write_csv_row(csv_file, GPU_TILE_J, GPU_ILP, bitset_bits, num_queries, num_refs,
                    repeat + 1, "kernel", kernel_ns);
      write_csv_row(csv_file, GPU_TILE_J, GPU_ILP, bitset_bits, num_queries, num_refs,
                    repeat + 1, "total", total_ns_ull);
      write_csv_row(csv_file, GPU_TILE_J, GPU_ILP, bitset_bits, num_queries, num_refs,
                    repeat + 1, "PCIe", pcie_ns);
      fflush(csv_file);
    }
  }

  double kernel_sum_ms = 0.0;
  double total_sum_ms = 0.0;
  for (int i = 0; i < gpu_iterations; ++i) {
    kernel_sum_ms += kernel_times[i];
    total_sum_ms += total_times[i];
  }
  const double kernel_avg_ms = kernel_sum_ms / gpu_iterations;
  const double total_avg_ms = total_sum_ms / gpu_iterations;
  const double cpu_avg_ms = (std::accumulate(cpu_overhead_times.begin(),
                                             cpu_overhead_times.end(), 0.0) /
                             gpu_iterations);

  result.per_iter_kernel_ms = std::move(kernel_times);
  result.per_iter_total_ms = std::move(total_times);
  result.per_iter_cpu_overhead_ms = std::move(cpu_overhead_times);
  result.per_iter_d2h_ms = std::move(d2h_times);
  result.per_iter_results = std::move(iteration_results);
  result.per_iter_transpose_ms.resize(1, transpose_ms);

  const double payload_per_iter_gb =
      (double)queries_bytes + (double)results_bytes;
  const double payload_per_iter_gb_norm =
      payload_per_iter_gb / (1024.0 * 1024.0 * 1024.0);

  result.num_pairs = num_queries * num_refs;
  result.kernel_time_ns = kernel_avg_ms * 1.0e6;
  result.total_time_ns = total_avg_ms * 1.0e6;
  result.cpu_overhead_ns = cpu_avg_ms * 1.0e6;
  result.compute_gbps = payload_per_iter_gb_norm / (kernel_avg_ms / 1e3);
  result.total_gbps = payload_per_iter_gb_norm / (total_avg_ms / 1e3);
  if (!result.per_iter_results.empty()) {
    result.max_result = *std::max_element(result.per_iter_results.begin(),
                                          result.per_iter_results.end());
  }
  result.checksum = 0;
  result.agreements = 0;
  result.disagreements = 0;
  if (cpu_results) {
    compare_gpu_to_cpu_results_wrapper(
        result.gpu_results, cpu_results, num_queries, num_refs,
        &result.agreements, &result.disagreements, &result.max_result);
  }
  for (size_t idx = 0; idx < num_queries * num_refs; ++idx) {
    result.checksum += result.gpu_results[idx];
  }

  if (csv_file) {
    fclose(csv_file);
  }

  GPU_CHECK(GPU_FREE(d_results));
  GPU_CHECK(GPU_FREE(d_refs_T));
  GPU_CHECK(GPU_FREE(d_refs));
  GPU_CHECK(GPU_FREE(d_queries));

  return result;
}

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 7) {
    fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference "
            "bitsets> <gpu_iterations> [<gpu_id> [<word_bits>]]\n",
            argv[0]);
    fprintf(stderr, "Example: %s 1024 1000 1000000 10 0 64\n", argv[0]);
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
  unsigned int word_bits = 64u;
  int arg_idx = 5;
  if (arg_idx < argc) {
    if (parse_word_bits(argv[arg_idx], &word_bits)) {
      arg_idx++;
    } else {
      gpu_id = atoi(argv[arg_idx]);
      arg_idx++;
    }
  }
  if (arg_idx < argc) {
    if (parse_word_bits(argv[arg_idx], &word_bits)) {
      arg_idx++;
    } else {
      fprintf(stderr, "Invalid argument: %s\n", argv[arg_idx]);
      return EXIT_FAILURE;
    }
  }
  if (arg_idx != argc) {
    fprintf(stderr, "Too many arguments.\n");
    return EXIT_FAILURE;
  }

  if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 ||
      gpu_iterations <= 0) {
    fprintf(stderr, "Error: size, number of bits, number of ref bits, and GPU "
                    "iterations must be positive integers.\n");
    return EXIT_FAILURE;
  }
  if (gpu_iterations > 1024) {
    fprintf(stderr, "Warning: gpu_iterations capped to 1024\n");
    gpu_iterations = 1024;
  }
  if (size < 128) {
    fprintf(stderr, "Warning: size increased to 128\n");
    size = 128;
  }

  GPU_CHECK(GPU_SET_DEVICE(gpu_id));
  printf("Starting GPU-only benchmark\n");
  printf("GPU_TILE_J: %d, GPU_ILP: %d, word_bits: %u\n", GPU_TILE_J, GPU_ILP, word_bits);

  const size_t words_per_bitset = (size_t)(size + word_bits - 1) / word_bits;
  const size_t query_words = words_per_bitset * (size_t)num_of_bits;
  const size_t ref_words = words_per_bitset * (size_t)num_of_ref_bits;

  if (word_bits == 32u) {
    std::mt19937_64 rng(static_cast<unsigned long>(std::time(nullptr)));
    uint32_t *h_queries = (uint32_t *)calloc(query_words, sizeof(uint32_t));
    uint32_t *h_refs = (uint32_t *)calloc(ref_words, sizeof(uint32_t));
    uint32_t *cpu_results = (uint32_t *)calloc(
        (size_t)num_of_bits * (size_t)num_of_ref_bits, sizeof(uint32_t));
    assert(h_queries && h_refs && cpu_results);

    fill_random_bitsets<uint32_t>(h_queries, (size_t)num_of_bits,
                                  words_per_bitset, rng);
    fill_random_bitsets<uint32_t>(h_refs, (size_t)num_of_ref_bits,
                                  words_per_bitset, rng);
    compute_cpu_popcount_reference_32bit(h_queries, h_refs, (size_t)size,
                                         (size_t)num_of_bits,
                                         (size_t)num_of_ref_bits, cpu_results);

    NativeBenchmarkResult result = benchmark_native_gpu<uint32_t>(
        h_queries, h_refs, cpu_results, (size_t)size, (size_t)num_of_bits,
        (size_t)num_of_ref_bits, gpu_iterations, gpu_id, word_bits);

    size_t agreements = result.agreements;
    size_t disagreements = result.disagreements;
    char test_name[64];
    snprintf(test_name, sizeof(test_name), "Container - GPU - %s",
             BACKEND_NAME);

    puts("GPU Algorithm Timing:");
    for (int i = 0; i < gpu_iterations; ++i) {
      summarize_results(test_name, ms_to_ns(result.per_iter_kernel_ms[i]),
                        i + 1, result.per_iter_results[i],
                        static_cast<float>(result.per_iter_kernel_ms[0] /
                                           result.per_iter_kernel_ms[i]));
    }
    puts("GPU Algorithm + PCIe Timings:");
    for (int i = 0; i < gpu_iterations; ++i) {
      summarize_results(test_name, ms_to_ns(result.per_iter_total_ms[i]), i + 1,
                        result.per_iter_results[i],
                        static_cast<float>(result.per_iter_total_ms[0] /
                                           result.per_iter_total_ms[i]));
    }
    puts("CPU Overhead Timings:");
    for (int i = 0; i < gpu_iterations; ++i) {
      summarize_results(test_name, ms_to_ns(result.per_iter_cpu_overhead_ms[i]),
                        i + 1, result.per_iter_results[i],
                        static_cast<float>(result.per_iter_cpu_overhead_ms[0] /
                                           result.per_iter_cpu_overhead_ms[i]));
    }
    puts("Transpose Timing:");
    printf("  transpose: %.3f ms\n", result.per_iter_transpose_ms[0]);
    puts("\nPer-Iteration Data Movement Breakdown:");
    printf("  db1 (queries) uploaded:   %.6lf GB\n",
           (double)words_per_bitset * 4.0 * num_of_bits /
               (1024.0 * 1024.0 * 1024.0));
    printf("  results downloaded:       %.6lf GB\n",
           (double)num_of_bits * num_of_ref_bits * sizeof(int) /
               (1024.0 * 1024.0 * 1024.0));
    printf("  db2 (reference, resident): %.6lf GB (NOT transferred)\n",
           (double)words_per_bitset * 4.0 * num_of_ref_bits /
               (1024.0 * 1024.0 * 1024.0));
    printf("  Total per-iteration:      %.6lf GB\n",
           ((double)words_per_bitset * 4.0 * num_of_bits +
            (double)num_of_bits * num_of_ref_bits * sizeof(int)) /
               (1024.0 * 1024.0 * 1024.0));
    const double payload_per_iteration_gb =
        ((double)words_per_bitset * 4.0 * num_of_bits +
         (double)num_of_bits * num_of_ref_bits * sizeof(int)) /
        (1024.0 * 1024.0 * 1024.0);
    std::vector<double> kernel_ns(gpu_iterations);
    std::vector<double> total_ns(gpu_iterations);
    std::vector<double> compute_gbps(gpu_iterations);
    std::vector<double> total_gbps(gpu_iterations);
    for (int i = 0; i < gpu_iterations; ++i) {
      kernel_ns[i] = result.per_iter_kernel_ms[i] * 1.0e6;
      total_ns[i] = result.per_iter_total_ms[i] * 1.0e6;
      compute_gbps[i] = payload_per_iteration_gb / (kernel_ns[i] / 1e9);
      total_gbps[i] = payload_per_iteration_gb / (total_ns[i] / 1e9);
    }
    double avg_kernel_ns = 0.0;
    double stddev_kernel_ns = 0.0;
    compute_mean_stddev(kernel_ns.data(), gpu_iterations, &avg_kernel_ns,
                        &stddev_kernel_ns);
    double avg_compute_gbps = 0.0;
    double stddev_compute_gbps = 0.0;
    compute_mean_stddev(compute_gbps.data(), gpu_iterations, &avg_compute_gbps,
                        &stddev_compute_gbps);
    double avg_total_ns = 0.0;
    double stddev_total_ns = 0.0;
    compute_mean_stddev(total_ns.data(), gpu_iterations, &avg_total_ns,
                        &stddev_total_ns);
    double avg_total_gbps = 0.0;
    double stddev_total_gbps = 0.0;
    compute_mean_stddev(total_gbps.data(), gpu_iterations, &avg_total_gbps,
                        &stddev_total_gbps);
    puts("\nEstimated Throughput (iterations 1-N, steady-state):");
    printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n", avg_kernel_ns,
           stddev_kernel_ns);
    printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
           avg_compute_gbps, stddev_compute_gbps);
    printf("Total operation time: mean=%.3f ns, stddev=%.3f ns\n", avg_total_ns,
           stddev_total_ns);
    printf("Total operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
           avg_total_gbps, stddev_total_gbps);
    printf("%s,backend=%s,method=%s,bitset_bits=%d,nelem=%d,iterations=%d,avg_"
           "ns=%.3f,stddev_ns=%.3f,gbps=%.6f,gbps_stddev=%.6f,max=%u\n",
           BACKEND_SUMMARY_LABEL, BACKEND_NAME, POPCOUNT_METHOD_LABEL, size,
           num_of_bits, gpu_iterations, avg_kernel_ns, stddev_kernel_ns,
           avg_compute_gbps, stddev_compute_gbps, result.max_result);
    printf("GPU/CPU agreement: %zu, disagreement: %zu\n", agreements,
           disagreements);
    if (disagreements > 0) {
      return EXIT_FAILURE;
    }
    free(result.gpu_results);
    free(h_queries);
    free(h_refs);
    free(cpu_results);
    return EXIT_SUCCESS;
  }

  std::mt19937_64 rng(static_cast<unsigned long>(std::time(nullptr)));
  uint64_t *h_queries = (uint64_t *)calloc(query_words, sizeof(uint64_t));
  uint64_t *h_refs = (uint64_t *)calloc(ref_words, sizeof(uint64_t));
  uint32_t *cpu_results = (uint32_t *)calloc(
      (size_t)num_of_bits * (size_t)num_of_ref_bits, sizeof(uint32_t));
  assert(h_queries && h_refs && cpu_results);

  fill_random_bitsets<uint64_t>(h_queries, (size_t)num_of_bits,
                                words_per_bitset, rng);
  fill_random_bitsets<uint64_t>(h_refs, (size_t)num_of_ref_bits,
                                words_per_bitset, rng);
  compute_cpu_popcount_reference(h_queries, h_refs, (size_t)size,
                                 (size_t)num_of_bits, (size_t)num_of_ref_bits,
                                 cpu_results);

  NativeBenchmarkResult result = benchmark_native_gpu<uint64_t>(
      h_queries, h_refs, cpu_results, (size_t)size, (size_t)num_of_bits,
      (size_t)num_of_ref_bits, gpu_iterations, gpu_id, word_bits);

  size_t agreements = result.agreements;
  size_t disagreements = result.disagreements;
  char test_name[64];
  snprintf(test_name, sizeof(test_name), "Container - GPU - %s", BACKEND_NAME);

  puts("GPU Algorithm Timing:");
  for (int i = 0; i < gpu_iterations; ++i) {
    summarize_results(test_name, ms_to_ns(result.per_iter_kernel_ms[i]), i + 1,
                      result.per_iter_results[i],
                      static_cast<float>(result.per_iter_kernel_ms[0] /
                                         result.per_iter_kernel_ms[i]));
  }
  puts("GPU Algorithm + PCIe Timings:");
  for (int i = 0; i < gpu_iterations; ++i) {
    summarize_results(test_name, ms_to_ns(result.per_iter_total_ms[i]), i + 1,
                      result.per_iter_results[i],
                      static_cast<float>(result.per_iter_total_ms[0] /
                                         result.per_iter_total_ms[i]));
  }
  puts("CPU Overhead Timings:");
  for (int i = 0; i < gpu_iterations; ++i) {
    summarize_results(test_name, ms_to_ns(result.per_iter_cpu_overhead_ms[i]),
                      i + 1, result.per_iter_results[i],
                      static_cast<float>(result.per_iter_cpu_overhead_ms[0] /
                                         result.per_iter_cpu_overhead_ms[i]));
  }
  puts("Transpose Timing:");
  printf("  transpose: %.3f ms\n", result.per_iter_transpose_ms[0]);
  puts("\nPer-Iteration Data Movement Breakdown:");
  printf("  db1 (queries) uploaded:   %.6lf GB\n",
         (double)words_per_bitset * 8.0 * num_of_bits /
             (1024.0 * 1024.0 * 1024.0));
  printf("  results downloaded:       %.6lf GB\n",
         (double)num_of_bits * num_of_ref_bits * sizeof(int) /
             (1024.0 * 1024.0 * 1024.0));
  printf("  db2 (reference, resident): %.6lf GB (NOT transferred)\n",
         (double)words_per_bitset * 8.0 * num_of_ref_bits /
             (1024.0 * 1024.0 * 1024.0));
  printf("  Total per-iteration:      %.6lf GB\n",
         ((double)words_per_bitset * 8.0 * num_of_bits +
          (double)num_of_bits * num_of_ref_bits * sizeof(int)) /
             (1024.0 * 1024.0 * 1024.0));
  const double payload_per_iteration_gb =
      ((double)words_per_bitset * 8.0 * num_of_bits +
       (double)num_of_bits * num_of_ref_bits * sizeof(int)) /
      (1024.0 * 1024.0 * 1024.0);
  std::vector<double> kernel_ns(gpu_iterations);
  std::vector<double> total_ns(gpu_iterations);
  std::vector<double> compute_gbps(gpu_iterations);
  std::vector<double> total_gbps(gpu_iterations);
  for (int i = 0; i < gpu_iterations; ++i) {
    kernel_ns[i] = result.per_iter_kernel_ms[i] * 1.0e6;
    total_ns[i] = result.per_iter_total_ms[i] * 1.0e6;
    compute_gbps[i] = payload_per_iteration_gb / (kernel_ns[i] / 1e9);
    total_gbps[i] = payload_per_iteration_gb / (total_ns[i] / 1e9);
  }
  double avg_kernel_ns = 0.0;
  double stddev_kernel_ns = 0.0;
  compute_mean_stddev(kernel_ns.data(), gpu_iterations, &avg_kernel_ns,
                      &stddev_kernel_ns);
  double avg_compute_gbps = 0.0;
  double stddev_compute_gbps = 0.0;
  compute_mean_stddev(compute_gbps.data(), gpu_iterations, &avg_compute_gbps,
                      &stddev_compute_gbps);
  double avg_total_ns = 0.0;
  double stddev_total_ns = 0.0;
  compute_mean_stddev(total_ns.data(), gpu_iterations, &avg_total_ns,
                      &stddev_total_ns);
  double avg_total_gbps = 0.0;
  double stddev_total_gbps = 0.0;
  compute_mean_stddev(total_gbps.data(), gpu_iterations, &avg_total_gbps,
                      &stddev_total_gbps);
  puts("\nEstimated Throughput (iterations 1-N, steady-state):");
  printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n", avg_kernel_ns,
         stddev_kernel_ns);
  printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
         avg_compute_gbps, stddev_compute_gbps);
  printf("Total operation time: mean=%.3f ns, stddev=%.3f ns\n", avg_total_ns,
         stddev_total_ns);
  printf("Total operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n",
         avg_total_gbps, stddev_total_gbps);
  printf("%s,backend=%s,method=%s,bitset_bits=%d,nelem=%d,iterations=%d,avg_ns="
         "%.3f,stddev_ns=%.3f,gbps=%.6f,gbps_stddev=%.6f,max=%u\n",
         BACKEND_SUMMARY_LABEL, BACKEND_NAME, POPCOUNT_METHOD_LABEL, size,
         num_of_bits, gpu_iterations, avg_kernel_ns, stddev_kernel_ns,
         avg_compute_gbps, stddev_compute_gbps, result.max_result);
  printf("GPU/CPU agreement: %zu, disagreement: %zu\n", agreements,
         disagreements);
  if (disagreements > 0) {
    free(result.gpu_results);
    free(h_queries);
    free(h_refs);
    free(cpu_results);
    return EXIT_FAILURE;
  }

  free(result.gpu_results);
  free(h_queries);
  free(h_refs);
  free(cpu_results);
  return EXIT_SUCCESS;
}