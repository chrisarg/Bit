#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FLOAT_TOL 1e-5f
#define DOUBLE_TOL 1e-12

static void print_usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s <problem_size> [device_id] [benchmark_iterations]\n"
    "\n"
    "Parameters:\n"
    "  problem_size : positive integer number of elements per array (required)\n"
    "  device_id    : OpenMP target device index to test (optional)\n"
    "                 if omitted, the OpenMP default device is used\n"
    "  benchmark_iterations : run long GPU benchmarks for this many iterations\n"
    "                         after correctness checks (optional, default: 0)\n"
    "\n"
    "Examples:\n"
    "  %s 4096\n"
    "  %s 100000 0\n"
    "  %s 1000000 0 500\n",
    prog, prog, prog, prog);
}

static bool parse_positive_int(const char* text, int* out_value) {
  char* endptr = NULL;
  long value = strtol(text, &endptr, 10);
  if (*text == '\0' || *endptr != '\0') {
    return false;
  }
  if (value <= 0 || value > INT_MAX) {
    return false;
  }
  *out_value = (int)value;
  return true;
}

static bool parse_non_negative_int(const char* text, int* out_value) {
  char* endptr = NULL;
  long value = strtol(text, &endptr, 10);
  if (*text == '\0' || *endptr != '\0') {
    return false;
  }
  if (value < 0 || value > INT_MAX) {
    return false;
  }
  *out_value = (int)value;
  return true;
}

static int probe_target_device_initial_state(int device_id) {
  int is_initial = 1;
#pragma omp target map(from: is_initial) device(device_id)
  { is_initial = omp_is_initial_device(); }
  return is_initial;
}

static void print_target_device_diagnostics(int num_devices) {
  int default_device = omp_get_default_device();
  printf("OpenMP target device diagnostics:\n");
  printf("  default_device=%d initial_device=%d\n", default_device,
    omp_get_initial_device());
  for (int device_id = 0; device_id < num_devices; device_id++) {
    int is_initial = probe_target_device_initial_state(device_id);
    printf("  device_id=%d offload=%s", device_id,
      is_initial ? "no (host fallback)" : "yes");
    if (device_id == default_device) {
      printf(" [default]");
    }
    printf("\n");
  }
}

static bool verify_target_device(int device_id) {
  int is_initial = probe_target_device_initial_state(device_id);

  if (is_initial) {
    fprintf(stderr,
      "FAIL: Target region executed on initial device (host fallback).\n");
    return false;
  }

  return true;
}

static bool test_integer_ops(int n, int device_id) {
  int* lhs = malloc((size_t)n * sizeof(int));
  int* rhs = malloc((size_t)n * sizeof(int));
  int* cpu = malloc((size_t)n * sizeof(int));
  int* gpu = malloc((size_t)n * sizeof(int));
  if (!lhs || !rhs || !cpu || !gpu) {
    fprintf(stderr, "FAIL: integer allocation failed\n");
    free(lhs);
    free(rhs);
    free(cpu);
    free(gpu);
    return false;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = i + 3;
    rhs[i] = (i % 17) + 5;
    cpu[i] = ((lhs[i] + rhs[i]) * 3) - (lhs[i] / 2) + (rhs[i] % 7);
  }

#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
  for (int i = 0; i < n; i++) {
    gpu[i] = ((lhs[i] + rhs[i]) * 3) - (lhs[i] / 2) + (rhs[i] % 7);
  }

  bool pass = true;
  for (int i = 0; i < n; i++) {
    if (gpu[i] != cpu[i]) {
      fprintf(stderr, "FAIL: integer mismatch at %d (cpu=%d, gpu=%d)\n", i,
        cpu[i], gpu[i]);
      pass = false;
      break;
    }
  }

  free(lhs);
  free(rhs);
  free(cpu);
  free(gpu);
  return pass;
}

static bool test_float_ops(int n, int device_id) {
  float* lhs = malloc((size_t)n * sizeof(float));
  float* rhs = malloc((size_t)n * sizeof(float));
  float* cpu = malloc((size_t)n * sizeof(float));
  float* gpu = malloc((size_t)n * sizeof(float));
  if (!lhs || !rhs || !cpu || !gpu) {
    fprintf(stderr, "FAIL: float allocation failed\n");
    free(lhs);
    free(rhs);
    free(cpu);
    free(gpu);
    return false;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (float)i * 0.25f + 1.0f;
    rhs[i] = (float)(i % 19) * 0.75f + 2.0f;
    cpu[i] = (lhs[i] * 1.25f) + (rhs[i] * 0.5f) - ((lhs[i] - rhs[i]) * 0.1f) +
      2.0f;
  }

#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
  for (int i = 0; i < n; i++) {
    gpu[i] = (lhs[i] * 1.25f) + (rhs[i] * 0.5f) - ((lhs[i] - rhs[i]) * 0.1f) +
      2.0f;
  }

  bool pass = true;
  for (int i = 0; i < n; i++) {
    if (fabsf(gpu[i] - cpu[i]) > FLOAT_TOL) {
      fprintf(stderr, "FAIL: float mismatch at %d (cpu=%f, gpu=%f)\n", i,
        cpu[i], gpu[i]);
      pass = false;
      break;
    }
  }

  free(lhs);
  free(rhs);
  free(cpu);
  free(gpu);
  return pass;
}

static bool test_double_ops(int n, int device_id) {
  double* lhs = malloc((size_t)n * sizeof(double));
  double* rhs = malloc((size_t)n * sizeof(double));
  double* cpu = malloc((size_t)n * sizeof(double));
  double* gpu = malloc((size_t)n * sizeof(double));
  if (!lhs || !rhs || !cpu || !gpu) {
    fprintf(stderr, "FAIL: double allocation failed\n");
    free(lhs);
    free(rhs);
    free(cpu);
    free(gpu);
    return false;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (double)i * 0.125 + 0.5;
    rhs[i] = (double)(i % 31) * 0.625 + 1.5;
    cpu[i] = (lhs[i] * 1.5) + (rhs[i] * 0.25) - ((lhs[i] + rhs[i]) * 0.05) +
      3.0;
  }

#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
  for (int i = 0; i < n; i++) {
    gpu[i] = (lhs[i] * 1.5) + (rhs[i] * 0.25) - ((lhs[i] + rhs[i]) * 0.05) +
      3.0;
  }

  bool pass = true;
  for (int i = 0; i < n; i++) {
    if (fabs(gpu[i] - cpu[i]) > DOUBLE_TOL) {
      fprintf(stderr, "FAIL: double mismatch at %d (cpu=%lf, gpu=%lf)\n", i,
        cpu[i], gpu[i]);
      pass = false;
      break;
    }
  }

  free(lhs);
  free(rhs);
  free(cpu);
  free(gpu);
  return pass;
}

static void benchmark_integer_ops(int n, int device_id, int iterations) {
  int* lhs = malloc((size_t)n * sizeof(int));
  int* rhs = malloc((size_t)n * sizeof(int));
  int* gpu = malloc((size_t)n * sizeof(int));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: integer benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = i + 3;
    rhs[i] = (i % 17) + 5;
  }

  long long checksum = 0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      gpu[i] = ((lhs[i] + rhs[i]) * 3) - (lhs[i] / 2) + (rhs[i] % 7) +
        (iter & 7);
    }
    checksum += gpu[(iter * 104729) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;

  printf("BENCH [MEMORY-BOUND]: int    iterations=%d elapsed=%.3fs throughput=%.3f Melem/s checksum=%lld\n",
    iterations, elapsed, elements / elapsed / 1e6, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_float_ops(int n, int device_id, int iterations) {
  float* lhs = malloc((size_t)n * sizeof(float));
  float* rhs = malloc((size_t)n * sizeof(float));
  float* gpu = malloc((size_t)n * sizeof(float));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: float benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (float)i * 0.25f + 1.0f;
    rhs[i] = (float)(i % 19) * 0.75f + 2.0f;
  }

  double checksum = 0.0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      gpu[i] = (lhs[i] * 1.25f) + (rhs[i] * 0.5f) - ((lhs[i] - rhs[i]) * 0.1f) +
        2.0f + (float)(iter & 3) * 0.01f;
    }
    checksum += gpu[(iter * 99991) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;

  printf("BENCH [MEMORY-BOUND]: float  iterations=%d elapsed=%.3fs throughput=%.3f Melem/s checksum=%.6f\n",
    iterations, elapsed, elements / elapsed / 1e6, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_double_ops(int n, int device_id, int iterations) {
  double* lhs = malloc((size_t)n * sizeof(double));
  double* rhs = malloc((size_t)n * sizeof(double));
  double* gpu = malloc((size_t)n * sizeof(double));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: double benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (double)i * 0.125 + 0.5;
    rhs[i] = (double)(i % 31) * 0.625 + 1.5;
  }

  double checksum = 0.0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      gpu[i] = (lhs[i] * 1.5) + (rhs[i] * 0.25) - ((lhs[i] + rhs[i]) * 0.05) +
        3.0 + (double)(iter & 3) * 0.001;
    }
    checksum += gpu[(iter * 65537) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;

  printf("BENCH [MEMORY-BOUND]: double iterations=%d elapsed=%.3fs throughput=%.3f Melem/s checksum=%.6f\n",
    iterations, elapsed, elements / elapsed / 1e6, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_compute_intense_ops(int n, int device_id, int iterations) {
  float* lhs = malloc((size_t)n * sizeof(float));
  float* rhs = malloc((size_t)n * sizeof(float));
  float* gpu = malloc((size_t)n * sizeof(float));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: compute-intense benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (float)i * 0.001f + 0.1f;
    rhs[i] = (float)(i % 97) * 0.002f + 0.2f;
  }

  const int inner_steps = 256;
  const double flops_per_step = 10.0;
  const double flops_per_element = inner_steps * flops_per_step;

  double checksum = 0.0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      float x = lhs[i] + (float)(iter & 7) * 0.0001f;
      float y = rhs[i] - (float)(iter & 3) * 0.0002f;
      for (int step = 0; step < inner_steps; step++) {
        x = x * 1.00011921f + y * 0.99988079f + 0.000001f;
        y = y * 1.00005960f - x * 0.00095367f + 0.0000001f;
      }
      gpu[i] = x + y;
    }
    checksum += gpu[(iter * 32771) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;
  double gflops = elements * flops_per_element / elapsed / 1e9;

  printf("BENCH [HYBRID-COMPUTE]: fp32-fma iterations=%d elapsed=%.3fs throughput=%.3f Melem/s est=%.3f GFLOP/s checksum=%.6f\n",
    iterations, elapsed, elements / elapsed / 1e6, gflops, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_device_resident_fp64_ops(int n, int device_id,
  int iterations) {
  double* lhs = malloc((size_t)n * sizeof(double));
  double* rhs = malloc((size_t)n * sizeof(double));
  if (!lhs || !rhs) {
    fprintf(stderr, "FAIL: device-resident fp64 benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (double)i * 0.001 + 0.125;
    rhs[i] = (double)(i % 97) * 0.002 + 0.25;
  }

  const int inner_steps = 512;
  const double flops_per_step = 10.0;
  const double flops_per_element = (double)iterations * inner_steps * flops_per_step;

  double checksum = 0.0;
  double start = omp_get_wtime();
#pragma omp target teams distribute parallel for reduction(+: checksum) map(to: lhs [0:n], rhs [0:n]) device(device_id)
  for (int i = 0; i < n; i++) {
    double x = lhs[i];
    double y = rhs[i];

    for (int iter = 0; iter < iterations; iter++) {
      for (int step = 0; step < inner_steps; step++) {
        x = x * 1.00000011921 + y * 0.99999988079 + 0.000000001;
        y = y * 1.00000005960 - x * 0.0000095367 + 0.0000000001;
      }
    }

    checksum += x + y;
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n;
  double gflops = elements * flops_per_element / elapsed / 1e9;

  printf("BENCH [COMPUTE-BOUND][DEVICE-RESIDENT]: fp64-resident iterations=%d elapsed=%.3fs throughput=%.3f Melem/s est=%.3f GFLOP/s checksum=%.6f\n",
    iterations, elapsed, elements / elapsed / 1e6, gflops, checksum);

  free(lhs);
  free(rhs);
}

static void benchmark_compute_fp64_ops(int n, int device_id, int iterations) {
  double* lhs = malloc((size_t)n * sizeof(double));
  double* rhs = malloc((size_t)n * sizeof(double));
  double* gpu = malloc((size_t)n * sizeof(double));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: fp64 compute benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (double)i * 0.001 + 0.1;
    rhs[i] = (double)(i % 97) * 0.002 + 0.2;
  }

  const int inner_steps = 192;
  const double flops_per_step = 10.0;
  const double flops_per_element = inner_steps * flops_per_step;

  double checksum = 0.0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      double x = lhs[i] + (double)(iter & 7) * 0.0001;
      double y = rhs[i] - (double)(iter & 3) * 0.0002;
      for (int step = 0; step < inner_steps; step++) {
        x = x * 1.00000011921 + y * 0.99999988079 + 0.000000001;
        y = y * 1.00000005960 - x * 0.0000095367 + 0.0000000001;
      }
      gpu[i] = x + y;
    }
    checksum += gpu[(iter * 32771) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;
  double gflops = elements * flops_per_element / elapsed / 1e9;

  printf("BENCH [COMPUTE-BOUND]: fp64-fma iterations=%d elapsed=%.3fs throughput=%.3f Melem/s est=%.3f GFLOP/s checksum=%.6f\n",
    iterations, elapsed, elements / elapsed / 1e6, gflops, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_compute_int64_ops(int n, int device_id, int iterations) {
  int64_t* lhs = malloc((size_t)n * sizeof(int64_t));
  int64_t* rhs = malloc((size_t)n * sizeof(int64_t));
  int64_t* gpu = malloc((size_t)n * sizeof(int64_t));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: int64 compute benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (int64_t)i * 13 + 7;
    rhs[i] = (int64_t)(i % 101) * 17 + 11;
  }

  const int inner_steps = 192;
  const double ops_per_step = 12.0;
  const double ops_per_element = inner_steps * ops_per_step;

  long long checksum = 0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      int64_t x = lhs[i] + (int64_t)(iter & 7);
      int64_t y = rhs[i] - (int64_t)(iter & 3);
      for (int step = 0; step < inner_steps; step++) {
        x = (x * 6364136223846793005LL) + (y ^ 1442695040888963407LL);
        y = (y * 2862933555777941757LL) - (x ^ 3202034522624059733LL);
      }
      gpu[i] = x ^ y;
    }
    checksum += (long long)gpu[(iter * 104729) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;
  double gops = elements * ops_per_element / elapsed / 1e9;

  printf("BENCH [COMPUTE-BOUND]: int64-ops iterations=%d elapsed=%.3fs throughput=%.3f Melem/s est=%.3f GOPS checksum=%lld\n",
    iterations, elapsed, elements / elapsed / 1e6, gops, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_compute_int32_ops(int n, int device_id, int iterations) {
  int32_t* lhs = malloc((size_t)n * sizeof(int32_t));
  int32_t* rhs = malloc((size_t)n * sizeof(int32_t));
  int32_t* gpu = malloc((size_t)n * sizeof(int32_t));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: int32 compute benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (int32_t)(i * 11 + 3);
    rhs[i] = (int32_t)((i % 97) * 19 + 5);
  }

  const int inner_steps = 256;
  const double ops_per_step = 10.0;
  const double ops_per_element = inner_steps * ops_per_step;

  long long checksum = 0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      int32_t x = lhs[i] + (int32_t)(iter & 7);
      int32_t y = rhs[i] - (int32_t)(iter & 3);
      for (int step = 0; step < inner_steps; step++) {
        x = (int32_t)((x * 1664525) + (y ^ 1013904223));
        y = (int32_t)((y * 22695477) - (x ^ 12345));
      }
      gpu[i] = x ^ y;
    }
    checksum += (long long)gpu[(iter * 65537) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;
  double gops = elements * ops_per_element / elapsed / 1e9;

  printf("BENCH [COMPUTE-BOUND]: int32-ops iterations=%d elapsed=%.3fs throughput=%.3f Melem/s est=%.3f GOPS checksum=%lld\n",
    iterations, elapsed, elements / elapsed / 1e6, gops, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void benchmark_compute_int16_ops(int n, int device_id, int iterations) {
  int16_t* lhs = malloc((size_t)n * sizeof(int16_t));
  int16_t* rhs = malloc((size_t)n * sizeof(int16_t));
  int16_t* gpu = malloc((size_t)n * sizeof(int16_t));
  if (!lhs || !rhs || !gpu) {
    fprintf(stderr, "FAIL: int16 compute benchmark allocation failed\n");
    free(lhs);
    free(rhs);
    free(gpu);
    return;
  }

  for (int i = 0; i < n; i++) {
    lhs[i] = (int16_t)(i * 7 + 1);
    rhs[i] = (int16_t)((i % 89) * 5 + 3);
  }

  const int inner_steps = 256;
  const double ops_per_step = 10.0;
  const double ops_per_element = inner_steps * ops_per_step;

  long long checksum = 0;
  double start = omp_get_wtime();
  for (int iter = 0; iter < iterations; iter++) {
#pragma omp target teams distribute parallel for map(to: lhs [0:n], rhs [0:n]) \
  map(from: gpu [0:n]) device(device_id)
    for (int i = 0; i < n; i++) {
      int32_t x = (int32_t)lhs[i] + (int32_t)(iter & 7);
      int32_t y = (int32_t)rhs[i] - (int32_t)(iter & 3);
      for (int step = 0; step < inner_steps; step++) {
        x = (int32_t)((x * 109) + (y ^ 37));
        y = (int32_t)((y * 113) - (x ^ 17));
      }
      gpu[i] = (int16_t)(x ^ y);
    }
    checksum += (long long)gpu[(iter * 40961) % n];
  }
  double elapsed = omp_get_wtime() - start;
  double elements = (double)n * (double)iterations;
  double gops = elements * ops_per_element / elapsed / 1e9;

  printf("BENCH [COMPUTE-BOUND]: int16-ops iterations=%d elapsed=%.3fs throughput=%.3f Melem/s est=%.3f GOPS checksum=%lld\n",
    iterations, elapsed, elements / elapsed / 1e6, gops, checksum);

  free(lhs);
  free(rhs);
  free(gpu);
}

static void run_gpu_benchmarks(int n, int device_id, int benchmark_iterations) {
  printf("Starting long GPU benchmark: n=%d, iterations=%d, device_id=%d\n",
    n, benchmark_iterations, device_id);
  printf("Benchmark classes: [MEMORY-BOUND] moves data each iteration; [HYBRID-COMPUTE] keeps compute-heavy work mixed with transfers; [COMPUTE-BOUND] keeps data resident and returns one checksum.\n");
  benchmark_integer_ops(n, device_id, benchmark_iterations);
  benchmark_float_ops(n, device_id, benchmark_iterations);
  benchmark_double_ops(n, device_id, benchmark_iterations);
  benchmark_compute_intense_ops(n, device_id, benchmark_iterations);
  benchmark_device_resident_fp64_ops(n, device_id, benchmark_iterations);
  benchmark_compute_fp64_ops(n, device_id, benchmark_iterations);
  benchmark_compute_int64_ops(n, device_id, benchmark_iterations);
  benchmark_compute_int32_ops(n, device_id, benchmark_iterations);
  benchmark_compute_int16_ops(n, device_id, benchmark_iterations);
  printf("Completed long GPU benchmark\n");
}

int main(int argc, char* argv []) {
  if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  if (argc != 2 && argc != 3 && argc != 4) {
    fprintf(stderr, "FAIL: invalid number of arguments.\n\n");
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int n = 0;
  if (!parse_positive_int(argv[1], &n)) {
    fprintf(stderr,
      "FAIL: <problem_size> must be a positive integer in [1, %d].\n\n",
      INT_MAX);
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int requested_device_id = -1;
  if (argc == 3 && !parse_non_negative_int(argv[2], &requested_device_id)) {
    fprintf(stderr, "FAIL: [device_id] must be a non-negative integer.\n\n");
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  if (argc == 4) {
    if (!parse_non_negative_int(argv[2], &requested_device_id)) {
      fprintf(stderr, "FAIL: [device_id] must be a non-negative integer.\n\n");
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  int benchmark_iterations = 0;
  if (argc == 4 && !parse_non_negative_int(argv[3], &benchmark_iterations)) {
    fprintf(stderr,
      "FAIL: [benchmark_iterations] must be a non-negative integer.\n\n");
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  int num_devices = omp_get_num_devices();
  printf("OpenMP target devices found: %d\n", num_devices);
  printf("Requested problem size: %d\n", n);
  if (num_devices > 0) {
    print_target_device_diagnostics(num_devices);
  }
  if (num_devices == 0) {
    fprintf(stderr,
      "Hint: no OpenMP target devices were discovered by libomptarget.\n");
    fprintf(stderr,
      "Hint: on systems with multiple NVIDIA GPUs, try limiting visibility ");
    fprintf(stderr,
      "with CUDA_VISIBLE_DEVICES=0 or CUDA_VISIBLE_DEVICES=1 and rerun.\n");
    fprintf(stderr,
      "Hint: for AMD offload, verify AMD_ARCH matches your GPU gfx target ");
    fprintf(stderr,
      "and try ROCR_VISIBLE_DEVICES=0 (or HSA_VISIBLE_DEVICES=0).\n");
  }

  int device_id;
  if (requested_device_id >= 0) {
    if (num_devices > 0 && requested_device_id >= num_devices) {
      fprintf(stderr,
        "FAIL: requested device_id=%d is out of range [0, %d].\n",
        requested_device_id, num_devices - 1);
      return EXIT_FAILURE;
    }
    device_id = requested_device_id;
    printf("Using user-specified device_id: %d\n", device_id);
  }
  else {
    if (num_devices > 0) {
      device_id = omp_get_default_device();
      if (device_id < 0 || device_id >= num_devices) {
        printf(
          "OpenMP default device %d is invalid; falling back to device 0.\n",
          device_id);
        device_id = 0;
      }
      printf("Using OpenMP default device_id: %d\n", device_id);
    }
    else {
      device_id = 0;
      printf("No devices reported by omp_get_num_devices(); probing device_id: 0\n");
    }
  }

  if (!verify_target_device(device_id)) {
    fprintf(stderr,
      "Hint: ensure the binary was built for the correct GPU architecture and "
      "that OpenMP offload runtime plugins are available.\n");
    fprintf(stderr,
      "Hint: try running with OMP_TARGET_OFFLOAD=MANDATORY and "
      "LIBOMPTARGET_INFO=16 for runtime diagnostics.\n");
    return EXIT_FAILURE;
  }

  bool int_ok = test_integer_ops(n, device_id);
  bool float_ok = test_float_ops(n, device_id);
  bool double_ok = test_double_ops(n, device_id);

  printf("%s: integer operations\n", int_ok ? "PASS" : "FAIL");
  printf("%s: float operations\n", float_ok ? "PASS" : "FAIL");
  printf("%s: double operations\n", double_ok ? "PASS" : "FAIL");

  if (int_ok && float_ok && double_ok) {
    printf("PASS: OpenMP target offload arithmetic checks\n");
    if (benchmark_iterations > 0) {
      run_gpu_benchmarks(n, device_id, benchmark_iterations);
    }
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
