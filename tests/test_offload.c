#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FLOAT_TOL 1e-5f
#define DOUBLE_TOL 1e-12

static void print_usage(const char* prog) {
  fprintf(stderr,
    "Usage: %s <problem_size> [device_id]\n"
    "\n"
    "Parameters:\n"
    "  problem_size : positive integer number of elements per array (required)\n"
    "  device_id    : OpenMP target device index to test (optional)\n"
    "                 if omitted, the OpenMP default device is used\n"
    "\n"
    "Examples:\n"
    "  %s 4096\n"
    "  %s 100000 0\n",
    prog, prog, prog);
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

static bool verify_target_device(int device_id) {
  int is_initial = 1;
#pragma omp target map(from: is_initial) device(device_id)
  { is_initial = omp_is_initial_device(); }

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

int main(int argc, char* argv []) {
  if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }

  if (argc != 2 && argc != 3) {
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

  int num_devices = omp_get_num_devices();
  printf("OpenMP target devices found: %d\n", num_devices);
  printf("Requested problem size: %d\n", n);

  if (num_devices <= 0) {
    fprintf(stderr, "FAIL: No OpenMP target devices available.\n");
    return EXIT_FAILURE;
  }

  int device_id;
  if (requested_device_id >= 0) {
    if (requested_device_id >= num_devices) {
      fprintf(stderr,
        "FAIL: requested device_id=%d is out of range [0, %d].\n",
        requested_device_id, num_devices - 1);
      return EXIT_FAILURE;
    }
    device_id = requested_device_id;
    printf("Using user-specified device_id: %d\n", device_id);
  }
  else {
    device_id = omp_get_default_device();
    if (device_id < 0 || device_id >= num_devices) {
      printf("OpenMP default device %d is invalid; falling back to device 0.\n",
        device_id);
      device_id = 0;
    }
    printf("Using OpenMP default device_id: %d\n", device_id);
  }

  if (!verify_target_device(device_id)) {
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
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
