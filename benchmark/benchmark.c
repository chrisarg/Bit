// Benchmarking code for the bit library
#define _POSIX_C_SOURCE 199309L
#include "bit.h"
#include <assert.h>
#include <immintrin.h> // For AVX, AVX2, SSE2 intrinsics
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BPQW (sizeof(unsigned long long) * 8) // bits per qword
#define BPB (sizeof(unsigned char) * 8)       // bits per byte
#define nqwords(len)                                                           \
  ((((len) + BPQW - 1) & (~(BPQW - 1))) / BPQW) // ceil(len/QBPW)

// Benchmarking function type definition
typedef int64_t (*benchmark_func)(int size, int iterations);

int64_t bench_Bit_aset(int size, int iterations);
int64_t bench_Bit_aclear(int size, int iterations);
int64_t bench_Bit_count(int size, int iterations);
int64_t bench_Bit_inter_count(int size, int iterations);
int64_t bench_Bit_inter_count_mem(int size, int iterations);
int64_t bench_Bit_inter(int size, int iterations);
int64_t bench_Bit_and(int size, int iterations);
int64_t bench_Bit_and_SIMD(int size, int iterations);
void summarize_results(char *test, int64_t timeElapsed, int num_of_iterations);
int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p);

int64_t bench_Bit_aset(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  Bit_T bit1 = Bit_new(size);
  int length_of_index = size / 2 >= 2048 ? 2048 : size / 2;
  int *indices = malloc((length_of_index) * sizeof(int));
  for (int i = 0; i < length_of_index; i++) {
    indices[i] = i;
  }
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    Bit_aset(bit1, indices, length_of_index);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  Bit_free(&bit1);
  free(indices);
  return timeElapsed;
}

int64_t bench_Bit_aclear(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  Bit_T bit1 = Bit_new(size);
  int length_of_index = size / 2 >= 2048 ? 2048 : size / 2;
  int *indices = malloc((length_of_index) * sizeof(int));
  for (int i = 0; i < length_of_index; i++) {
    indices[i] = i;
  }
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    Bit_aclear(bit1, indices, length_of_index);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  Bit_free(&bit1);
  free(indices);
  return timeElapsed;
}
int64_t bench_Bit_count(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  volatile int result;
  Bit_T bit1 = Bit_new(size);
  Bit_set(bit1, size / 2, size - 1);
  Bit_bset(bit1, 0);
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    result = Bit_count(bit1);
  }
  
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  Bit_free(&bit1);
  return timeElapsed;
}

int64_t bench_Bit_inter_count(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  volatile int result;
  Bit_T bit1 = Bit_new(size);
  Bit_T bit2 = Bit_new(size);
  Bit_set(bit1, size / 2, size - 1);
  Bit_bset(bit1, 0);
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    result = Bit_inter_count(bit1, bit2);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  Bit_free(&bit1);
  Bit_free(&bit2);
  return timeElapsed;
}
int64_t bench_Bit_inter_count_mem(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  volatile int result;
  Bit_T bit1 = Bit_new(size);
  Bit_T bit2 = Bit_new(size);
  Bit_set(bit1, size / 2, size - 1);
  Bit_bset(bit1, 0);
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    Bit_T bit3 = Bit_inter(bit1, bit2);
    result = Bit_count(bit3);
    Bit_free(&bit3);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  Bit_free(&bit1);
  Bit_free(&bit2);
  return timeElapsed;
}

int64_t bench_Bit_inter(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  Bit_T bit1 = Bit_new(size);
  Bit_T bit2 = Bit_new(size);
  Bit_set(bit1, size / 2, size - 1);
  Bit_bset(bit1, 0);
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    Bit_T bit3 = Bit_inter(bit1, bit2);
    Bit_free(&bit3);
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  return timeElapsed;
}

int64_t bench_Bit_and(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  size_t size_in_qwords = nqwords(size);
  size_t size_in_bytes = size_in_qwords * BPQW / BPB;

  unsigned long long *bit1 = malloc(size_in_bytes);
  unsigned long long *bit2 = malloc(size_in_bytes);

  unsigned long long volatile result;

  // Initialize with some data pattern
  for (size_t i = 0; i < size_in_qwords; i++) {
    bit1[i] = i + 1;
    bit2[i] = ~i;
  }
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for (int i = 0; i < iterations; i++) {
    for (int j = size_in_qwords; --j >= 0;) {
      result = bit1[j] & bit2[j];
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  return timeElapsed;
}

int64_t bench_Bit_and_SIMD(int size, int iterations) {
  struct timespec start_time, end_time;
  int64_t timeElapsed = 0;
  size_t size_in_qwords = nqwords(size);
  size_t size_in_bytes = size_in_qwords * BPQW / BPB;

  unsigned long long *bit1 = malloc(size_in_bytes);
  unsigned long long *bit2 = malloc(size_in_bytes);
  unsigned long long volatile result;
  // Initialize with some data pattern
  for (size_t i = 0; i < size_in_qwords; i++) {
    bit1[i] = i + 1;
    bit2[i] = ~i;
  }
  clock_gettime(CLOCK_MONOTONIC, &start_time);
#if defined(__AVX512__)
  // AVX512 version - process 8 qwords (512 bits) at once
  for (int i = 0; i < iterations; i++) {
    int j = size_in_qwords;
    // Process 8 qwords at a time
    for (; j >= 8; j -= 8) {
      __m512i a = _mm512_loadu_si512((__m512i *)(bit1 + j - 8));
      __m512i b = _mm512_loadu_si512((__m512i *)(bit2 + j - 8));
      volatile __m512i c = _mm512_and_si512(a, b);
    }
    // Handle remaining elements
    for (; j > 0; j--) {
      volatile unsigned long long result = bit1[j - 1] & bit2[j - 1];
    }
  }
#elif defined(__AVX2__)
  // AVX2 version - process 4 qwords (256 bits) at once
  for (int i = 0; i < iterations; i++) {
    int j = size_in_qwords;
    // Process 4 qwords at a time
    for (; j >= 4; j -= 4) {
      __m256i a = _mm256_loadu_si256((__m256i *)(bit1 + j - 4));
      __m256i b = _mm256_loadu_si256((__m256i *)(bit2 + j - 4));
      volatile __m256i c = _mm256_and_si256(a, b);
    }
    // Handle remaining elements
    for (; j > 0; j--) {
      volatile unsigned long long result = bit1[j - 1] & bit2[j - 1];
    }
  }
#elif defined(__SSE2__)
  // SSE2 version - process 2 qwords (128 bits) at once
  for (int i = 0; i < iterations; i++) {
    int j = size_in_qwords;
    // Process 2 qwords at a time
    for (; j >= 2; j -= 2) {
      __m128i a = _mm_loadu_si128((__m128i *)(bit1 + j - 2));
      __m128i b = _mm_loadu_si128((__m128i *)(bit1 + j - 2));
      volatile __m128i c = _mm_and_si128(a, b);
    }
    // Handle remaining elements
    for (; j > 0; j--) {
      volatile unsigned long long result = bit1[j - 1] & bit2[j - 1];
    }
  }
#else
  // Scalar version (fallback)
  volatile unsigned long long result;
  for (int i = 0; i < iterations; i++) {
    for (int j = size_in_qwords; --j >= 0;) {
      result = bit1[j] & bit2[j];
    }
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  timeElapsed = timeDiff(&end_time, &start_time);
  return timeElapsed;
}

int main() {
  int size_array[] = {128,   256,   512,   1024,   2048,   4096,   8192,
                      16384, 32768, 65536, 131072, 262144, 524288, 1048576};
  char *test_array[] = {"Count", "Inter Count", "Inter Count Mem",
                        "Inter", "And",         "And_SIMD",
                        "aset",  "aclear"};
#if defined(__AVX512__)
  printf("AVX512 detected\n");
#elif defined(__AVX2__)
  printf("AVX2 detected\n");
#elif defined(__SSE2__)
  printf("SSE2 detected\n");
#else
  printf("ENIAC detected\n")
#endif

#ifdef BUILTIN_POPCOUNT
  printf("Using builtin popcount\n");
#else
  printf("Using library popcount\n");
#endif
  // Array of benchmark functions
  benchmark_func benchmark_funcs[] = {
      bench_Bit_count,       bench_Bit_inter_count, bench_Bit_inter_count_mem,
      bench_Bit_inter_count, bench_Bit_and,         bench_Bit_and_SIMD,
      bench_Bit_aset,        bench_Bit_aclear,
  };
  char *test_explantion[] = {
      "Count the number of bits set in the bitset",
      "Count the number of bits set in an intersection",
      ("Count the number of bits set in the intersection by first\n"
       "\tforming the intersection and then counting"),
      "Intersection of two bitsets",
      "Bitwise AND of two buffers",
      "Bitwise AND of two buffers using SIMD intrinsics",
      "Set an array of bits (up to 2048) in the bitset",
      "Clear an array of bits (up to 2048) in the bitset",
  };
  // print an one line summary of the tests
  printf("Benchmarking the bit library\n");
  for (size_t i = 0; i < sizeof(test_array) / sizeof(char *); i++) {
    printf("%s => %s\n", test_array[i], test_explantion[i]);
  }
  int iterations = 1000;
  int64_t timeElapsed;
  char s[50];
  for (size_t j = 0; j < sizeof(test_array) / sizeof(char *); j++) {
    for (size_t i = 0; i < sizeof(size_array) / sizeof(int); i++) {
      timeElapsed = benchmark_funcs[j](size_array[i], iterations);
      snprintf(s, sizeof(s), "Bit %15s (size = %10d)", test_array[j],
               size_array[i]);
      summarize_results(s, timeElapsed, iterations);
    }
  }

  return 0;
}

void summarize_results(char *test, int64_t timeElapsed, int num_of_iterations) {
  printf("Total time for %20s: %15ld ns\t", test, timeElapsed);
  printf("Time per iteration: %10.2lf ns\t",
         (double)timeElapsed / num_of_iterations);
  printf("Iterations per second %10.2lg\n",
         (double)num_of_iterations * 1e9 / timeElapsed);
}

int64_t timeDiff(struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec - timeB_p->tv_sec) * 1000000000 + timeA_p->tv_nsec -
          timeB_p->tv_nsec);
}
