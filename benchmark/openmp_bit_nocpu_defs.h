// types and definitions for the OpenMP GPU benchmark without CPU overhead measurement
#pragma once
#include <stdbool.h> // For bool type (is_Bit_T_allocated)
#include <time.h>    // For struct timespec

typedef struct {
  struct timespec start_time;
  struct timespec end_time;
  struct timespec start_PCIe_time;
  struct timespec end_PCIe_time;
  struct timespec start_CPU_overhead;
  struct timespec end_CPU_overhead;
  struct timespec start_GPU_transpose_time;
  struct timespec end_GPU_transpose_time;
} GPU_Instrumentation;

struct Bit_T {
  int length;                 // capacity of the bitset in bits
  int size_in_bytes;          // number of bytes of the 8 bit container
  int size_in_qwords;         // number of qwords of the 64 bit container
  bool is_Bit_T_allocated;    // true if allocated by the library
  unsigned char *bytes;       // pointer to the first byte
  unsigned long long *qwords; // pointer to the first qword
};

// Bitset DB structure
/*
 The ADT provides access to containers of bitsets that pack fixed number of
  the bitset data in a single container for locality of memory access when
  processing large number of bitsets.
*/
struct Bit_DB_T {
  int nelem;                  // number of bitsets in the packed container
  int length;                 // capacity of the bitset in bits
  int size_in_bytes;          // number of bytes of the 8 bit set container
  int size_in_qwords;         // number of qwords of the 64 bit set container
  bool is_Bit_T_allocated;    // true if allocated by the library
  unsigned char *bytes;       // pointer to the first byte
  unsigned long long *qwords; // pointer to the first qword
};

