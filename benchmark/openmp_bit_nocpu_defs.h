// types and definitions for the OpenMP GPU benchmark without CPU overhead measurement
#pragma once
#include <stdbool.h> // For bool type (is_Bit_T_allocated)
#include <stdint.h>  // For uint64_t
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
  unsigned int length;        // capacity of the bitset in bits
  unsigned int size_in_bytes; // number of bytes of the 8 bit container
  unsigned int size_in_qwords; // number of qwords of the 64 bit container
  unsigned char *bytes;       // pointer to the first byte
  uint64_t *qwords;           // pointer to the first qword
  bool is_Bit_T_allocated;    // true if allocated by the library
};

// Bitset DB structure
/*
 The ADT provides access to containers of bitsets that pack fixed number of
  the bitset data in a single container for locality of memory access when
  processing large number of bitsets.
*/
struct Bit_DB_T {
  unsigned int nelem;         // number of bitsets in the packed container
  unsigned int length;        // capacity of the bitset in bits
  unsigned int size_in_bytes; // number of bytes of the 8 bit set container
  unsigned int size_in_qwords; // number of qwords of the 64 bit set container
  unsigned char *bytes;       // pointer to the first byte
  uint64_t *qwords;           // pointer to the first qword
  bool is_Bit_T_allocated;    // true if allocated by the library
};

