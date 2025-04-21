# Bit - A High-Performance Bitset Library

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-BSD%202--Clause-blue)]()

Bit is a high-performance, uncompressed bitset implementation in C, optimized
for modern architectures. The library provides an efficient way to create,
manipulate, and query bitsets with a focus on performance and memory alignment.
The API and the interface is largely based on David Hanson's Bit_T library 
discussed in Chapter 13 of "C Interfaces and Implementations", Addison-Wesley
ISBN 0-201-49841-3 extended to incorporate additional operations (such as counts
on unions/differences/intersections of sets) and fast population counts (see below)

## Features

- **Optimized for Modern Hardware**: Automatic alignment detection for different architectures (32/64-bit)
- **SIMD Acceleration**: Leverages AVX, AVX2, and AVX512 instructions where
  available for set operations (Under Development)
- **Specialized Population Counting**: Integration of the libpopcnt library 
  offering  multiple implementations of bit counting algorithms ("population
  counts", aka _popcount_ aka _popcnt_) including:
  - Hardware POPCNT
  - Wilkes-Wheeler-Gill algorithm
  - SIMD-accelerated counting
- **Set Operations**: Union, intersection, difference, and symmetric difference
- **Memory Efficiency**: Precise memory management with aligned allocations
- **Comprehensive API**: Based on David Hanson's "C Interfaces and Implementations" design
- **Thread-Safety**: No global state, all operations are reentrant
- **Loading of bitset buffers**: This will facilitate integration and coupling
  with scripting languages (under development)
- **Perl interface**: Both object oriented and functional (under development)

## Installation

### Prerequisites

- C compiler (GCC, Clang, or Intel ICC)
- GNU Make

### Building

```bash
# Clone the repository
git clone https://github.com/username/Bit.git
cd Bit

# Build the library
make

# Build with libpopcnt integration (for faster population counting)
make LIBPOPCNT=1

# Run tests
make test

# Run benchmarks
make bench
```

## Usage Example

```c
#include "bit.h"
#include <stdio.h>

int main() {
    // Create a bitset with 1024 bits
    Bit_T bitset1 = Bit_new(1024);
    Bit_T bitset2 = Bit_new(1024);
    
    // Set some bits
    Bit_bset(bitset1, 42);
    Bit_bset(bitset1, 100);
    Bit_bset(bitset2, 42);
    Bit_bset(bitset2, 200);
    
    // Calculate intersection count
    int count = Bit_inter_count(bitset1, bitset2);
    printf("Intersection count: %d\n", count);  // Output: 1
    
    // Create intersection bitset
    Bit_T intersection = Bit_inter(bitset1, bitset2);
    
    // Clean up
    Bit_free(bitset1);
    Bit_free(bitset2);
    Bit_free(intersection);
    
    return 0;
}
```

## API Overview

### Creation and Destruction

```c
Bit_T Bit_new(int length);
void Bit_free(T set);
void Bit_free_safe(T *set);
```

### Bitset Properties

```c
int Bit_length(T set);
int Bit_count(T set);
int size_of_Bit_T(void);
int Bit_size(int length);
int Bit_buffer_size(int length);
```

### Bit Manipulation

```c
void Bit_bset(T set, int index);
void Bit_bclear(T set, int index);
void Bit_clear(T set, int lo, int hi);
int Bit_get(T set, int index);
void Bit_map(T set, void apply(int n, int bit, void *cl), void *cl);
void Bit_not(T set, int lo, int hi);
int Bit_put(T set, int n, int bit);
void Bit_set(T set, int lo, int hi);
```

### Bitset Comparisons

```c
int Bit_eq(T s, T t);
int Bit_leq(T s, T t);
int Bit_lt(T s, T t);
```

### Set Operations

```c
T Bit_diff(T s, T t);
T Bit_inter(T s, T t);
T Bit_minus(T s, T t);
T Bit_union(T s, T t);

int Bit_diff_count(T s, T t);
int Bit_inter_count(T s, T t);
int Bit_minus_count(T s, T t);
int Bit_union_count(T s, T t);
```

## Libraries Used

This project incorporates or is inspired by several open-source libraries:

- **libpopcnt**: A C/C++ library for counting the number of 1 bits (bit population count) specialized for different CPU architectures. Licensed under BSD 2-Clause.
- **sse-popcount**: The SIMD population count implementation of the Harley-Seal
  algorithm based on the paper "Faster Population Counts using AVX2
  Instructions" by Daniel Lemire, Nathan Kurz and Wojciech Mula.
- **Wilkes-Wheeler-Gill Algorithm**: A highly portable and efficient algorithm
  for counting set bits documented in "The Preparation of Programs for an
  Electronic Digital Computer". Nearly 70 years after it's introduction, this
  scalar algorithm outperforms hardware popcounts (including SSE popcounts and
  hardware scalar popcounts)

## Performance

The library is optimized for performance, with specialized implementations of
the population counts for different CPU architectures:

- **AVX512**: Utilizes 512-bit vector operations for maximum throughput
- **AVX2**: Uses 256-bit vector operations on supported CPUs
- **NEON**: Falls back to 128-bit vector operations on older CPUs
- **SVE**
- **Scalar**: Provides optimized scalar implementations for universal compatibility

The benchmarking suite demonstrates significant performance advantages over naive implementations, especially for large bitsets.

## Applications

Bit is particularly useful for:

- Bioinformatics and genomic data processing (k-mer encoding)
- Network packet filtering and bloom filters
- Dense data representation (for sparse bitsets over large domains, one is
  probably better off exploring sparse representations e.g. roaring bitsets)
- High-performance set operations

## License

BSD 2-Clause License. See the LICENSE file for details.

## Author

Christos Argyropoulos (April 2025)