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

- C compiler (GCC, Clang, or Intel ICX)
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

# Make the benchmark (this will also build the OpenMP benchmark)
make bench

# Make the open-mp benchmark
make bench_omp
```
### Benchmarking

```bash
# Runs various benchmarks
./build/benchmark  

# OpenMP benchmark
Usage: ./build/openmp_bit <size> <number of reference bitsets> <max threads>
```

The OpenMP benchmark assesses the scaling of searching (intersection count) of a
single bitset of given size(capacity) against a database of reference bitsets. 
The benchmark will run 3 repetitions of a single threaded search, and then run
the same query using OpenMP from 1 to the maximum number of threads to assess
the scaling of the performance. A guided schedule is used internally to schedule
the OpenMP threads.

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
Create, free and free safely (by nullifying the pointer of a bitset). If you are
using this library in a C/C++ project, I strongly suggest you free safely.
Bit_free is intended to be used by the Perl interfaces (both functional and
object oriented) to manage the buffer of the bitset. You can still use them in
your C projects, but they offer no protection from double "freeing".

```c
Bit_T Bit_new(int length);
void Bit_free(T set);
void Bit_free_safe(T *set);
```

### Bitset Properties
Return the length (capacity) of the bitset, the current population count of the
bitset. 
The last 3 functions will be used in future versions to facilitate the creation
of memory buffers that will be loaded onto bitsets. For now you can refer to the
header file to see what they do.
```c
int Bit_length(T set);
int Bit_count(T set);
int size_of_Bit_T(void);
int Bit_size(int length);
int Bit_buffer_size(int length);
```

### Bit Manipulation
Setting and clearing of irregular arrays (aset/aclear) of bits in a bitset,
setting and clearing of ranges of bits (set/clear), or individual bits
(bset/bclear).
Other functions put a specific value in a given bit (and return the old bit),
i.e. put, or return the current value of the bit (get). We can also map a
function on the entire bit, clear the entire bitset or negate the bitset (not).

```c
void Bit_aset(T set, int indices[], int n); 
void Bit_bset(T set, int index);
void Bit_aclear(T set, int indices[], int n);
void Bit_bclear(T set, int index);
void Bit_clear(T set, int lo, int hi);
int Bit_get(T set, int index);
void Bit_map(T set, void apply(int n, int bit, void *cl), void *cl);
void Bit_not(T set, int lo, int hi);
int Bit_put(T set, int n, int bit);
void Bit_set(T set, int lo, int hi);
```

### Bitset Comparisons
Standard equality, less than equal, more than equal operations between two
bitsets

```c
int Bit_eq(T s, T t);
int Bit_leq(T s, T t);
int Bit_lt(T s, T t);
```

### Set Operations
Those are grouped in functions that return a bitset that is the difference
(minus), symmetric difference (diff), union and intersection of two bitsets.
Alternatively, one returns the population counts of the result of these set
operations, without actually forming it.

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

- **libpopcnt**: A C/C++ library for counting the number of 1 bits (bit
  population count) specialized for different CPU architectures. Licensed under
  BSD 2-Clause.
    https://github.com/kimwalisch/libpopcnt
- **sse-popcount**: The SIMD population count implementation of the Harley-Seal
  algorithm based on the paper "Faster Population Counts using AVX2
  Instructions" by Daniel Lemire, Nathan Kurz and Wojciech Mula.
    https://github.com/WojciechMula/sse-popcount
- **cii** : The Bit_T library in C interfaces and implementations by David
  Hanson
    https://github.com/drh/cii 


## Performance

The library is optimized for performance, with specialized implementations of
the population count for different CPU architectures:

- **AVX512**: Utilizes 512-bit vector operations for maximum throughput
- **AVX2**: Uses 256-bit vector operations on supported CPUs
- **NEON**: Falls back to 128-bit vector operations on older CPUs
- **SVE**
- **Scalar**: Provides optimized scalar implementations for universal
  compatibility. The scalar implementation is based on the **Wilkes-Wheeler-Gill
  Algorithm**: A highly portable and efficient algorithm   for counting set bits 
  documented in "The Preparation of Programs for an Electronic Digital
  Computer". 
  Nearly 70 years after it's introduction, this scalar algorithm outperforms
  hardware popcounts (including SSE popcounts and hardware scalar popcounts)

The benchmarking suite demonstrates significant performance advantages over
naive implementations, especially for large bitsets.
Note that currently the population counts on the results of set operations use
the builtin popcount or the WCG scalar implementations. The compiler may (or may
not) promote them to a vectorized popcount depending on the compiler, the
architecture, the flags, the alignment of the planets etc. 

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