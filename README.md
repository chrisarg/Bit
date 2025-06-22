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
  counts", aka _popcount_ aka _popcnt_) for CPUs including:
  - Hardware POPCNT
  - Wilkes-Wheeler-Gill (WWG) algorithm
  - SIMD-accelerated counting
  The optimal method is chosen by the library during cpuid (invoked once at
  first use).
  This can be turned off by setting LIBPOPCNT environment variable to 0 no n
  false f off during compilation. In that case, we will fall back to the WWG algorithm.
- **Set Operations**: Union, intersection, difference, and symmetric difference
- **Comprehensive API**: Based on David Hanson's "C Interfaces and Implementations" design
- **Thread-Safety**: No global state, all operations are reentrant
- **Loading of bitset buffers**: This will facilitate integration and coupling
  with scripting languages (under development)
- **Hardware (GPU) acceleration**: Using OpenMP to offload set operations over
  bit containers in Graphic Processing Units and TPUs (e.g. Coral TPU, under
  development). Population counts are carried out using the WWG algorithm.  
- **Perl interface**: Both object oriented and functional (under development)

## Installation

### Prerequisites

- C compiler (GCC, Clang, or Intel ICX) : GCC extensively tested, Clang and ICX
  support still experimental
- GNU Make

### Building

```bash
# Clone the repository
git clone https://github.com/username/Bit.git
cd Bit

# Build the library
make


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
    
    // Create a bitset to hold the interesection
    Bit_T intersection = Bit_inter(bitset1, bitset2);
    
    // Clean up
    Bit_free(&bitset1);
    Bit_free(&bitset2);
    Bit_free(&intersection);

    // How to properly utilize an externally allocated buffer
    int length = 1024;
    int nbytes = Bit_buffer_size(length);
    unsigned char *buffer = malloc(nbytes);
    fill_with_values(buffer); 
    Bit_T bitset = Bit_load(length,buffer);
    do_things_with(bitset);
    Bit_free(&bitset);
    do_otherthings_with(buffer);
    free(buffer);

    //----------------------------------------------------------------
    // Using the packed containers; based on the openmp_bit.c benchmark
    int num_of_bits = 65536;
    int num_of_bits = 1000;
    int num_of_ref_bitsets = 5000;

    // allocate the bitsets as arrays of Bit_T & put some data in them
    Bit_T *bits = malloc(num_of_bits * sizeof(Bit_T));
    Bit_T *bitsets = malloc(num_of_ref_bits * sizeof(Bit_T));
    for (int i = 0; i < num_of_bits; i++) {
      bits[i] = Bit_new(size);
      Bit_set(bits[i], size / 2, size - 1);
    }
    for (int i = 0; i < num_of_ref_bits; i++) {
      bitsets[i] = Bit_new(size);
      Bit_set(bitsets[i], size / 2, size - 1);
    }

    // move them to packed containers 
    Bit_T_DB db1 = BitDB_new(size, num_of_bits);
    Bit_T_DB db2 = BitDB_new(size, num_of_ref_bits);
    for (int i = 0; i < num_of_bits; i++)
      BitDB_put_at(db1, i, bits[i]);
    for (int i = 0; i < num_of_ref_bits; i++)
      BitDB_put_at(db2, i, bitsets[i]);

    // These give equivalent results 
    int *results_CPU_container = 
      BitDB_inter_count_cpu(db1, db2, 
      (SETOP_COUNT_OPTS){.num_cpu_threads = num_threads});

    int *results_GPU_container = BitDB_inter_count_gpu(db1, db2,
                           (SETOP_COUNT_OPTS){.device_id = 0,
                                              .upd_1st_operand = true,
                                              .upd_2nd_operand = false,
                                              .release_1st_operand = true,
                                              .release_2nd_operand = true,
                                              .release_counts = true});

    // In case you would like to do intersection counts on Bit_T arrays
    size_t workload = (size_t)num_of_bits * (size_t)num_of_ref_bits;
    int *results_CPU_OMP = (int *)calloc(workload, sizeof(int));
      assert(counts != NULL);
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < num_of_bits; i++) {
      for (int j = 0; j < num_of_ref_bits; j++) {
        counts[i * num_of_ref_bits + j] = Bit_inter_count(bit[i], bitsets[j]);
      }
    }

    
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
bitset or the size (in bytes) of the buffer needed to store the bits in the bitset
of a given length.
```c
int Bit_length(T set);
int Bit_count(T set);
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

- **AVX512**: Utilizes 512-bit vector operations for maximum throughput. 
The implementation will depend on the processor architecture and may include HW
popcounts or the Harley - Searl algorithm.
- **AVX2**: Uses 256-bit vector operations on supported CPUs to implement the 
Harley - Searl algorithm
- **NEON**: Falls back to 128-bit vector operations on older CPUs
- **SVE**
- **Scalar**: Provides optimized scalar implementations for universal
  compatibility. The scalar implementation is based on the **Wilkes-Wheeler-Gill
  Algorithm**: A highly portable and efficient algorithm   for counting set bits 
  documented in "The Preparation of Programs for an Electronic Digital
  Computer". 

The Wilkes-Wheeler-Gill algorithm is used as default for GPU deployments given
the straightforward translation into highly efficient GPU code (under -O3).

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