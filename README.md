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

- **Specialized Population Counting**: Integration of the libpopcnt library 
  offering  multiple implementations of bit counting algorithms ("population
  counts", aka _popcount_ aka _popcnt_) for CPUs including:
  - Hardware POPCNT
  - Wilkes-Wheeler-Gill (WWG) algorithm
  - SIMD-accelerated popcounting based on AVX, AVX2 or AVX512 instructions
  The optimal method is chosen by the library during cpuid (invoked once at
  first use).
  This can be turned off by setting LIBPOPCNT environment variable to one of 
  0 , no , n , false , f , off during compilation. 
  In that case, we will fall back to the portable WWG algorithm.
- **Set Operations**: Union, intersection, difference, and symmetric difference
- **Comprehensive API**: Based on David Hanson's "C Interfaces and Implementations" design
- **Thread-Safety**: No global state, all operations are reentrant
- **Utilizing externally allocated buffers**: Allows one to store (and extract)
  bitsets in externally allocated buffers.
- **Hardware (GPU) acceleration**: Using OpenMP to offload set operations over
  bit containers in Graphic Processing Units. The default is to offload to an
  NVIDIA, but one can turn off offloading, in which case the GPU functionality
  will default to CPUs. Offloading to TPUs (e.g. Coral TPU) is under
  development. 
  Population counts in GPUs are carried out using the WWG algorithm.  
- **Containerized operations**: These allow operations (e.g. intersect counts)
  between two packed containers of Bits using either the CPU or the GPU. 
  Multithreading in the CPU and GPU offloading requires the presence of OpenMP
- **Perl interface**: Interface is provided by the Bit::Set MetaCPAN [package](https://metacpan.org/pod/Bit::Set)

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

# Build the library without GPU support
make GPU=NONE

# Build the library for an AMD GPU
make GPU=AMD

# Run tests
make test

# Make the benchmark (this will also build the OpenMP benchmark)
make bench

# Make the open-mp benchmarks: with and without GPU support if GPU is not set to NONE
# or without GPU support otherwise
make bench_omp
```
### Benchmarking

```bash
# Runs various benchmarks
./build/benchmark  

# OpenMP benchmark
Usage: ./build/openmp_bit <size> <number of bitsets> <number of reference bitsets> <max threads>
```

The OpenMP benchmark assesses the scaling of searching (intersection count) of a
number of bits of given size(capacity) against a database of reference bitsets. 
The benchmark will run:
 * 3 repetitions of a single threaded search
 * the same query using OpenMP without containers utilizing 1 to max_threads
 * containerized OpenMP query utilizing 1 to max_threads
 * containerized OpenMP query using GPU offloading  

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
    Bit_DB_T db1 = BitDB_new(size, num_of_bits);
    Bit_DB_T db2 = BitDB_new(size, num_of_ref_bits);
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
Create, free a *Bitset* or load and extract a bitset using externally allocated 
buffers.
```c
extern Bit_T Bit_new(int length);  
extern void* Bit_free(T* set); 
extern Bit_T Bit_load(int length, void* buffer);
extern int Bit_extract(Bit_T set, void* buffer);

```

Create and free a *Bitset container* aka a *BitDB* 

```C
extern Bit_T_DB BitDB_new(int length, int num_of_bitsets);
extern void* BitDB_free(T_DB* set);
```

### Bitset and Bitset container properties
Return the length (capacity) of the *Bitset*, the current population count of the
bitset or the size (in bytes) of the buffer needed to store the bits in the bitset
of a given length.
```c
int Bit_length(Bit_T set);
int Bit_count(Bit_T set);
int Bit_buffer_size(int length);
```

Return the length of a *Bitset* in a *BitDB* container and the number of bitsets
in the container. Other functions return the total population count of the
container, or the count at a particular index in the container

```c
extern int BitDB_length(Bit_DB_T set);
extern int BitDB_nelem(Bit_DB_T set);
extern int BitDB_count_at(Bit_DB_T set, int index);
extern int* BitDB_count(Bit_DB_T set);
```


### Bitset and Bitset container Manipulation
Setting and clearing of irregular arrays (aset/aclear) of bits in a *Bitset*,
setting and clearing of ranges of bits (set/clear), or individual bits
(bset/bclear).
Other functions put a specific value in a given bit (and return the old bit),
i.e. put, or return the current value of the bit (get). We can also map a
function on the entire bit, clear the entire bitset or negate the bitset (not).

```c
void Bit_aset(Bit_T set, int indices[], int n); 
void Bit_bset(Bit_T set, int index);
void Bit_aclear(Bit_T set, int indices[], int n);
void Bit_bclear(Bit_T set, int index);
void Bit_clear(Bit_T set, int lo, int hi);
int Bit_get(Bit_T set, int index);
void Bit_map(Bit_T set, void apply(int n, int bit, void *cl), void *cl);
void Bit_not(Bit_T set, int lo, int hi);
int Bit_put(Bit_T set, int n, int bit);
void Bit_set(Bit_T set, int lo, int hi);
```

For the *Bitset* container (*BitDB*), the manipulation functions operate on
individual bitsets within the container. There are functions that extract
bitsets from the given index of a container and return them as a bitset, or 
functions that replace bitsets at a particular index. 
One can also use raw byte buffers to extract or replace bitsets at specific
indices. Finally one can clear entire bitset containers, or bitsets in a 
particular index in the container. 
 
```c
extern Bit_T BitDB_get_from(Bit_DB_T set, int index);
extern void BitDB_put_at(Bit_DB_T set, int index, T bitset);
extern int BitDB_extract_from(Bit_DB_T set, int index, void* buffer);
extern void BitDB_replace_at(Bit_DB_T set, int index, void* buffer);
extern void BitDB_clear(Bit_DB_T set);
extern void BitDB_clear_at(Bit_DB_T set, int index);

```

### Bitset Comparisons
Standard equality, less than equal, more than equal operations between two
bitsets

```c
int Bit_eq(Bit_T s, Bit_T t);
int Bit_leq(Bit_T s, Bit_T t);
int Bit_lt(Bit_T s, Bit_T t);
```

### Set Operations
Those are grouped in functions that return a *Bitset* that is the difference
(minus), symmetric difference (diff), union and intersection of two bitsets.
Alternatively, one returns the population counts of the result of these set
operations, without actually forming it.

```c
Bit_T Bit_diff(Bit_T s, Bit_T t);
Bit_T Bit_inter(Bit_T s, Bit_T t);
Bit_T Bit_minus(Bit_T s, Bit_T t);
Bit_T Bit_union(Bit_T s, Bit_T t);

int Bit_diff_count(Bit_T s, Bit_T t);
int Bit_inter_count(Bit_T s, Bit_T t);
int Bit_minus_count(Bit_T s, Bit_T t);
int Bit_union_count(Bit_T s, Bit_T t);
```

*Bitset container* operations are available through two separate interfaces:

* A *macro-based interface* for use within C
* A *function-based interface* when interfacing with foreign code

#### Macro based interface
The macro-based interface involves two sets of four functions:
BitDB_SETOP_count and BitDB_SETOP_count_store, where SETOP can be one of the following:
  1. inter = intersection
  2. union = union
  3. diff = difference
  4. minus = symmetric difference

Each of these functions take as arguments two Bitset containers, performs the
SETOP operation, for all the bitsets  in these containers and either returns 
the result as an array of integers or uses the an externally allocated
buffer that is provided as an argument to the (store) functions. 

For example:

```c
BitDB_inter_count(bit, bits, opts, TARGET); 
BitDB_inter_count_store(bit, bits, opts, results, TARGET)
```

both calculate the population count of the intersection of the bitsets of the 
cartesian product of the two containers. While the former returns the result as an
array of integers, the latter stores the result in the provided buffer
(results). In these invocations, if the number of elements of the first bitset is N,
and that of the second index is M, the total number of elements returned by the
first function call will be N * M (i.e. a two-dimensional array stored in
row-major order). Similarly, the space that must be pre-allocated to hold the
results will need to be of size N * M * sizeof(int) bytes.


When invoking the functions, the TARGET is one of cpu or gpu providing the 
execution context. The opts is a structure of type SETOP_COUNT_OPTS 
that is defined as follows:

```c
typedef struct {
    int num_cpu_threads;  // number of CPU threads
    int device_id;        // GPU device ID, ignored for CPU
    bool upd_1st_operand; // if true, update the first container in the GPU
    bool upd_2nd_operand; // if true, update the second container in the GPU
    bool release_1st_operand; // if true, release the first container in the GPU
    bool release_2nd_operand; // if true, release the second container in the GPU
    bool release_counts;    // if true, release the counts buffer in the GPU
} SETOP_COUNT_OPTS;
```

This structure provides the number of CPU threads that will be utilized when
running the code in the CPU, the device id for GPU execution, and various flags
for managing the GPU memory. Memory allocations and de-allocations in the CPU
are very costly, so it pays handsomely in terms of performance if one did not
have to move things around unless absolutely necessary. 
Consider for example the scenario in which one has 3 containers, each of size N
that must be matched against a single container of size M. The device has enough
memory to fit a single container of size N, another one of size N, and the
results of size N * M. In this case,

```c
SETOP_COUNT_OPTS opts_1to2 = {
    .device_id = -1,
    .upd_1st_operand = true,
    .upd_2nd_operand = false,
    .release_1st_operand = false,
    .release_2nd_operand = false,
    .release_counts = false
};
```

instructs the mapper to update the first operand in the GPU when iterating over
the first two containers of size N. To process the final container, one can use

```c
SETOP_COUNT_OPTS opts_3 = {
    .device_id = -1,
    .upd_1st_operand = true,
    .upd_2nd_operand = false,
    .release_1st_operand = true,
    .release_2nd_operand = true,
    .release_counts = true
};
```

which will update the first operand in the GPU and *release* all the buffers
on the device upon exit. Since OpenMP manages device memory regions using
reference counting, releasing of the regions amounts to decreasing the reference
counters for each of the regions. Regions that are no longer referenced will be
automatically de-allocated.

#### Function based interface

The macro interface expands to the functions in the function based interface.
Those are the following:

```c
extern int* BitDB_inter_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_inter_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_inter_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_inter_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);

extern int* BitDB_union_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_union_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_union_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_union_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);

extern int* BitDB_diff_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_diff_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_diff_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_diff_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);

extern int* BitDB_minus_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_minus_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_minus_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_minus_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
```

## Error checking for functions in the interface

C's assert is used to validate input parameters, memory allocations and internal
states within the functions. It is highly advisable NOT to define NDEBUG (e.g.
as a compiler flag) when compiling the library, as this will disable all checks
whatsoever. If you do so, please feel free to email me any disasters you may
encounter, especially in your GPU deployments.

The checked runtime errors for each function are described in the header file of
the API.

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

## A note about the implementation details for multi-threaded CPU and GPU deployments

The bitset operations can be very easily parallelized on the CPU using OpenMP or
even native C threads. By far the easiest path to parallelization on the CPU is 
offered by OpenMP, e.g. the following code in the `openmp_bit.c` source will
carry out a similarity search using the count of the intersection to find the
most similar reference bitset for each query bitset.

```c
int database_match_omp(Bit_T* bit, Bit_T* bitsets, int num_of_bits,
  int num_of_ref_bits, int threads) {
  // Perform the intersection count in parallel
  int max = 0, current = 0;
  size_t workload = (size_t)num_of_bits * (size_t)num_of_ref_bits;
  int* counts = (int*)calloc(workload, sizeof(int));
  if(counts == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for counts array of size %zu in %s\n", workload,__func__);
    exit(EXIT_FAILURE);
  }
  omp_set_num_threads(threads);
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_of_bits; i++) {
    for (int j = 0; j < num_of_ref_bits; j++) {
      counts[i * num_of_ref_bits + j] = Bit_inter_count(bit[i], bitsets[j]);
    }
  }

  for (size_t i = 0; i < workload; i++) {
    current = counts[i];
    if (current > max) {
      max = current;
    }
  }
  free(counts);
  return max;
}
```

The container versions fully leverage the capabilities of OpenMP to generate
code for either CPU or GPU environments. In the absence of a functional OpenMP
installation, the build will simply fail because of the failure to compile the
GPU code. Future releases of the library may include alternative compilation
paths that strip the GPU code entirely.
Those who are interested in the implementation feel free to look into code of
`bit.c`. I found the C preprocessor to be a valuable tool for managing the
complexity of the codebase and enabling code reuse. As the OpenMP itself uses
#pragma directive for parallel regions in both CPU and GPU, parameterizing these
directives, required the liberal use of the `_Pragma` operator to construct
#pragma directives from macro expansions. At some unspecified point in the
future, these and possibly other macros may be split into a header only library
to manage the expressive complexity of OpenMP for beginners. 


## Applications

Bit is particularly useful for:

- Bioinformatics and genomic data processing (k-mer encoding)
- Network packet filtering and bloom filters
- Dense data representation (for sparse bitsets over large domains, one is
  probably better off exploring sparse representations e.g. roaring bitsets)
- High-performance set operations

## TO-DO
- Implement additional set-op operations (e.g. the Jaccard index)
- Implement additional, OS agnostic build systems
- Code the setop functions (e.g. and, not, xor etc) using SIMD directives
- Ensure that clang and icx are fully tested and supported
- Test (including the build system!) on AMD and Intel GPUs
- TPU & NPU support (low priority but will be cool with all the new chips)


## License

BSD 2-Clause License. See the LICENSE file for details.

## Author

Christos Argyropoulos (April - September 2025)