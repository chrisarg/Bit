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
  offering multiple implementations of bit counting algorithms ("population
  counts", aka _popcount_ aka _popcnt_) for CPUs including:
  - Hardware POPCNT
  - Wilkes-Wheeler-Gill (WWG) method[^1]
  - SIMD-accelerated popcounting based on AVX, AVX2 or AVX512 instructions
    The optimal method is chosen by the library upon first use. 
    This can be turned off by setting LIBPOPCNT environment variable to one of
    0 , no , n , false , f , off during compilation.
    In that case, we will fall back to the portable WWG algorithm.
- **Set Operations**: Union, intersection, difference, and symmetric difference
- **Comprehensive API**: Based on David Hanson's "C Interfaces and Implementations" design
- **Thread-Safety**: No global state, all operations are reentrant
- **Utilizing externally allocated buffers**: Allows one to store (and extract)
  bitsets in externally allocated buffers.
- **Hardware (GPU) acceleration**: Using OpenMP to offload set operations over
  bit containers in Graphic Processing Units. The default build (`GPU=NONE`)
  routes all GPU-facing calls to CPU implementations; pass `GPU=NVIDIA` or
  `GPU=AMD` to enable device offloading. Supported targets include NVIDIA GPUs
  (via CUDA/nvptx OpenMP offload), AMD GPUs (via ROCm/amdgcn offload), and
  integrated GPUs (tested with Intel iGPUs, though performance gains are minimal
  and Unified Shared Memory has not been exploited yet). Offloading to TPUs
  (e.g. Coral TPU) is under development.
  Population counts on the GPU are carried out using the WWG algorithm.
- **Containerized operations**: These allow operations (e.g. intersect counts)
  between two packed containers of Bits using either the CPU or the GPU.
  Multithreading in the CPU and GPU offloading requires the presence of OpenMP.
  The storage space of containers managed by the library will be aligned to 
  64 bit (or even 32 bit!) addresses to assist with SIMD operations when
  operating at large collections of packed bitsets. 
- **Perl interface**: Interface is provided by the Bit::Set MetaCPAN [package](https://metacpan.org/pod/Bit::Set)

## Installation

### Prerequisites

- C compiler (`gcc` or `clang`)
- GNU Make

### Building

This assumes that you have CUDA installed for NVIDIA builds and, if you want to
use AMD offload, a ROCm-supported AMD GPU with a matching ROCm/LLVM stack. If
you do not want to build for GPU, ignore all GPU sections.

```bash
# Clone the repository
git clone https://github.com/username/Bit.git
cd Bit

# Build the library (default: GPU=NONE; all GPU functions delegate to CPU)
make

# Build for NVIDIA with clang (supported path; auto-detect visible GPU SMs)
make CC=clang GPU=NVIDIA

# Build for one specific NVIDIA architecture
make CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_75

# Build for an explicit comma-separated list of NVIDIA architectures
make CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_70,sm_80

# GCC NVIDIA build (policy-implicit; picks the earliest compiler-supported SM)
make CC=gcc GPU=NVIDIA

# Build for NVIDIA using a specific CUDA toolkit location
make CC=clang GPU=NVIDIA CUDA_PATH=/usr/local/cuda-11.8

# Combine both: custom CUDA location + one architecture
make CC=clang GPU=NVIDIA CUDA_PATH=/usr/local/cuda-11.8 NVIDIA_ARCH=sm_70


# Build the library for an AMD GPU with clang/ROCm offload
make CC=clang GPU=AMD

# Override the AMD target architecture when needed
make CC=clang GPU=AMD AMD_ARCH=gfx90a
```

Plain `make` now defaults to `GPU=NONE`.

On this developer machine, the following combinations are known to build:

- `make GPU=NONE`
- `make CC=gcc GPU=NVIDIA` auto-detecting `sm_35`
- `make CC=clang GPU=NVIDIA` auto-detecting `sm_52,sm_70`

The AMD path is not validated on this machine because the installed GPU is not
supported by ROCm. In particular, `make CC=gcc GPU=AMD` does not work here.
If you have a ROCm-supported AMD GPU, you should run the AMD tests on that
machine to confirm the architecture/runtime pairing.

For `CC=clang GPU=NVIDIA`, the supported flow is:

- use `NVIDIA_ARCH=sm_xy[,sm_ab,...]` for explicit targets,
- otherwise the Makefile auto-detects visible GPU architectures via `nvidia-smi`.

If architecture detection fails (no `nvidia-smi` output and no explicit `NVIDIA_ARCH`), the build errors out early. The canonical recovery is:

```bash
make CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_xy
```

NVIDIA build variables:

- `NVIDIA_ARCH`: one SM target or a comma-separated list such as `sm_70,sm_80`; input is normalized case-insensitively, each entry must match `sm_<digits>`. When omitted, visible GPU architectures are auto-detected via `nvidia-smi`.
- `CUDA_PATH`: path to the CUDA toolkit used by clang offload (default: `/usr/lib/cuda`).

AMD build variables:

- `AMD_ARCH`: one AMD gfx target such as `gfx90a`; input is normalized case-insensitively and must match `gfx<target>`.

The Makefile rejects `NVIDIA_ARCH` when `GPU=AMD`, rejects `AMD_ARCH` when `GPU=NVIDIA`, and rejects both when `GPU=NONE`.

If you set `NVIDIA_ARCH` explicitly on a multi-GPU system, make sure the list
covers the GPU you will actually run on. Otherwise LLVM OpenMP can fail at
runtime with a message like `No images found compatible with the installed
hardware`, even though the build succeeded. In that case, either include all
required SM targets in `NVIDIA_ARCH` or restrict execution with
`CUDA_VISIBLE_DEVICES` to a compatible GPU.

`CC=gcc GPU=NVIDIA` remains best-effort experimental. Both `gcc` and `clang`
accept a multi-SM list via `NVIDIA_ARCH`. Their behaviour differs:
`clang` embeds one native cubin per SM target (a true fat binary);
`gcc` does not inject an explicit NVIDIA `-march` target unless `NVIDIA_ARCH`
is provided. When `NVIDIA_ARCH` is set, it chooses the minimum SM from the
list and relies on the CUDA PTX JIT driver to produce native code for any newer
GPU at load time. If `NVIDIA_ARCH` is omitted, GCC uses its own default/PTX
fallback behaviour.
PTX JIT requires a CUDA driver at runtime — toolkit-only installs without a
driver may fail to offload on GPUs newer than the minimum target SM.


### GPU Troubleshoooting and Benchmarking

Building of the library should happen without issues, in the absence of GPU offloading. 
While GPU offloading should work mostly out of the box, compiler and runtime incompatibilities may surface.
Furthermore, the code relies heavily on preprocessor directives, and complex OpenMP directives which may not
be supported by your runtime. To isolate general GPU offloading issues from the use of the complex directives, 
you can use the `test_offload` binary which checks offloading through three classes of benchmarks:

- **[MEMORY-BOUND]**: Transfer data from host to device and back in each iteration.
  Measures GPU throughput for data-parallel elementwise operations.
  
- **[HYBRID-COMPUTE]**: Performs compute-heavy work per element but still transfers
  the entire output array to the host after each iteration.
  Partially isolates computation from memory I/O but does not fully eliminate
  PCIe/bus overhead.
  
- **[COMPUTE-BOUND][DEVICE-RESIDENT]**: Transfers input data once at the start,
  computes entirely on the GPU for all iterations, and returns only a single
  checksum result.
  This benchmark isolates pure GPU compute performance from host-device bus overhead,
  enabling measurement of sustained flop/ops rates without PCIe transfer bottlenecks.

Use the device-resident benchmark to profile GPU peak compute throughput. Compare
it against the hybrid-compute benchmark to quantify the cost of data movement
on your specific hardware. This benchmark uses simpler GPU kernels than the ones used
by the bit library and can help isolate compile/runtime issues and/or mismathces

#### Building `test_offload` with Custom Architectures

The `test_offload` binary inherits all compiler and GPU offload flags from the library build.
You can pass `NVIDIA_ARCH` or `AMD_ARCH` when invoking `make test_offload`:

**NVIDIA:**
```bash
# Build for a specific NVIDIA architecture
make test_offload CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_70

# Build for multiple NVIDIA architectures
make test_offload CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_70,sm_80

# Auto-detect visible NVIDIA architectures via nvidia-smi
make test_offload CC=clang GPU=NVIDIA

# Restrict execution to a GPU whose SM matches the built image set
CUDA_VISIBLE_DEVICES=1 OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0
```

**AMD:**
```bash
# Build for a specific AMD architecture
make test_offload CC=clang GPU=AMD AMD_ARCH=gfx900

# Build for a different AMD architecture
make test_offload CC=clang GPU=AMD AMD_ARCH=gfx90a
```

The `test_offload` binary can then be used to run correctness tests and benchmarks
with the chosen architecture. Pass benchmark iterations as the third argument to run
the GPU benchmark suite:

```bash
# Correctness tests only; require real device offload
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0

# Correctness tests + benchmarks with 100 iterations
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0 100
```

Using `OMP_TARGET_OFFLOAD=MANDATORY` is recommended when validating GPU builds so
host fallback is treated as failure instead of looking like a successful offload run.

#### libomptarget diagnostics defaults (overrideable)

The `Makefile` sets runtime diagnostics to quiet mode by default:

- `LIBOMPTARGET_INFO=0`
- `LIBOMPTARGET_DEBUG=0`

This avoids verbose `omptarget device ... info` logs unless explicitly requested.
You can override either variable at build time and they will be exported to
subprocesses spawned by `make`:

```bash
# Enable verbose runtime diagnostics
make LIBOMPTARGET_INFO=16 LIBOMPTARGET_DEBUG=1 test_offload CC=clang GPU=AMD AMD_ARCH=gfx900

# Keep default quiet mode explicitly
make LIBOMPTARGET_INFO=0 LIBOMPTARGET_DEBUG=0 test_offload CC=clang GPU=AMD AMD_ARCH=gfx900
```

If you run binaries directly from the shell (outside `make`), you can override
at runtime in the same way:

```bash
LIBOMPTARGET_INFO=16 LIBOMPTARGET_DEBUG=1 ./build/test_offload 100000 0
```

#### AMD Remediation (OpenMP Offload)
Common AMD GPU architectures that you can use as arguments:

- `gfx900` - Vega
- `gfx906` - Vega 20
- `gfx908` - CDNA 1 / MI100
- `gfx90a` - CDNA 2 / MI200

If AMD offload does not initialize, use this remediation flow on a machine with
a ROCm-supported AMD GPU:

```bash
# 1) Confirm your detected AMD gfx target
rocminfo | grep -Eo 'gfx[0-9]+' | sort -u

# 2) Build explicitly for the detected target
make clean
make test_offload CC=clang GPU=AMD AMD_ARCH=<gfx_target>

# 3) Run with mandatory offload
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 4096 0
```

If your system has multiple accelerators and AMD visibility is ambiguous, pin
the AMD device explicitly:

```bash
ROCR_VISIBLE_DEVICES=0 OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 4096 0
```

If `make` reports that no OpenMP AMD device runtime exists for a selected
`AMD_ARCH`, rebuild with a supported target for your installed LLVM/ROCm stack.

If you are validating the AMD path on supported hardware, run the full test
suite with mandatory offload so host fallback cannot hide a broken device path:

```bash
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0 100
```

If these tests fail on a ROCm-supported GPU, adjust `AMD_ARCH` to match the
installed LLVM/ROCm runtime and rerun before trusting the build.

If you are building for an architecture that is not supported (e.g. due to AMD's aggressive planned obsolescence),
you may be able to use the following workaround (tested with AMD Radeon Pro W5500 and llvm 18):
1. pick a supported architecture that is "close" to what you want to build for (e.g. use gfx1010 or 1030 for gfx1012)
and compile with that architecture.
2. In the shell you will be executing, fake the supported architecture by:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.1.0
```
3. Run normally
(note this hack is more likely to work with llvm than gcc)

#### NVIDIA GPU remediation (OpenMP offload)

On a multi-GPU NVIDIA system, the most reliable way to choose which card to
test is to use `CUDA_VISIBLE_DEVICES` and keep the OpenMP device index at `0`.
That makes the chosen physical GPU the only visible device, so the program sees
it as logical device `0`.

Examples:

```bash
# Test physical GPU 0
CUDA_VISIBLE_DEVICES=0 OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 1000000 0 1

# Test physical GPU 1
CUDA_VISIBLE_DEVICES=1 OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 1000000 0 1
```

If you expose both NVIDIA GPUs at once, current LLVM/OpenMP offload builds in
this environment can report zero devices or fail to initialize correctly.
That looks like a `libomptarget`/runtime quirk rather than a Bit issue, because
the same binary works when each GPU is tested individually. In practice, the
safe workaround is to run one GPU at a time.
On the other hand, gcc is more than happy to work with more than 1 NVIDIA GPUs.


### Compiler Bug Reports

The `Makefile` provides a `bug_report` target that captures compiler diagnostics,
build logs, and a preprocessed source file in `bug_reports/`.

By default:

- `BUG_REPORT_DIR=bug_reports`
- `BUG_TARGET=bench_omp`
- `BUG_REPORT_TAG=$(notdir CC)-$(GPU)-<arch-tag>` (auto-computed; arch tag is the detected or explicit arch list for NVIDIA/AMD, and `cpu` for `GPU=NONE`)

Each invocation creates a new report subdirectory:

- `bug_reports/<timestamp>-<BUG_REPORT_TAG>/`

The target automatically applies compiler-specific reporting flags:

- GCC: `-freport-bug`
- Clang: `-gen-reproducer -fcrash-diagnostics-dir=<report-subdir>`

Examples:

```bash
# GCC AMD offload bug report
make bug_report CC=gcc GPU=AMD AMD_ARCH=gfx900 BUG_TARGET=bench_omp

# Clang AMD offload bug report
make bug_report CC=clang GPU=AMD AMD_ARCH=gfx90a BUG_TARGET=bench_omp

# Clang NVIDIA offload bug report with explicit SM target
make bug_report CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_70 BUG_TARGET=bench_omp
```

After completion, inspect files in the new per-run report directory, especially:

- `build.log`
- `config.txt`
- `backtrace.txt`
- `src-bit.preprocessed.i`

For `CC=gcc`, the report also includes GCC-specific files aligned with
`https://gcc.gnu.org/bugs/` guidance:

- `gcc-v.txt` (exact GCC version, target system type, and configure options from `gcc -v`)
- `gcc-repro-command.txt` (complete reproduction compile command)
- `gcc-save-temps.log` (compiler output from `-v -save-temps` reproduction run)

If Clang emits crash reproducers, they will also be written in the same report
subdirectory. The `bug_report` target removes temporary debug artifacts
(`*.i`, `*.ii`, `*.s`, `*.bc`, `*.cui`) from `build/` after report collection.

When a build fails, `bug_report` automatically reruns the failing target under
`gdb` and writes a full stack trace to `backtrace.txt`. If `gdb` is not
available, `backtrace.txt` records that limitation.



### Benchmarking Bit

Now you are ready to run the tests and build the benchmarks. These invocation of make take the
same arguments used to build the library. If you use different arguments to make and test, prepare for a poor experience.

```bash
# Run tests (GPU=NONE by default; pass GPU=NVIDIA or GPU=AMD to test offload)
make test

# Make the benchmark (this will also build the OpenMP benchmark)
make bench 

# Make the OpenMP benchmarks
make bench_omp GPU=NONE                          # CPU only (default)
make bench_omp CC=clang GPU=NVIDIA               # NVIDIA offload, auto-detect SM
make bench_omp CC=clang GPU=AMD AMD_ARCH=gfx90a  # AMD offload
```

```bash
# Runs various benchmarks
./build/benchmark

# Mixed CPU/GPU OpenMP benchmark
Usage: ./build/openmp_bit <size> <number of bitsets> <number of reference bitsets> <max threads> [<gpu_id>]

# CPU-only OpenMP benchmark
Usage: ./build/openmp_bit_nogpu <size> <number of bitsets> <number of reference bitsets> <max threads>

# GPU-only OpenMP benchmark
Usage: ./build/openmp_bit_nocpu <size> <number of bitsets> <number of reference bitsets> <gpu iterations> [<gpu_id>]
```

The OpenMP benchmark assesses the scaling of searching (intersection count) of a
number of bits of given size(capacity) against a database of reference bitsets.
The benchmark will run:

- 3 repetitions of a single threaded search
- the same query using OpenMP without containers utilizing 1 to max_threads
- containerized OpenMP query utilizing 1 to max_threads
- containerized OpenMP query using GPU offloading

The CPU-only benchmar (`openmo_bit_nogpu`) omits entirely the GPU benchmarks.

The repository [benchmarking-bits](https://github.com/chrisarg/benchmarking-bits) 
contains benchmarks against other bitset/bitvector/bitmaps in C and Perl.

### Working with the gpuOpt branch
This branch exposes an additional makefile (`Makefile.bench`) that is used in active 
development of native CUDA/HIP code that may at some point replace the OpenMP code. 
The CUDA/HIP sections are being developed with heavy AI assist, so if you end up using, you are at the mercy of the clankers (mostly Raptor mini, Gemini Flash with the occasional Grok). This is how you

The GPU-only benchmark (`openmp_bit_nocpu`) runs only containerized GPU
offloaded intersection counts. Its 4th argument uses the same CLI position as
`max threads`, but is interpreted as the number of GPU iterations.
This is a safe harbor for testing various implementations of GPU code (e.g. 
OpenMP directives, or popcount algorithms) without messing with `bit.c`.

This is how you use the the gpuOpt branch specific makefile, `Makefile.bench`:

```bash 
# GPU-only OpenMP benchmark and native CUDA/HIP benchmarks (Makefile.bench)
make -f Makefile.bench openmp_bit_nocpu CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_70
make -f Makefile.bench openmp_bit_nocpu CC=clang GPU=AMD AMD_ARCH=gfx90a
make -f Makefile.bench cuda_gpu_bench GPU=NVIDIA NVIDIA_ARCH=sm_70
make -f Makefile.bench hip_gpu_bench GPU=AMD AMD_ARCH=gfx90a

```
#### Interpreting `openmp_bit_nocpu` Output

The GPU-only benchmark prints multiple timing blocks so on can separate kernel
cost from host/device transfer and orchestration overhead:

- **GPU Algorithm Timing**
  - Time spent in the GPU intersection algorithm itself (kernel-focused view).
  - Use this to compare algorithmic efficiency across architectures/compilers.

- **GPU Algorithm + PCIe Timings**
  - End-to-end device path including host/device movement.
  - Use this to evaluate real execution cost when transfers are part of each
    iteration.

- **CPU Overhead Timings**
  - Host-side setup/dispatch/synchronization overhead around GPU work.
  - Useful for understanding launch/runtime overhead at small problem sizes.

- **Per-Iteration Data Movement Breakdown**
  - Reports transfer volume and transfer time per iteration.
  - Use this to see whether runtime is dominated by movement or compute.

- **Estimated Throughput**
  - Effective rate computed from processed data and measured time.
  - Treat this as a practical end-to-end throughput metric (not a pure bus
    bandwidth number unless only transfer time is used in the denominator).

In short: compare **GPU Algorithm Timing** for compute behavior, and compare
**GPU Algorithm + PCIe Timings** plus movement breakdown for real workload
performance.




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

Create, free a _Bitset_ or load and extract a bitset using externally allocated
buffers.

```c
extern Bit_T Bit_new(int length);
extern void* Bit_free(T* set);
extern Bit_T Bit_load(int length, void* buffer);
extern int Bit_extract(Bit_T set, void* buffer);

```

Create (de novo or from an external buffer) and free a _Bitset container_ aka a _BitDB_

```C
extern Bit_T_DB BitDB_new(int length, int num_of_bitsets);
extern void* BitDB_free(T_DB* set);
extern Bit_T_DB BitDB_load(int length, int num_of_bitsets, void* buffer);
```

Both free functions return the NULL pointer if the buffer was allocated by the
library, or the pointer to the buffer that was loaded externally.

### Bitset and Bitset container properties

Return the length (capacity) of the _Bitset_, the current population count of the
bitset or the size (in bytes) of the buffer needed to store the bits in the bitset
of a given length.

```c
int Bit_length(Bit_T set);
int Bit_count(Bit_T set);
int Bit_buffer_size(int length);
```

Return the length of a _Bitset_ in a _BitDB_ container and the number of bitsets
in the container. Other functions return the total population count of the
container, or the count at a particular index in the container

```c
extern int BitDB_length(Bit_DB_T set);
extern int BitDB_nelem(Bit_DB_T set);
extern int BitDB_count_at(Bit_DB_T set, int index);
extern int* BitDB_count(Bit_DB_T set);
```

### Bitset and Bitset container Manipulation

Setting and clearing of irregular arrays (aset/aclear) of bits in a _Bitset_,
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

For the _Bitset_ container (_BitDB_), the manipulation functions operate on
individual bitsets within the container. There are functions that extract
bitsets from the given index of a container and return them as a bitset, or
functions that replace bitsets at a particular index.
One can also use raw byte buffers to extract or replace bitsets at specific
indices. Finally one can clear entire bitset containers, or bitsets in a
particular index in the container.

```c
extern Bit_T BitDB_get_from(Bit_DB_T set, int index);
extern void BitDB_put_at(Bit_DB_T set, int index, T bitset);
extern void BitDB_extract_from(Bit_DB_T set, int index, void* buffer);
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

Those are grouped in functions that return a _Bitset_ that is the difference
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

_Bitset container_ operations are available through two separate interfaces:

- A _macro-based interface_ for use within C
- A _function-based interface_ when interfacing with foreign code

#### Macro based interface

The macro-based interface involves two sets of four functions:
BitDB_SETOP_count and BitDB_SETOP_count_store, where SETOP can be one of the following:

1. inter = intersection
2. union = union
3. diff = difference
4. minus = symmetric difference

Each of these functions take as arguments two Bitset containers, performs the
SETOP operation, for all the bitsets in these containers and either returns
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
first function call will be N _ M (i.e. a two-dimensional array stored in
row-major order). Similarly, the space that must be pre-allocated to hold the
results will need to be of size N _ M \* sizeof(int) bytes.

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
results of size N \* M. In this case,

```c
SETOP_COUNT_OPTS opts_1to2 = {
    .device_id = 0,
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
    .device_id = 0,
    .upd_1st_operand = true,
    .upd_2nd_operand = false,
    .release_1st_operand = true,
    .release_2nd_operand = true,
    .release_counts = true
};
```

which will update the first operand in the GPU and _release_ all the buffers
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
extern void BitDB_inter_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_inter_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_inter_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);

extern void BitDB_union_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern void BitDB_union_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_union_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_union_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);

extern void BitDB_diff_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern void BitDB_diff_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern int* BitDB_diff_count_cpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);
extern int* BitDB_diff_count_gpu(Bit_DB_T bit, Bit_DB_T bits, SETOP_COUNT_OPTS opts);

extern void BitDB_minus_count_store_cpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
    SETOP_COUNT_OPTS opts);
extern void BitDB_minus_count_store_gpu(Bit_DB_T bit, Bit_DB_T bits, int* buffer,
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

The library `libpopcnt` is optimized for performance, with specialized implementations of
the *population count* for different CPU architectures:

- **AVX512**: Utilizes 512-bit vector operations for maximum throughput.
  The implementation will depend on the processor architecture and may include HW
  popcounts or the Harley - Searl algorithm.
- **AVX2**: Uses 256-bit vector operations on supported CPUs to implement the
  Harley - Searl algorithm
- **NEON**: Falls back to 128-bit vector operations on older CPUs
- **SVE**
- **Scalar**: Provides optimized scalar implementations for universal
  compatibility. The scalar implementation is based on the **Wilkes-Wheeler-Gill
  Algorithm**: A highly portable and efficient algorithm for counting set bits
  documented in "The Preparation of Programs for an Electronic Digital
  Computer".

The Wilkes-Wheeler-Gill algorithm is used as default for GPU deployments given
the straightforward translation into highly efficient GPU code (under -O3). Native GPU popcount instructions do exist and are 2.5x faster ONLY is one can stage their data to operate in registers or in shared (thread local) memory. For practical applications, performance is memory bound so it makes absolutely no practical difference is one is using the builtin directive or the WWG function in the GPU. 

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
    fprintf(stderr, "Error: Unable to allocate memory for counts array of "
    "size %zu in %s\n", workload,__func__);
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
code for either CPU or GPU environments. GPU offloading is opt-in: the default
build (`GPU=NONE`) routes all GPU calls to CPU implementations. Pass
`GPU=NVIDIA` or `GPU=AMD` to enable device offloading.
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

- Build the entire library in the CPU using [SIMDe](https://github.com/simd-everywhere/simde) (SIMD everywhere). This will allow portable, vectorized operations for both bitsets and containers. At that point we will no longer rely on `libpopcnt`. 
- Implement additional set-op operations (e.g. the Jaccard index)
- Implement additional, OS agnostic build systems 
- Code the setop functions (e.g. and, not, xor etc) using SIMD directives. This will require us to redesign the `cii` set-op interface for these operations. 
- Ensure that gcc and clang are fully tested across CPU, NVIDIA, and AMD paths
- CUDA and HIP implementations to replace OpenMP implementations in systems that feature the `nvcc` or the `hipcc` compiler
- Utilize Unified Shared Memory if available in the system
- TPU & NPU support (low priority but will be cool with all the new chips)

## License

BSD 2-Clause License. See the LICENSE file for details.

## Author

Christos Argyropoulos (April 2025 -  May 2026)

## AI disclosure
Github Copilot has been very helpful when it comes to generating the makefile, run ideas about the OpenMP and to generate the CUDA and HIP implementations. 

[^1]: Historical Trivia: The method is identified as the Gillies-Miller 
"sideways addition” in the original reference (Maurice V. Wilkes, 
David J. Wheeler, and Stanley Gill. _The Preparation of Programs for 
an Electronic Digital Computer_, chapter Gillies–Miller method for 
sideways addition, pages 191–193. Addison-Wesley Publishing Company, 
Reading, Mass., 2nd edition, 1957.) but it was named the 
”Wilkes-Wheeler-Gill function in C” by Mula, Kurz and Lemire 
(Faster population counts using avx2 instructions. _The Computer Journal_,
 61(1):111–120, May 2017.), leaving some confusion about who originated 
 the method, though the first implementation may had been written by
  Wilkes, Wheeler andGill in support of their 1957 book.