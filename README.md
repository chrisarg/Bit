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

- C compiler (`gcc`, `clang`, `amdclang`, or `icx`)
- GNU Make
- **For Native GPU Benchmarks:** A C++ compiler is required (`nvcc` targeting C++14 for CUDA, or `hipcc` targeting C++17 for HIP builds).

### Building

This assumes that you have CUDA installed for NVIDIA builds and, if you want to
use AMD offload, a ROCm-supported AMD GPU with a matching ROCm/LLVM stack. If
you do not want to build for GPU, ignore all GPU sections. Plain `make` defaults to `GPU=NONE`.

```bash
# Clone the repository
git clone [https://github.com/username/Bit.git](https://github.com/username/Bit.git)
cd Bit

# Build the library (default: GPU=NONE; all GPU functions delegate to CPU)
make

# Build for NVIDIA with clang (supported path; auto-detect visible GPU SMs)
make CC=clang GPU=NVIDIA

# Build for specific GPU architectures using the universal GPU_ARCH variable
# (Requires sm_ or compute_ prefix for NVIDIA, and gfx prefix for AMD)
make CC=clang GPU=NVIDIA GPU_ARCH=sm_75
make CC=clang GPU=NVIDIA GPU_ARCH=compute_80

# Build for an explicit comma-separated list of architectures
make CC=clang GPU=NVIDIA GPU_ARCH=sm_70,compute_80

# Build for NVIDIA using a specific CUDA toolkit location
make CC=clang GPU=NVIDIA CUDA_PATH=/usr/local/cuda-11.8

# Build the library for an AMD GPU with clang/ROCm offload 
make CC=clang GPU=AMD

# Override the AMD target architecture when needed (requires gfx prefix)
make CC=clang GPU=AMD GPU_ARCH=gfx90a

# Heterogeneous multi-GPU "Fat Binary" compilation for NVIDIA and AMD offload targets
# The Makefile seamlessly routes both sm_ and gfx prefixes directly from GPU_ARCH
make CC=clang GPU=NVIDIA,AMD GPU_ARCH=sm_70,gfx90a
```

Plain `make` now defaults to `GPU=NONE`.

The multi-GPU build was validated on a machine with the following GPUs: RTX960 (sm_52), Titan V (sm_70), and Radeon Pro W5500 (gfx1012).
For both NVIDIA and AMD, the makefile will use `nvidia-smi` or `rocm-smi` to find the architectures present in a system if the
GPU_ARCH argument is not provided and attempt to build for those. If a given detected architecture is not supported by the compiler, the build will fail.


### Compiler & GPU Target Support Matrix

A comprehensive breakdown of the supported targets for each valid Compiler and GPU combination is given below:

#### Support Matrix

| Compiler (`CC=`) | GPU List (`GPU=`) | Core / Host Targets <br>*(from `Makefile`)* | OpenMP Offload Targets <br>*(from `Makefile` & `Makefile_bench.mak`)* | Native GPU Targets <br>*(from `Makefile_bench.mak`)* |
| :--- | :--- | :--- | :--- | :--- |
| **`gcc`** or **`clang`** | `NONE` | `all`, `test`, `bench`, `bug_report` | `test_offload`, `bench_omp` *(host-fallback)* | *None* *(Errors out)* |
| **`gcc`** or **`clang`** | `NVIDIA` | `all`, `test`, `bench`, `bug_report` | `test_offload`, `bench_omp`, `openmp_bit_nocpu` | `cuda_gpu_bench`, `gpu_bench_csv` |
| **`gcc`** or **`clang`** | `AMD` | `all`, `test`, `bench`, `bug_report` | `test_offload`, `bench_omp`, `openmp_bit_nocpu` | `hip_gpu_bench`, `gpu_bench_csv` |
| **`gcc`** or **`clang`** | `NVIDIA,AMD` | `all`, `test`, `bench`, `bug_report` | `test_offload`, `bench_omp`, `openmp_bit_nocpu` | `cuda_gpu_bench`, `hip_gpu_bench`, `gpu_bench_csv` |
| **`amdclang`** | `AMD` | `all`, `test`, `bench`, `bug_report` | `test_offload`, `bench_omp`, `openmp_bit_nocpu` | `hip_gpu_bench`, `gpu_bench_csv` |
| **`amdclang`** | *Any other* | *None (Make aborts parsing)* | *None* | *None* |
| **`icx`** (Intel) | `INTEL` | `all`, `test`, `bench`, `bug_report` | `test_offload`, `bench_omp`, `openmp_bit_nocpu`* | *None (Will Error)* |
| **`icx`** (Intel) | *Any other* | *None (Make aborts parsing)* | *None* | *None* |

---

## Important Notes & Restrictions

### 1. Invalid Combinations (Make Aborts)

* If you pass `CC=amdclang` and anything other than `GPU=AMD`, the build will immediately error out.
* If you pass `CC=icx` and anything other than `GPU=INTEL`, the build will immediately error out.
* You cannot combine `NONE` with other GPUs (e.g., `GPU=NONE,NVIDIA` throws a fatal error).

### 2. Native GPU Targets (`cuda_gpu_bench`, `hip_gpu_bench`)

* These strictly ignore your `CC` variable for the device code compilation, delegating instead to `nvcc` and `hipcc` respectively. 
* However, they **do** rely on the `GPU=` parameter guardrails. Attempting to build `hip_gpu_bench` without `GPU=AMD` (or `cuda_gpu_bench` without `GPU=NVIDIA`) will trigger a fatal error, regardless of the compiler.

### 3. `openmp_bit_nocpu` Target

* This explicitly verifies that offloading is enabled. If `GPU=NONE` is set, Make blocks execution and throws: *"openmp_bit_nocpu requires functional offloading; specify NVIDIA or AMD in your target array."*
* *(Note on Intel: Technically, the logic allows `icx` + `INTEL` to build this target because `INTEL` passes the `NONE` filter, but the error message hints it was predominantly designed for NVIDIA/AMD).*

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
You can pass explicitly pass `GPU_ARCH` when invoking `make test_offload`:

**NVIDIA:**

```bash
# Build for a specific NVIDIA architecture
make test_offload CC=clang GPU=NVIDIA GPU_ARCH=sm_70

# Build for multiple NVIDIA architectures
make test_offload CC=clang GPU=NVIDIA GPU_ARCH=sm_70,sm_80

# Auto-detect visible NVIDIA architectures via nvidia-smi
make test_offload CC=clang GPU=NVIDIA

# Restrict execution to a GPU whose SM matches the built image set
CUDA_VISIBLE_DEVICES=1 OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0
```

**AMD:**

```bash
# Build for a specific AMD architecture
make test_offload CC=clang GPU=AMD GPU_ARCH=gfx900

# Build for a different AMD architecture
make test_offload CC=clang GPU=AMD GPU_ARCH=gfx90a
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

The `Makefile` sets runtime diagnostics to quiet mode by default when using `clang`:

- `LIBOMPTARGET_INFO=0`
- `LIBOMPTARGET_DEBUG=0`

This avoids verbose `omptarget device ... info` logs unless explicitly requested.
You can override either variable at build time and they will be exported to
subprocesses spawned by `make`:

```bash
# Enable verbose runtime diagnostics
make LIBOMPTARGET_INFO=16 LIBOMPTARGET_DEBUG=1 test_offload CC=clang GPU=AMD GPU_ARCH=gfx900

# Keep default quiet mode explicitly
make LIBOMPTARGET_INFO=0 LIBOMPTARGET_DEBUG=0 test_offload CC=clang GPU=AMD GPU_ARCH=gfx900
```


#### AMD Remediation (OpenMP Offload)

Common AMD GPU architectures that you can use as arguments (but see below about not officially supported architectures):

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
make test_offload CC=clang GPU=AMD GPU_ARCH=<gfx_target>

# 3) Run with mandatory offload
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 4096 0
```

If your system has multiple accelerators and AMD visibility is ambiguous, pin
the AMD device explicitly:

```bash
ROCR_VISIBLE_DEVICES=0 OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 4096 0
```

If `make` reports that no OpenMP AMD device runtime exists for a selected
`GPU_ARCH`, rebuild with a supported target for your installed LLVM/ROCm stack.

If you are validating the AMD path on supported hardware, run the full test
suite with mandatory offload so host fallback cannot hide a broken device path:

```bash
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0 100
```

If these tests fail on a ROCm-supported GPU, adjust `GPU_ARCH` to match the
installed LLVM/ROCm runtime and rerun before trusting the build.

If you are building for an architecture that is not supported (e.g. due to AMD's aggressive planned obsolescence),
you may be able to use the following workaround (tested with AMD Radeon Pro W5500 and llvm 18):

1. pick a supported architecture that is "close" to what you want to build for  e.g. use gfx1010 or 1030 for gfx1012)and compile with that architecture.

2. In the shell you will be executing, fake the supported architecture by:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.1.0
```

Sometimes you also have to symlink to trick LLVM's openMP to play ball:

```bash
sudo ln -s libomptarget-amdgpu-gfx1010.bc /usr/lib/llvm-18/lib/libomptarget-amdgpu-gfx1012.bc
sudo ln -s /usr/lib/llvm-18/lib/libomptarget-amdgpu-gfx1010.bc /usr/lib/llvm-18/lib/libomptarget-amdgpu-gfx1012.bc
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

If you expose multiple NVIDIA GPUs at once, certain LLVM/OpenMP offload builds in
this environment can report zero devices or fail to initialize correctly.
That looks like a `libomptarget`/runtime quirk rather than a Bit issue, because
the same binary works when each GPU is tested individually. In practice, the
safe workaround is to run one GPU at a time.
On the other hand, gcc is more than happy to work with more than 1 NVIDIA GPUs, but architecture support and use of thread local memory via OpenMP leaves much to be desired.

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
make bug_report CC=gcc GPU=AMD GPU_ARCH=gfx900 BUG_TARGET=bench_omp

# Clang AMD offload bug report
make bug_report CC=clang GPU=AMD GPU_ARCH=gfx90a BUG_TARGET=bench_omp

# Clang NVIDIA offload bug report with explicit SM target
make bug_report CC=clang GPU=NVIDIA GPU_ARCH=sm_70 BUG_TARGET=bench_omp
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


### Script helpers

This repository includes helper scripts under `scripts/` for compiler setup and
branch management.

- `scripts/generate_bug_report.sh` is the backend used by `make bug_report`. It
  captures the build log, compiler configuration, preprocessing output, and
  optional crash/backtrace information for a reproducible bug report.
- `scripts/push_gpuOpt_to_main.sh` cherry-picks a curated set of GPU-related
  files from the `gpuOpt` branch into `main`, commits them, and pushes `main` to
  the remote repository. It requires a clean working tree and that you run it
  from `gpuOpt`.
- `scripts/push_gpuOpt_to_inteliGPU.sh` merges `gpuOpt` into `inteliGPU`. It
  also requires a clean working tree and must be run from `gpuOpt`.

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
make bench_omp CC=clang GPU=AMD GPU_ARCH=gfx90a  # AMD offload
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

The containerized operations in the CPU are approximately twice as fast as the OpenMP accelerated equivalent non containerized operations for long bitsets because of the memory locality property. GPU acceleration is also considerable but the actual mileage may vary according to the OpenMP kernel execution strategies. The CPU-only benchmark (`openmo_bit_nogpu`) omits entirely the GPU benchmarks.

#### CPU container-kernel tuning sweep

`scripts/sweep_cpu_tuning.pl` automates CPU tuning of the containerized
intersection-count kernel. For each configuration it performs a clean rebuild,
runs the focused `openmp_bit_container` benchmark with a fixed CPU affinity,
and collects a `perf stat` profile. It is intended for comparing the CPU tile,
K block, outer-product microkernel shape, and algorithm-specific unrolling or
scratch-buffer choices without timing unrelated benchmark work.

Build the CPU benchmark target once to verify that the toolchain works, then
run the script from the repository root:

```bash
make bench_omp GPU=NONE CC=clang

ELEVATE=always CORES=0-9 REPS=5 PERF_REPS=3 ./scripts/sweep_cpu_tuning.pl
```

The default sweep evaluates the direct SIMD implementation (`LIBPOPCNT=0`). To
compare it with the independent libpopcnt scratch-buffer implementation, include
both modes explicitly:

```bash
LIBPOPCNT_MODES=0,1 ELEVATE=always ./scripts/sweep_cpu_tuning.pl
```

To exhaustively evaluate the current 10-core i9-7900X machine, including both
algorithms, every default tuning parameter, and every diagnostic perf profile,
run the following from the repository root. This is a long-running measurement:
448 build configurations and 4,928 profiled benchmark invocations.

```bash
LIBPOPCNT_MODES=0,1 \
CORES=0-9 THREADS=10 \
REPS=5 PERF_REPS=3 \
PERF_PROFILES=summary,cache-l1,cache-l2,cache-l3-dram,cache-stalls,buffers-pending,buffers-store,execution-uops,execution-ports,frontend,frequency \
ELEVATE=always \
./scripts/sweep_cpu_tuning.pl
```

Start with a small trial when changing machines or counter sets:

```bash
MAX_CONFIGS=2 REPS=1 PERF_REPS=1 ELEVATE=always ./scripts/sweep_cpu_tuning.pl
```

### NUMA CPU tuning sweeps

On a multi-socket NUMA machine, a process affinity mask alone does not choose
where its memory pages are allocated. The focused benchmark initializes shared
input containers before its OpenMP workers begin, so Linux's default first-touch
policy can place most input memory on one node. This can cause workers on the
other socket to repeatedly access remote memory.

Use `scripts/run_numa_sweeps.sh` to run four comparable full tuning sweeps on a
dual-socket Xeon E5-2697 v4 topology with 18 physical cores per socket. It
requires `numactl`, runs the sweeps sequentially, and uses only the physical
cores numbered `0-35` (not their SMT siblings). Run it from the repository root:

```bash
./scripts/run_numa_sweeps.sh
```

The runner performs these experiments:

1. `socket0-local`: CPUs `0-17`, 18 OpenMP threads, and memory bound to NUMA node 0.
2. `socket1-local`: CPUs `18-35`, 18 OpenMP threads, and memory bound to NUMA node 1.
3. `dual-first-touch-spread`: CPUs `0-35`, 36 OpenMP threads, OpenMP workers spread across cores, and Linux's default first-touch memory policy.
4. `dual-interleave`: CPUs `0-35`, 36 OpenMP threads, OpenMP workers spread across cores, and allocations interleaved across NUMA nodes 0 and 1.

The single-socket runs establish local-memory baselines. Comparing the two
dual-socket runs shows whether interleaving reduces an asymmetric first-touch
placement effect. Interleaving balances allocation across nodes; it does not
make every memory access local.

Before using this runner on another NUMA machine, inspect its topology and edit
the CPU lists, thread counts, NUMA-node IDs, and `ARCH_TAG` in the script to
match it:

```bash
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE
```

Each sweep sets `OMP_PLACES=cores` and an explicit `OMP_PROC_BIND` policy. The
tuning script forwards these settings to the benchmark even when `perf` is run
through `sudo`. Every report records the requested NUMA and OpenMP policies.

Each run publishes compact, architecture-labelled summaries such as
`tuning-results/summary-x86-64-intel-core-i9-7900x-<timestamp>.csv` and
`tuning-results/llm-summary-x86-64-intel-core-i9-7900x-<timestamp>.md`. Set
`RUN_LABEL` to include an experiment identifier between the architecture and
timestamp, for example `...-dual-socket-interleave-<timestamp>.md`. These
files are intended to be committed, allowing GitHub to retain results from
multiple machines. Per-configuration build, benchmark, and perf logs remain
local in `tuning-results/.work/<architecture>-<timestamp>/` by default and are
ignored by Git. The Markdown summary is ranked by average elapsed time and is
designed to be supplied directly to an LLM. The script runs `make distclean`
before every configuration, so do not keep required uncommitted build artifacts
in `build/` while it is running.

All controls are environment variables. Comma-separated values define a sweep;
single values hold that parameter fixed.

| Variable | Default | Description |
|---|---|---|
| `LIBPOPCNT_MODES` | `0` | Algorithms to compare: `0` is the direct SIMD kernel and `1` is the libpopcnt scratch-buffer path. |
| `CPU_TILES` | `4,8,16,32` | CPU database tile sizes compiled as `CPU_TILE`. |
| `K_BLOCKS` | `256,512,768,1024` | K-dimension block sizes compiled as `BITVECTOR_TILE`. |
| `SHAPES` | `1x1,2x2,2x4,4x2` | Outer microkernel shapes, written as `ROWSxCOLS`. |
| `UNROLLS` | `1,2,4` | `OUTER_VEC_BLK` values; swept only for mode `0`. |
| `BUFFER_SIZES` | `16,32,64,128` | `BUFFER_SIZE` values; swept only for mode `1`. |
| `CC` | `clang` | Compiler supplied to `make`. |
| `CORES` | `0-9` | CPU list passed to `taskset -c`. Match this to physical cores where possible. |
| `BITS`, `LEFT`, `RIGHT` | `65536`, `10240`, `1024` | Bitset length and left/right container counts passed to `openmp_bit_container`. |
| `THREADS`, `REPS` | `10`, `5` | OpenMP thread count and timed benchmark repetitions per invocation. |
| `PERF_REPS` | `3` | Repetitions requested from `perf stat` for each profile and configuration. |
| `PERF_PROFILES` | summary, cache L1/L2/L3/DRAM/stalls, fill/store buffers, execution uops/ports, front end, frequency | Comma-separated diagnostic profiles. Each profile is a separate, deliberately small `perf stat` event group to avoid PMU multiplexing. Use `PERF_PROFILES=summary` for a faster ranking-only sweep. |
| `PERF_EVENTS` | summary event group | Optional comma-separated replacement event list for the `summary` profile. It preserves compatibility with custom counter sets. |
| `ELEVATE` | `auto` | `never` avoids `sudo`; `auto` uses a cached noninteractive sudo credential when available; `always` obtains a sudo credential once, then reuses it for every profiled run. |
| `PRIORITY` | `nice` | Process scheduling policy: `normal`, `nice` (nice level `-20`), or `rr` (real-time round-robin priority 50). `nice` and `rr` need elevation. |
| `MAX_CONFIGS` | `0` | Stop after this many configurations; `0` means no limit. |
| `ARCH_TAG` | detected architecture and CPU model | Optional safe filename label for published summaries; use it to distinguish otherwise similar systems or non-Linux CPU descriptions. |
| `RUN_LABEL` | unset | Optional safe experiment label inserted after `ARCH_TAG` in report, CSV, and raw-artifact names. |
| `NUMA_POLICY` | `default OS policy` | Descriptive NUMA-policy text recorded in the Markdown report; apply the actual policy by launching the sweep through `numactl`. |
| `RESULTS_DIR` | `tuning-results` | Directory for compact, commit-ready `summary-<architecture>-<timestamp>.csv` and `llm-summary-<architecture>-<timestamp>.md` results. |
| `OUT_DIR` | `tuning-results/.work/<architecture>-<timestamp>` | Local directory for per-configuration build, benchmark, and perf logs. This is ignored by Git by default. |

The default direct-SIMD sweep contains 192 configurations; its eleven default
perf profiles therefore execute 2,112 profiled benchmark invocations. Enabling
both algorithms produces 448 configurations, so a full diagnostic run can take
substantial time. Use `PERF_PROFILES=summary` while narrowing the tuning space,
then run the complete profile set on the best candidates. `perf` access is controlled by the host's
`kernel.perf_event_paranoid` setting; use `ELEVATE=always` where permitted or
adjust that policy according to local system-administration requirements.

The repository [benchmarking-bits](https://github.com/chrisarg/benchmarking-bits) 
contains benchmarks against other bitset/bitvector/bitmaps in C and Perl.

### OpenMP Parallel Region/Worksharing strategies in CPU and GPU

The CPU OpenMP implementation is a tiled implementation of a collapsed `omp parallel for` region that attempts to squeeze as much performance as possible by exploiting memory alignment (or lack thereof) of the containerized buffers. This is an enhancement over the very first implementation of the OpenMP code that did not use tiling. The code as is, is similar to the `SHARED_TILE_ILP` experimental GPU kernel in which both containers are presented to the algorithm in their untransposed version. In the present implementation the GPU kernel follows the `TEAM_PARALLEL_SIMD` algorithm (see the gpuOpt branch for details of this and other algorithms that are currently being evaluated). Currently I am evaluating numerous alternative approaches to see how much OpenMP can be pushed to deliver performance comparable to native CUDA and HIP implementations. Internally these algorithms are implemented via highly structured, modular preprocessor macros, so extension is fairly straightforward.

### Working with the gpuOpt branch

This branch exposes an additional makefile (`Makefile_bench.mak`) that is used in active development of native CUDA/HIP code that may at some point replace the OpenMP code.  The CUDA/HIP sections are being developed with heavy AI assist, so if you end up using, you are at the mercy of the clankers (mostly Raptor mini, Gemini Flash with the occasional Grok). The branch also contains a GPU only target that is being used to experiment with different offload kernels that are used in the library.

The GPU-only benchmark (`openmp_bit_nocpu`) runs only containerized GPU
offloaded intersection counts. Its 4th argument uses the same CLI position as
`max threads`, but is interpreted as the number of GPU iterations.
This is a safe harbor for testing various implementations of GPU code (e.g. 
OpenMP directives, or popcount algorithms) without messing with `bit.c`.

This is how you use the the gpuOpt branch specific makefile, `Makefile_bench.mak`:

```bash
# GPU-only OpenMP benchmark and native CUDA/HIP benchmarks (Makefile_bench.mak)
make -f Makefile_bench.mak openmp_bit_nocpu CC=clang GPU=NVIDIA GPU_ARCH=sm_70
make -f Makefile_bench.mak openmp_bit_nocpu CC=clang GPU=AMD GPU_ARCH=gfx90a
make -f Makefile_bench.mak cuda_gpu_bench GPU=NVIDIA GPU_ARCH=sm_70
make -f Makefile_bench.mak hip_gpu_bench GPU=AMD GPU_ARCH=gfx90a

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

#### OpenMP Kernel Execution Strategies - GPU

When compiling device benchmarks in this branch, you can dictate the parallelization and memory layout technique utilized by OpenMP. Pass the OPENMP_GPU_IMPL variable to swap between bitwise optimization strategies:

```bash
# Default strategy utilizing team parallel SIMD execution
make CC=clang GPU=NVIDIA OPENMP_GPU_IMPL=TEAM_PARALLEL_SIMD

# Transposed variant of team parallel SIMD processing
make CC=clang GPU=NVIDIA OPENMP_GPU_IMPL=TRANSPOSED_TEAM_PARALLEL_SIMD

# Strategy optimized for shared tile Instruction-Level Parallelism
make CC=clang GPU=NVIDIA OPENMP_GPU_IMPL=SHARED_TILE_ILP
```

The `TEAM_PARALLEL_SIMD` utilizes all three levels of parallelization that OpenMP uses to map  concepts from the OpenMP CPU world to the GPU universe. GCC seems to like this strategy especially for NVIDIA cards.
The `TRANPOSED_TEAM_PARALLEL_SIMD` is effectively the same algorithm, but "transposes" the reference bitsets before carrying out the same operations. While this approach will generate less performant code with gcc, code generated via clang will often boost performance by 200% or more relative to `TEAM_PARALLEL_SIMD` in the same cards. See the section about Concurrency Safety regarding the "mechanics" and potential for racing conditions with this approach.
`SHARED_TILE_ILP` uses tiling and instruction level parallelism, ILP (this is also approach used by the CUDA and HIP backends in the gpuOpt branch) within OpenMP. When computing the population count after a bitlevel operation (e.g. AND or XOR) in the GPU, one faces similar challenges as writing matrix multiplication kernels in the GPU. By manipulating the size of the tiles,  the extent of the ILP and thread local memory one can squeeze extremely high level of performance from the native code, and this performance seems to transplate to the OpenMP world as well. Unfortunately this kernel will not generate correct results with gcc (v 12,13), though it is the most performant method when clang is used to build the library. 

## Usage Example

For those of you who (like me) are dazzled by C:

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

Keep reading for a deeper overview of the library's API.


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
first function call will be N \* M (i.e. a two-dimensional array stored in
row-major order). Similarly, the space that must be pre-allocated to hold the
results will need to be of size N \* M \* sizeof(int) bytes.

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

## Concurency safety in the CPU and GPU

The library has the aspiration to eventually be absolutely safe for concurency operations, but the implementation only partially lives up to the aspiration. 

1. For individual Bitsets, you the user are absoluterly responsible for ensuring concurency safety as with any buffer in a C program. People can play with atomics, native C thread or OpenMP, but at the end of the day you are only as safe as the context of use of `Bit`. The section "rationale for multi-threaded CPU and GPU deployments" provides one such implementation using OpenMP, but I encourage you to explore others e.g. tasks. 
2. For containerized operations in the CPU, OpenMP provides some form of concurency safety assuming you utilize the OpenMP functionality of containers from a single (main) thread: the fork-join model will allow you to use threads and cores of your processor in a safe manner, but the parent thread one should only be one. I have not for example tried to use these containerized operations within tasks, but I welcome you to try and share the results! Another area with the potential of reward, but also pain is the use of multi-processing to _fork_ distinct processes that then utilize containerized operations. This may seem like a weird thing to do, but it may be absolutely necessary if one were to squeeze the last drop of compute power in multi-socket systems,  if one were to use the library to go through large amounts of work when the scaling of the OpenMP containerized operations starts dropping off, or for parallel scripting with the Perl interface. In that case be aware that you must pause OpenMP  e.g. by including `omp_pause_resource_all(omp_pause_hard);` in your C code before forking.
3. Containerized operations in the GPU pause an interesting dilemna. Currently the library allows the launching of a large number of GPU threads from a single (blocking) CPU thread. There is no asynchronous communication and no opportunity to use devices for offloading from the controlling CPU thread (though one could _fork_ multiple processes and offload to multiple CPUs if they are present in the system). The infrastructure for creating multithread safe applications across CPU and GPU is present (i.e. the relevant data structures have been created), but the functionality has not been implemented. This infrastructure is currently used to ensure that buffers that must be transposed in certain GPU algorithms are transposed only once during the execution of a single task in the experimental gpuOpt branch, but are currently not used in the main branch. 

## Libraries Used

This project incorporates other open-source libraries:

- **libpopcnt**: A C/C++ library for counting the number of 1 bits (bit
  population count) specialized for different CPU architectures. Licensed under
  BSD 2-Clause.
  https://github.com/kimwalisch/libpopcnt

- **SIMDe**: The SIMDe header-only library provides fast, portable implementations of SIMD intrinsics on hardware which doesn't natively support them, such as calling SSE functions on ARM. There is no performance penalty if the hardware supports the native implementation (e.g., SSE/AVX runs at full speed on x86, NEON on ARM, etc.).
https://github.com/simd-everywhere/simde

## Libraries that inspired this project

All the libraries below inspired me to dive in the C preprocessor, learn new stuff or simply structure my code in a better manner.

- **sse-popcount**: The SIMD population count implementation of the Harley-Seal
  algorithm based on the paper "Faster Population Counts using AVX2
  Instructions" by Daniel Lemire, Nathan Kurz and Wojciech Mula.
  https://github.com/WojciechMula/sse-popcount

- **cii** : The Bit_T library in C interfaces and implementations by David
  Hanson
  https://github.com/drh/cii

## Performance

The library `libpopcnt` is optimized for performance, with specialized implementations of
the _population count_ for different CPU architectures:

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
the straightforward translation into highly efficient GPU code (under -O3). Native GPU popcount instructions do exist and are 2.5x faster ONLY is one can stage their data to operate in registers or in shared (thread local) memory. For naive implementations of the container operations (such as mine!), performance is memory bound so it makes absolutely no practical difference is one is using the builtin directive or the WWG function in the GPU.

## A note about the rationale for multi-threaded CPU and GPU deployments

The non-containerized bitset operations can be very easily parallelized on the CPU using OpenMP or
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

The containerized versions not only fully leverage the capabilities of OpenMP to generate
code for either CPU or GPU environments, but go a step further by exploiting memory locality to deliver even higher performance. The icing on the cake is GPU offloading which is entirely opt-in: the default
build (`GPU=NONE`) routes all GPU calls to CPU implementations. Pass `GPU=NVIDIA` or `GPU=AMD` to enable device offloading as we discussed previously.Those who are interested in the implementation feel free to look into code of
`bit.c`. I found the C preprocessor to be a valuable tool for managing the complexity of the codebase and enabling code reuse. As the OpenMP itself uses
#pragma directive for parallel regions in both CPU and GPU, parameterizing these directives, required the liberal use of the `_Pragma` operator to construct
#pragma directives from macro expansions. At some unspecified point in thefuture, these and possibly other macros may be split into a header only library
to manage the expressive complexity of OpenMP for beginners.


## Applications

Bit is particularly useful for:

- Bioinformatics and genomic data processing (k-mer encoding)
- Network packet filtering and bloom filters
- Dense data representation (for sparse bitsets over large domains, one is
  probably better off exploring sparse representations e.g. roaring bitsets)
- High-performance set operations

## TO-DO

- Build the entire library in the CPU using [SIMDe](https://github.com/simd-everywhere/simde) (SIMD everywhere). This will allow portable, vectorized operations for both bitsets and containers. At that point we may no longer rely on `libpopcnt`.
- Port experimental algorithms for setop count operations from the gpuOpt branch to the main branch
- Implement additional set-op operations (e.g. the Jaccard index)
- Implement additional, OS agnostic build systems (lowest priority)
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