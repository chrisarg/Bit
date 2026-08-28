# Bit - High-Performance C Bitsets

[![License](https://img.shields.io/badge/license-BSD%202--Clause-blue)](LICENSE)

Bit is a fixed-capacity, uncompressed bitset library for C. It provides
individual bitsets, packed collections of bitsets, set operations, population
counts, and OpenMP-enabled container operations. The public interface is based
on David Hanson's `Bit_T` design and extended with packed `Bit_DB_T`
containers.

The source tree is the authoritative description of behavior. In particular,
the public declarations in `include/bit.h`, the implementations in `src/`, and
the tests in `tests/` take precedence over historical notes (because updating the README.md is always less fun than coding!).

## Contents

- [Project Background and Features](#project-background-and-features)
- [Branch Status](#branch-status)
- [Build and Test](#build-and-test)
- [GPU Troubleshooting and Validation](#gpu-troubleshooting-and-validation)
- [Using the Library](#using-the-library)
- [Public API Reference](#public-api-reference)
- [Container Counts](#container-counts)
- [Benchmarks and Experiments](#benchmarks-and-experiments)
- [Automation Scripts](#automation-scripts)
- [Constraints and Current Status](#constraints-and-current-status)
- [Design, Concurrency, and Performance Notes](#design-concurrency-and-performance-notes)
- [Dependencies, Inspiration, and Applications](#dependencies-inspiration-and-applications)
- [Roadmap](#roadmap)

## Project Background and Features

Bit began as a retype and extension of David Hanson's `Bit_T` interface from
Chapter 13 of *C Interfaces and Implementations* (Addison-Wesley,
ISBN 0-201-49841-3). The project keeps the original emphasis on a small C
interface while extending it with setop count operations, packed bitset containers,
OpenMP execution paths, and performance-oriented population counting.

I started with Hanson's deliberately small interface because it is easy to
reason about, then kept adding the things my own workloads needed: fast setop counts,
borrowed storage, batches of equally sized bitsets (effectively packed vector databases), and enough CPU/GPU
experimentation to make the preprocessor earn its keep. The result is still a
small bitset library at heart, but it now has two useful levels of abstraction:
an individual `Bit_T` and a packed `Bit_DB_T` for bulk work.

The library is intended for dense, fixed-capacity bitsets and workloads where
bitwise set operations, population counts, storage layout, and predictable
memory behavior matter. It is not a compressed or dynamically growing bitmap
library.

- **Population counting:** The bundled libpopcnt integration can use
  CPU-specific population-count implementations when enabled. The project also
  retains a portable Wilkes-Wheeler-Gill (WWG) / sideways-addition path and
  SIMD-oriented CPU code through SIMDe.
- **Set operations:** Union, intersection, symmetric difference, and set
  difference are available for individual bitsets and packed containers.
- **External storage:** Bitsets and containers can borrow caller-owned buffers
  when their storage is allocated with the size and padding required by the
  public API.
- **Packed containers:** `Bit_DB_T` stores equally sized bitsets in contiguous
  storage for all-pairs count operations on CPU or, where configured, GPU
  offload.
- **OpenMP offload:** NVIDIA, AMD, and experimental Intel paths are opt-in;
  the default `GPU=NONE` build keeps GPU-facing operations on the CPU.

The current implementation favors explicit configuration over hidden magic:
build variables select toolchains and targets, and callers remain responsible
for synchronizing concurrent mutation of the same bitset or container.

## Branch Status

This repository intentionally has branch-specific tooling. Do not assume that
every command documented below exists on every branch.

| Branch | Purpose | Notes |
| --- | --- | --- |
| `main` | Baseline library, SIMD, CPU tuning, NUMA experiments, and shared benchmark work | Owns the CPU sweep/tuning scripts and the `main`-to-branch synchronization helpers. |
| `gpuOpt` | GPU/offload kernel and comparative benchmark work | Owns `Makefile_bench.mak`, GPU-only and native CUDA/HIP benchmarks, GPU sweep/plot tooling and results, FAISS comparisons, and the `gpuOpt`-to-branch synchronization helpers. |
| `inteliGPU` | Intel oneAPI CPU build and offload validation | Build with `CC=icx GPU=INTEL`. Its `scripts/` directory retains only the shared bug-report helper after branch-specific cleanup. |

Identify the checked-out branch, source revision, and working-tree state with:

```bash
git branch --show-current
git rev-parse --short HEAD
git status -sb
```

`git branch --show-current` prints nothing for a detached `HEAD`; in that case,
use the commit printed by `git rev-parse --short HEAD` as the source revision.

## Build and Test

### Requirements

- A C compiler supported by the current `Makefile`: `clang`, `gcc`,
  `amdclang`, or `icx`.
- GNU Make.
- OpenMP support from the selected compiler.
- CUDA and an OpenMP offload-capable LLVM toolchain for NVIDIA offload.
- ROCm and a compatible LLVM/ROCm stack for AMD offload.
- Intel oneAPI `icx` for experimental Intel OpenMP offload.
- `nvcc` for the experimental CUDA benchmark and `hipcc` for the experimental
  HIP benchmark.

Clone and build the default CPU configuration:

```bash
git clone https://github.com/chrisarg/Bit.git
cd Bit

make clean
make GPU=NONE
```

The default configuration is `GPU=NONE`. GPU-facing container calls use their
CPU implementations in that configuration.

The `test` target builds `build/test_bit`; it does not execute it. Build and
run it explicitly:

```bash
make test GPU=NONE
./build/test_bit
```

### Offload Builds

The primary GPU selection is `GPU=`. NVIDIA architectures use `sm_` or
`compute_` prefixes; AMD architectures use `gfx` prefixes. The current
Makefile has no equivalent `GPU_ARCH` selector for Intel targets.

```bash
# NVIDIA OpenMP offload. Omit GPU_ARCH only when nvidia-smi can detect a target.
make CC=clang GPU=NVIDIA GPU_ARCH=sm_70

# AMD OpenMP offload.
make CC=clang GPU=AMD GPU_ARCH=gfx90a

# Experimental Intel OpenMP offload.
make CC=icx GPU=INTEL
```

`GPU=INTEL` requires `CC=icx`; `CC=amdclang` requires `GPU=AMD`; and `GPU=NONE`
cannot be combined with an active offload target.

Validate an offload configuration with `test_offload`. Setting
`OMP_TARGET_OFFLOAD=MANDATORY` prevents an accidental host fallback from being
reported as a successful device test.

```bash
make test_offload CC=clang GPU=NVIDIA GPU_ARCH=sm_70
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0

make test_offload CC=clang GPU=AMD GPU_ARCH=gfx90a
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0

make test_offload CC=icx GPU=INTEL
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0
```

The Intel command is an experimental compatibility check, not a performance
claim. Confirm it on the target hardware before depending on it.

### Build Configuration

These are Make variables, not runtime environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `GPU` | `NONE` | CPU fallback, or `NVIDIA`, `AMD`, and experimental `INTEL` offload. |
| `GPU_ARCH` | Auto-detected when possible | NVIDIA `sm_`/`compute_` and AMD `gfx` architecture list. |
| `LIBPOPCNT` | `1` | Enables bundled libpopcnt integration; set `LIBPOPCNT=0` to disable it. |
| `SIMD_DIAGNOSTICS` | `0` | Enables SIMD configuration diagnostics. |
| `APPLY_LTO` | `1` | Enables LTO for supported compilers; set to `0` to disable it. |
| `CLANG_RUNTIME_RPATH` | `1` | Embeds the selected Clang OpenMP runtime path; set to `0` only when deliberately testing another runtime. |
| `USE_BUILTIN_POPCOUNT` | `0` | Requests built-in GPU popcount instead of the default software implementation. |

### Compiler and GPU Target Matrix

The standard `Makefile` builds the library and ordinary benchmarks on `main`
and the specialized branches. The rightmost column below is `gpuOpt`-only: its
GPU-only and native targets require `make -f Makefile_bench.mak`.

| Compiler (`CC=`) | GPU target (`GPU=`) | Standard targets | Standard OpenMP/offload checks | `gpuOpt` experimental benchmark targets |
| --- | --- | --- | --- | --- |
| `gcc` or `clang` | `NONE` | library, `test`, `bench`, `bench_omp`, `bug_report` | `test_offload` builds but detects host fallback; `bench_omp` is CPU-only | none |
| `gcc` or `clang` | `NVIDIA` | library, tests, benchmarks, bug reports | `test_offload`, `bench_omp` | `openmp_bit_nocpu`, `cuda_gpu_bench`, `gpu_bench_csv` |
| `gcc` or `clang` | `AMD` | library, tests, benchmarks, bug reports | `test_offload`, `bench_omp` | `openmp_bit_nocpu`, `hip_gpu_bench`, `gpu_bench_csv` |
| `gcc` or `clang` | `NVIDIA,AMD` | library, tests, benchmarks, bug reports | `test_offload`, `bench_omp` | `openmp_bit_nocpu`, CUDA, HIP, and CSV runner targets |
| `amdclang` | `AMD` | library, tests, benchmarks, bug reports | `test_offload`, `bench_omp` | `openmp_bit_nocpu`, HIP, and CSV runner targets |
| `icx` | `INTEL` | library, tests, benchmarks, bug reports | experimental `test_offload` and `bench_omp` | experimental `openmp_bit_nocpu`; no native CUDA/HIP backend |

The Makefile rejects `CC=amdclang` with a GPU target other than `AMD`,
`CC=icx` with a GPU target other than `INTEL`, and combinations such as
`GPU=NONE,NVIDIA`.

Native CUDA and HIP benchmarks deliberately compile their device source with
`nvcc` and `hipcc`, respectively, rather than the value passed through `CC`.
They still use `GPU=NVIDIA` or `GPU=AMD` as build guards.

On `gpuOpt`, `openmp_bit_nocpu` is blocked when `GPU=NONE`. Its Makefile guard
tests whether a non-`NONE` target was selected; validate the experimental Intel
path with `test_offload` on the target machine.

### GPU Troubleshooting and Validation

GPU offload is opt-in, and OpenMP can fall back to the host when an image,
plugin, driver, or device is unavailable. Verify the target with
`test_offload` before moving on to the library's container kernels.

```text
build/test_offload <problem_size> [device_id] [benchmark_iterations]
```

The test first verifies integer, float, and double target calculations against
host results. When `benchmark_iterations` is positive, it also reports
memory-bound, compute-heavy, and device-resident benchmark modes. The
device-resident mode transfers its working data once and is useful for
separating steady-state device computation from host/device transfer overhead.

Use `OMP_TARGET_OFFLOAD=MANDATORY` for validation so an unintended host fallback
fails visibly:

```bash
# Correctness checks only.
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0

# Correctness checks plus 100 benchmark iterations.
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0 100
```

The executable prints the detected target-device count, default device, and
whether each probed target region ran on the host or a device. If no devices are
reported or a target region runs on the initial device, rebuild for the intended
architecture, verify the selected runtime plugin, and rerun with diagnostics.

#### Runtime Diagnostics

The current Makefile exports quiet Clang runtime defaults:

- `LIBOMPTARGET_INFO=0`
- `LIBOMPTARGET_DEBUG=0`

Override them while diagnosing device discovery, image loading, or launch
behavior:

```bash
LIBOMPTARGET_INFO=16 LIBOMPTARGET_DEBUG=1 \
  make test_offload CC=clang GPU=AMD GPU_ARCH=gfx90a

OMP_TARGET_OFFLOAD=MANDATORY LIBOMPTARGET_INFO=16 \
  ./build/test_offload 100000 0
```

#### NVIDIA Offload Notes

NVIDIA builds accept `sm_<target>` or `compute_<target>` values. The Makefile
can derive `sm_` values from `nvidia-smi` when `GPU_ARCH` is not supplied, but
an explicit target is usually easier to reproduce:

```bash
make test_offload CC=clang GPU=NVIDIA GPU_ARCH=sm_70
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 100000 0

# Build multiple NVIDIA images when the installed toolchain supports them.
make test_offload CC=clang GPU=NVIDIA GPU_ARCH=sm_70,sm_80
```

On a multi-GPU host, restrict visible devices before running the test. The
selected physical device normally becomes OpenMP logical device `0`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_TARGET_OFFLOAD=MANDATORY \
  ./build/test_offload 1000000 0 1

CUDA_VISIBLE_DEVICES=1 OMP_TARGET_OFFLOAD=MANDATORY \
  ./build/test_offload 1000000 0 1
```

Some LLVM/libomptarget and driver combinations behave more reliably with one
visible NVIDIA device at a time. If a multi-device run reports zero target
devices or fails during initialization, narrow `CUDA_VISIBLE_DEVICES`, keep the
program device ID at `0`, and rerun the mandatory-offload check.

#### AMD Offload Notes

AMD targets use the `gfx<target>` spelling. The Makefile can query `rocm-smi`
when `GPU_ARCH` is omitted; inspect the system directly before pinning a target:

```bash
rocminfo | grep -Eo 'gfx[0-9a-f]+' | sort -u
rocm-smi --showproductname

make clean
make test_offload CC=clang GPU=AMD GPU_ARCH=gfx90a
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 4096 0
```

If several AMD devices are visible or the OpenMP runtime selects the wrong one,
restrict visibility and rerun the same mandatory-offload test:

```bash
ROCR_VISIBLE_DEVICES=0 OMP_TARGET_OFFLOAD=MANDATORY \
  ./build/test_offload 4096 0

# Alternative ROCr visibility variable used by some installations.
HSA_VISIBLE_DEVICES=0 OMP_TARGET_OFFLOAD=MANDATORY \
  ./build/test_offload 4096 0
```

If the runtime cannot load an AMD target image, first make `GPU_ARCH` match the
actual `gfx` target and the installed LLVM/ROCm stack. The Makefile exposes
`ROCM_PATH` and `ROCM_DEVICE_LIB_PATH` for non-default installations; inspect
those paths before changing system libraries.

##### Legacy AMD Architecture Workaround

This historical recipe was used for a Radeon Pro W5500 (`gfx1012`) with LLVM
18. It compiles for the nearby `gfx1010` target and presents that target to the
runtime for the current shell. Treat it as a record of one working environment that will allow you to repurpose a cheap GPU for real work,
then verify your own setup with `OMP_TARGET_OFFLOAD=MANDATORY`.

```bash
# Historical example: compile a nearby supported target, then present that
# target to the runtime for this shell only.
make test_offload CC=clang GPU=AMD GPU_ARCH=gfx1010
export HSA_OVERRIDE_GFX_VERSION=10.1.0
OMP_TARGET_OFFLOAD=MANDATORY ./build/test_offload 4096 0
```

Some LLVM 18 installations also looked for a matching
`libomptarget-amdgpu-*.bc` filename for the physical target. System-wide aliases
change the compiler installation, so inspect the active toolchain first and
make that change only with an administrator and a rollback plan:

```bash
find "$ROCM_DEVICE_LIB_PATH" -maxdepth 1 -name 'libomptarget-amdgpu-*.bc' -print
```

Prefer a ROCm/LLVM release that supports the actual target. If an alias is used,
record it and retest after compiler, runtime, or driver updates.

### Compiler Bug Reports

`make bug_report` creates a timestamped directory under `bug_reports/` with
the build log, configuration, preprocessed source, and, for a failing build,
an attempted backtrace. The target defaults to `BUG_TARGET=bench_omp`.

```bash
make bug_report CC=clang GPU=NVIDIA GPU_ARCH=sm_70 BUG_TARGET=bench_omp
```

Review `build.log`, `config.txt`, `backtrace.txt`, and
`src-bit.preprocessed.i` in the generated report directory.

The report script records the selected compiler, target list, effective flags,
and the failing build output. It also adds compiler-specific reproduction
artifacts:

- GCC: `gcc-v.txt`, `gcc-repro-command.txt`, and `gcc-save-temps.log`.
- Clang: compiler crash reproducers when Clang emits them under the configured
  report directory.

If the target build fails and `gdb` is available, the script reruns the target
under batch `gdb` and writes a full backtrace to `backtrace.txt`. When the build
succeeds, that file explicitly records that no failing process was available.
Temporary preprocessing and intermediate compiler artifacts in `build/` are
removed after collection.

## Using the Library

Include `bit.h` and link against `build/libbit.so` or `build/libbit.a` after
building the library. `Bit_T` and `Bit_DB_T` are opaque handles.

Bitsets have fixed capacity. Valid bit indexes are in the range
`0 .. Bit_length(bitset) - 1`; use `Bit_buffer_size(length)` when allocating
external storage.

### Public API Reference

The declarations live in `include/bit.h`; this section is the practical map of
the interface. `Bit_T` and `Bit_DB_T` are opaque handles, so applications work
through these functions rather than depending on their private layouts.

| Public type | Purpose |
| --- | --- |
| `Bit_T` | One fixed-capacity mutable bitset. |
| `Bit_DB_T` | A packed collection of equally sized bitsets. |
| `SETOP_COUNT_OPTS` | CPU thread count plus GPU device-residency controls for all-pairs container counts. |

#### Individual Bitset API

| Family | Functions | Contract |
| --- | --- | --- |
| Lifecycle and storage | `Bit_new`, `Bit_load`, `Bit_free`, `Bit_extract`, `Bit_buffer_size` | Create library-owned storage, borrow caller storage, release the wrapper, or copy bytes out. |
| Properties | `Bit_length`, `Bit_count` | Return fixed capacity or population count. |
| Single-bit mutation | `Bit_bset`, `Bit_bclear`, `Bit_get`, `Bit_put` | Access one indexed bit; `Bit_put` returns its previous value. |
| Bulk/range mutation | `Bit_aset`, `Bit_aclear`, `Bit_set`, `Bit_clear`, `Bit_not` | Apply an index array or an inclusive `[lo, hi]` range. |
| Callback traversal | `Bit_map` | Visit indexes from left to right with the current bit value and caller closure. |
| Comparisons | `Bit_eq`, `Bit_leq`, `Bit_lt` | Compare equal-length bitsets. |
| Allocating set operations | `Bit_union`, `Bit_inter`, `Bit_diff`, `Bit_minus` | Return a newly allocated result that must be passed to `Bit_free`. |
| Count-only set operations | `Bit_union_count`, `Bit_inter_count`, `Bit_diff_count`, `Bit_minus_count` | Return the result population count without constructing a bitset. |

Set-operation names follow the implementation and tests:

| Operation | Expression | Example for $A=\{1,3,5\}$ and $B=\{3,5,7\}$ |
| --- | --- | --- |
| `union` | $A \mathbin{\mathrm{OR}} B$ | $\{1,3,5,7\}$ |
| `inter` | $A \mathbin{\mathrm{AND}} B$ | $\{3,5\}$ |
| `diff` | $A \mathbin{\mathrm{XOR}} B$ | $\{1,7\}$ |
| `minus` | $A \mathbin{\mathrm{AND\mbox{-}NOT}} B$ | $\{1\}$ |

For these individual-bitset set operations, one NULL operand is interpreted as
the empty set. Thus `Bit_union(set, NULL)` and `Bit_minus(set, NULL)` return a
copy of `set`, while `Bit_inter(set, NULL)` returns an empty bitset. Passing
both operands as NULL is invalid.

#### Packed Container API

| Family | Functions | Contract |
| --- | --- | --- |
| Lifecycle and storage | `BitDB_new`, `BitDB_load`, `BitDB_free` | Create or borrow storage for a fixed number of equal-length bitsets. |
| Properties and counts | `BitDB_length`, `BitDB_nelem`, `BitDB_count_at`, `BitDB_count` | Query shape or population counts; `BitDB_count` allocates an array the caller frees. |
| Element access | `BitDB_get_from`, `BitDB_put_at` | Copy one element out as a new `Bit_T`, or copy one equal-length `Bit_T` into the container. |
| Buffer access | `BitDB_extract_from`, `BitDB_replace_at` | Copy one element to or from a caller-provided buffer. |
| Clearing | `BitDB_clear_at`, `BitDB_clear` | Clear one element or the complete packed container. |
| Allocating all-pairs counts | `BitDB_{inter,union,diff,minus}_count_{cpu,gpu}` | Allocate and return an `int` matrix; the caller uses `free`. |
| Caller-owned all-pairs counts | `BitDB_{inter,union,diff,minus}_count_store_{cpu,gpu}` | Write into a caller-provided `int` matrix. |
| Target convenience macros | `BitDB_{inter,union,diff,minus}_count(..., cpu\|gpu)` | Select the corresponding direct CPU or GPU function in C source. |
| Build diagnostics | `print_Bit_configuration` | Print the compiled tile, buffer, popcount, and OpenMP configuration. |

Container binary operations require two non-NULL containers whose bitsets have
the same length. If the left and right containers hold $N$ and $M$ bitsets,
the result contains $N \times M$ integers in row-major order. `diff` and
`minus` retain the XOR and left AND-NOT meanings shown above.

The header also defines target-selecting `_store_` macros using the same order
as the direct functions:

```c
BitDB_inter_count_store(left, right, results, options, cpu);
```

The final token may be `cpu` or `gpu`. Direct `_store_cpu` and `_store_gpu`
functions remain useful for foreign-function interfaces and callers that cannot
use C preprocessor macros.

#### Ownership and Validation

| Value | Owner and release rule |
| --- | --- |
| `Bit_new` / `BitDB_new` result | Library owns storage. `Bit_free` / `BitDB_free` releases it, sets the handle to NULL, and returns NULL. |
| `Bit_load` / `BitDB_load` result | Caller owns storage. The matching free routine destroys the wrapper, sets the handle to NULL, and returns the borrowed pointer for the caller to free or reuse. |
| `BitDB_get_from` or an allocating `Bit_*` operation | Caller owns the new `Bit_T` and releases it with `Bit_free`. |
| `BitDB_count` or non-store container count | Caller owns the returned `int *` and releases it with `free`. |
| `_store_` container count | Caller allocates and retains the result buffer. |

The implementation uses `assert` for most pointer, index, length, allocation,
and equal-shape checks. Defining `NDEBUG` removes those checks; it does not turn
an undersized external buffer or invalid index into a recoverable error. In
particular, the library cannot determine the allocation size behind a raw
pointer, so callers must size borrowed and extraction buffers correctly.

### Individual Bitsets

```c
#include "bit.h"
#include <stdio.h>

int main(void) {
  Bit_T left = Bit_new(128);
  Bit_T right = Bit_new(128);
  Bit_bset(left, 3);
  Bit_bset(left, 64);
  Bit_bset(right, 64);
  Bit_bset(right, 100);

  Bit_T overlap = Bit_inter(left, right);
  printf("intersection count: %d\n", Bit_count(overlap));

  Bit_free(&overlap);
  Bit_free(&right);
  Bit_free(&left);
  return 0;
}
```

The distinction between `diff` and `minus` is easier to see with actual bits.
For $A=\{1,3,5\}$ and $B=\{3,5,7\}$, symmetric difference keeps `1` and `7`,
whereas left set difference keeps only `1`:

```c
#include "bit.h"
#include <assert.h>

int main(void) {
  Bit_T left = Bit_new(128);
  Bit_T right = Bit_new(128);
  int left_bits[] = {1, 3, 5};
  int right_bits[] = {3, 5, 7};
  Bit_aset(left, left_bits, 3);
  Bit_aset(right, right_bits, 3);

  Bit_T symmetric = Bit_diff(left, right);
  Bit_T remainder = Bit_minus(left, right);
  assert(Bit_count(symmetric) == 2);
  assert(Bit_get(symmetric, 1) && Bit_get(symmetric, 7));
  assert(Bit_count(remainder) == 1 && Bit_get(remainder, 1));

  Bit_free(&remainder);
  Bit_free(&symmetric);
  Bit_free(&right);
  Bit_free(&left);
  return 0;
}
```

The corresponding `Bit_*_count` functions compute the same population counts
without constructing result bitsets.

### External Storage

`Bit_load` and `BitDB_load` borrow caller-owned storage. The caller must
allocate enough padded storage and later free the pointer returned by the
matching free routine.

```c
#include "bit.h"
#include <stdlib.h>

int main(void) {
  const int length = 130;
  const int bytes = Bit_buffer_size(length);
  void *storage = calloc(1, (size_t)bytes);
  if (storage == NULL) {
    return 1;
  }

  Bit_T borrowed = Bit_load(length, storage);
  Bit_bset(borrowed, 129);

  /* Bit_free returns storage for externally loaded bitsets. */
  free(Bit_free(&borrowed));
  return 0;
}
```

For a bitset allocated with `Bit_new`, `Bit_free(&bitset)` frees its internal
storage and returns `NULL`. `BitDB_free` follows the same ownership rule for
packed containers. `Bit_extract` copies a bitset into caller-provided storage
and returns the number of bytes written.

Borrowed container storage is the per-bitset buffer size multiplied by the
number of elements. `BitDB_free` returns that original pointer rather than
freeing it behind the caller's back:

```c
#include "bit.h"
#include <stdlib.h>

int main(void) {
  const int length = 130;
  const int count = 4;
  const size_t bytes = (size_t)Bit_buffer_size(length) * count;
  void *storage = calloc(1, bytes);
  if (storage == NULL) {
    return 1;
  }

  Bit_DB_T borrowed = BitDB_load(length, count, storage);
  Bit_T seed = Bit_new(length);
  Bit_bset(seed, 129);
  BitDB_put_at(borrowed, 0, seed);

  Bit_free(&seed);
  storage = BitDB_free(&borrowed);
  free(storage);
  return 0;
}
```

## Container Counts

`Bit_DB_T` stores equally sized bitsets in a packed container. Create a
container with `BitDB_new(length, count)` and fill it with `BitDB_put_at`.

Container element functions copy data rather than exposing an internal
`Bit_T`. `BitDB_get_from` creates a new bitset, while extraction and replacement
use a caller-owned byte buffer:

```c
#include "bit.h"
#include <assert.h>
#include <stdlib.h>

int main(void) {
  const int length = 128;
  Bit_DB_T database = BitDB_new(length, 2);
  Bit_T seed = Bit_new(length);
  Bit_bset(seed, 9);
  BitDB_put_at(database, 0, seed);

  Bit_T copy = BitDB_get_from(database, 0);
  assert(Bit_get(copy, 9) == 1);

  void *buffer = calloc(1, (size_t)Bit_buffer_size(length));
  if (buffer == NULL) {
    Bit_free(&copy);
    Bit_free(&seed);
    BitDB_free(&database);
    return 1;
  }
  BitDB_extract_from(database, 0, buffer);
  BitDB_clear_at(database, 0);
  assert(BitDB_count_at(database, 0) == 0);
  BitDB_replace_at(database, 1, buffer);
  assert(BitDB_count_at(database, 1) == 1);

  free(buffer);
  Bit_free(&copy);
  Bit_free(&seed);
  BitDB_free(&database);
  return 0;
}
```

```c
#include "bit.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  Bit_T seed = Bit_new(128);
  Bit_bset(seed, 10);
  Bit_bset(seed, 65);

  Bit_DB_T queries = BitDB_new(128, 2);
  Bit_DB_T references = BitDB_new(128, 3);
  for (int index = 0; index < BitDB_nelem(queries); ++index) {
    BitDB_put_at(queries, index, seed);
  }
  for (int index = 0; index < BitDB_nelem(references); ++index) {
    BitDB_put_at(references, index, seed);
  }

  SETOP_COUNT_OPTS options = {.num_cpu_threads = 2};
  int *counts = BitDB_inter_count_cpu(queries, references, options);
  if (counts != NULL) {
    printf("first intersection count: %d\n", counts[0]);
  }

  free(counts);
  BitDB_free(&references);
  BitDB_free(&queries);
  Bit_free(&seed);
  return 0;
}
```

`BitDB_count(container)` returns a newly allocated array containing one
population count per stored bitset. The non-store container count functions
(`BitDB_inter_count_cpu`, `BitDB_union_count_gpu`, and so on) return a newly
allocated result array. In both cases, callers free the returned array.

The result array for a binary container operation has
`BitDB_nelem(left) * BitDB_nelem(right)` elements in row-major order:

```c
result[left_index * BitDB_nelem(right) + right_index]
```

Use `_store_` variants when the caller owns the result buffer instead:

```c
size_t result_count = (size_t)BitDB_nelem(queries) * BitDB_nelem(references);
int *results = malloc(result_count * sizeof(*results));
if (results != NULL) {
  BitDB_inter_count_store_cpu(queries, references, results, options);
  free(results);
}
```

The macros `BitDB_inter_count`, `BitDB_union_count`, `BitDB_diff_count`, and
`BitDB_minus_count` select a `cpu` or `gpu` function at compile time. Use the
function forms when linking against a shared library from code that cannot see
the macros.

`SETOP_COUNT_OPTS` separates CPU execution from advanced GPU data-residency
decisions:

| Field | Current behavior |
| --- | --- |
| `num_cpu_threads` | A positive value selects the CPU OpenMP thread count; a nonpositive value uses the OpenMP runtime maximum. |
| `device_id` | Selects the OpenMP target device for GPU calls; ignored by CPU calls. |
| `upd_1st_operand`, `upd_2nd_operand` | Refresh an operand that is already present on the selected device. An absent operand is mapped on first use regardless of the update flag. |
| `release_1st_operand`, `release_2nd_operand` | Release the corresponding device mapping after the operation. Leave false only when a later call deliberately reuses that mapping. |
| `release_counts` | Requests release of the device-side result mapping after results are returned. |
| `algorithm` | Present in the public structure, but not read by the current library dispatch. It is not a runtime kernel selector. |

A repeated-query workflow can therefore keep an unchanged reference container
mapped, refresh each modified query container, and release both operand mappings
on the final call. That optimization also creates a responsibility: if host data
changes while its update flag is false, the device is allowed to keep using the
older mapped contents. Keep the default one-call lifecycle until residency is a
measured bottleneck, then make the update/release sequence explicit in the
calling code.

Container operation names use the same set semantics as individual bitsets:
`diff` is XOR/symmetric difference and `minus` is AND-NOT/set difference.

## Benchmarks and Experiments

### Standard Benchmarks

Build the CPU benchmark suite:

```bash
make bench_omp GPU=NONE
```

For `GPU=NONE`, this creates:

- `build/openmp_bit_nogpu`
- `build/openmp_bit_container`
- `build/cpu_param_sweep`

`build/cpu_param_sweep` is the four-argument C benchmark used by the broad CPU
parameter workflow. It is distinct from `scripts/cpu_param_sweep.pl`, the
JSON-driven Perl coordinator described in [Automation Scripts](#automation-scripts).

`make bench` additionally builds `build/benchmark`.

The OpenMP benchmark command lines are defined by their source files:

```text
build/openmp_bit <size> <number-of-bitsets> <number-of-reference-bitsets> <max-threads> [gpu-id]
build/openmp_bit_nogpu <size> <number-of-bitsets> <number-of-reference-bitsets> <max-threads>
build/openmp_bit_container <bits> <left-bitsets> <right-bitsets> <threads> <repetitions>
```

With a non-`NONE` GPU target, `make bench_omp` also builds `build/openmp_bit`.
The fourth argument of `openmp_bit_nogpu` is a maximum CPU thread count.

### Benchmark Scope and Interpretation

The OpenMP benchmark is a practical all-pairs intersection-count comparison:
it searches query bitsets against a reference collection and reports the
largest observed intersection count. The mixed benchmark establishes serial
baselines, sweeps OpenMP thread counts for the ordinary bitset representation,
and repeats the workload with packed `Bit_DB_T` containers. With a configured
offload target, it also exercises container operations through the GPU path.

`openmp_bit_nogpu` keeps the serial, OpenMP, and packed-container CPU portions
while excluding GPU execution. Use it when characterizing CPU tiles, OpenMP
scheduling, affinity, and memory locality without an offload runtime in the
measurement.

The speedup values use the run's first serial measurement as their baseline.
They are useful for comparing configurations on the same machine and workload;
bitset shape, topology, compiler, OpenMP runtime, memory placement, and GPU
strategy all change the result when that context changes.

### Experimental `gpuOpt` Benchmark Layer

`Makefile_bench.mak` is an experimental extension for GPU-only OpenMP kernels
and native CUDA/HIP benchmarks. It exists only on `gpuOpt` and is not the public
library build interface. Switch branches before using any command in this
section:

```bash
git switch gpuOpt

# GPU-only OpenMP benchmark.
make -f Makefile_bench.mak openmp_bit_nocpu \
  CC=clang GPU=NVIDIA GPU_ARCH=sm_70

# Native benchmark backends.
make -f Makefile_bench.mak cuda_gpu_bench GPU=NVIDIA GPU_ARCH=sm_70
make -f Makefile_bench.mak hip_gpu_bench GPU=AMD GPU_ARCH=gfx90a

# Run the selected native backend and write CSV/log output.
make -f Makefile_bench.mak gpu_bench_csv GPU=NVIDIA GPU_ARCH=sm_70
```

The GPU-only executable accepts:

```text
build/openmp_bit_nocpu <size> <number-of-bitsets> <number-of-reference-bitsets> <gpu-iterations> [gpu-id]
```

Its fourth argument is GPU iterations, not a CPU thread count.

`OPENMP_GPU_IMPL` is a compile-time choice for the experimental GPU-only
benchmark, not a public runtime option. The active `gpuOpt` values are:

- `TEAM_PARALLEL_SIMD`
- `TRANSPOSED_TEAM_PARALLEL_SIMD`
- `SHARED_TILE_ILP`
- `TRANSPOSED_TILED_GEMM`

For example:

```bash
make -f Makefile_bench.mak openmp_bit_nocpu \
  CC=clang GPU=NVIDIA GPU_ARCH=sm_70 \
  OPENMP_GPU_IMPL=TRANSPOSED_TILED_GEMM
```

These kernels and native CUDA/HIP paths are experimental. Build for the target
compiler and architecture, confirm agreement with the CPU reference, and then
measure the workload you care about.

The strategy selector and these benchmark targets belong to `gpuOpt`; `main`
and `inteliGPU` retain only the standard Makefile build surfaces.

#### Interpreting `openmp_bit_nocpu` Output

The GPU-only benchmark is a safe place to experiment with container kernels
without changing the standard library path. It creates random query and
reference bitsets, computes a CPU reference result, performs a warm-up run, and
then reports several timing views:

- **GPU Algorithm Timing:** kernel-focused time for the intersection-count
  operation.
- **GPU Algorithm + PCIe Timings:** end-to-end device-path time, including
  staging and transfer work measured by the benchmark.
- **GPU Transpose Timings:** layout preparation time for strategies that need
  a transposed representation.
- **CPU Overhead Timings:** host-side setup, dispatch, and synchronization
  work around device execution.
- **Per-Iteration Data Movement Breakdown:** query upload and result download
  volume; the resident reference database is reported separately and excluded
  from the repeated-transfer payload.
- **Agreement/Disagreement Counts:** comparison with the CPU reference. Treat
  any disagreement warning as a correctness failure to investigate before
  interpreting throughput.
- **Estimated Throughput:** both kernel-focused and total-operation rates,
  with the latter representing the user-visible combination of staging,
  transfers, and computation.

Use **GPU Algorithm Timing** to compare kernel and layout choices. Use the
total-operation view when transfers are part of the workload, and check the
movement breakdown when a large result matrix makes download time dominant.

#### OpenMP Strategy Notes

The strategy selector changes compile-time layout and parallelization choices.
`TEAM_PARALLEL_SIMD` is the baseline team/parallel/SIMD approach;
`TRANSPOSED_TEAM_PARALLEL_SIMD` prepares a column-oriented reference layout;
`SHARED_TILE_ILP` combines shared tiles with instruction-level parallelism; and
`TRANSPOSED_TILED_GEMM` is a further experimental tiled formulation.

These names describe implementation choices, not a ranking. Earlier GCC 12/13
and Clang experiments produced different correctness and performance outcomes
for the same transposed and shared-tile structures. For each candidate, keep
the compiler, architecture, and command with the result; check CPU agreement
first, then compare timings. The GPU-only layer keeps that exploration separate
from the library's public execution path.

## Automation Scripts

### CPU Sweep Workflow (`main`)

The `main` CPU tools are complementary stages of investigation, not
interchangeable benchmark front ends:

| Stage | Tool and benchmark | Configuration | Measurements and artifacts |
| --- | --- | --- | --- |
| Broad discovery | `scripts/cpu_param_sweep.pl` -> `build/cpu_param_sweep` | JSON matrices and command-line overrides | External wall-clock timings for ordinary-bitset and packed-container paths, plus host telemetry, CSV, and raw logs under `benchmark_CPU_params/`. |
| Profiling for focused tuning | `scripts/sweep_cpu_tuning.pl` -> `build/openmp_bit_container` | Environment-variable matrix | Repeated packed-container timings, CPU affinity, and `perf stat` profiles under `tuning-results/`. |
| Dual-socket analysis | `scripts/run_numa_sweeps.sh` -> `sweep_cpu_tuning.pl` | Four comparable topology/memory-policy cases | Socket-local baselines plus first-touch and interleaved dual-socket results in the focused tuner's artifact layout. |

Use the broad sweep to find candidates across compiler, kernel, workload, and
placement choices. Use the focused profiler when you need further insights before locking the configuration for a specific architecture, then use the NUMA runner when the question is memory placement on a
dual-socket host and how this affects performance. The stages can be used independently when that is the only
question being investigated.

These scripts and their configuration live on `main`. The benchmark sources
and Makefile targets may also exist on another branch without the coordinating
scripts being present there.

#### 1. Broad Parameter Discovery: `cpu_param_sweep.pl`

`scripts/cpu_param_sweep.pl` is the JSON-driven coordinator; it is distinct
from `build/cpu_param_sweep`, the C benchmark executable that it rebuilds and
invokes. It requires `--config` and uses
`scripts/benchmark_config_cpu.json` for build matrices, runtime matrices,
telemetry, commands, and output parsing.

The checked-in configuration uses paths relative to the `scripts/` directory,
so run it from there:

```bash
git switch main
cd scripts
perl ./cpu_param_sweep.pl --config ./benchmark_config_cpu.json
```

##### Configuration Model

The framework is schema-driven: the JSON file is the source of truth for the
build matrix, run matrix, system environment, telemetry, and output parser.
The Perl engine dynamically enumerates the build/run matrix, gathers telemetry,
interpolates commands, and runs each configured instance. It is generic within
the three supported blocks, `build_matrix`, `run_matrix`, and `system_env`; the
current configuration still supplies the benchmark-specific `build_cmd`,
`run_cmd`, and parser contract.

1. `benchmark_config_cpu.json` defines compiler and Make-variable combinations,
   workload sizes, threads, NUMA policies, command templates, output locations,
   telemetry extractors, and CSV columns.
2. `cpu_param_sweep.pl` reads that configuration at runtime, creates its
   Cartesian build/run space, adds command-line bindings for configuration keys,
   and executes each workload using the configured `taskset`, `numactl`, and
   scheduling settings.

The current configuration requires Perl 5.36, `numactl`, `taskset`, and the
selected compiler/toolchain. The active runner imports `Algorithm::Loops`,
`IPC::Run`, `Log::Log4perl`, and `JSON::PP`; ensure the required modules are
available before starting a sweep.

```bash
perl -MAlgorithm::Loops -MIPC::Run -MLog::Log4perl -MJSON::PP -e 1
```

It writes CSV and raw logs beneath `benchmark_CPU_params/` by default.

##### Command-Line Overrides

The runner creates options from every key in `build_matrix`, `run_matrix`, and
`system_env`. A value passed on the command line overrides the corresponding
JSON value; comma-separated matrix values become a smaller sweep.

```bash
cd scripts

# Override the configured output directory and thread-count matrix.
perl ./cpu_param_sweep.pl \
  --config ./benchmark_config_cpu.json \
  --out_dir ../custom_results \
  --threads 1,2,4,8,16,20
```

The JSON uses underscore-style option names, such as `out_dir`, because those
are the configuration keys consumed by `GetOptions`.

##### Telemetry and CSV Parsing

Telemetry is described in JSON rather than embedded as benchmark-specific Perl
logic. The current configuration reads CPU and operating-system files and runs
a compiler-version command. Its extractors include CPU model, SIMD tier, and
hardware population-count capability:

```json
"telemetry": {
  "hardware": {
    "file": "/proc/cpuinfo",
    "extractors": {
      "Processor": "model name\\s*:\\s*(.+)",
      "SIMD": "(?:flags|Features|isa).*?\\b(avx512f|avx2(?!.*\\bavx512f\\b)|avx(?!.*\\bavx2\\b|.*\\bavx512f\\b)|sse4_2(?!.*\\bavx)|sve2|sve(?!.*\\bsve2\\b)|asimd(?!.*\\bsve)|rv64[a-z]*v[a-z]*)\\b",
      "vpopcountHW": "(?:flags|Features|isa).*?\\b(avx512_vpopcntdq|avx512_bitalg|zvbb|asimd)\\b"
    }
  },
  "os": {
    "file": "/etc/os-release",
    "extractors": {
      "Operating_System": "PRETTY_NAME=\\\"([^\\\"]+)\\\""
    }
  },
  "toolchain": {
    "cmd": "{cc} --version | head -n 1",
    "extractors": {
      "Compiler_Version": "(.*)"
    }
  }
}
```

Likewise, `output_parser` captures the benchmark's standard output using named
regular-expression groups and writes selected values into CSV columns. Its map
normalizes the optional container label before output:

```json
"output_parser": {
  "regex": "(?i)Total\\s+time\\s+for\\s+(?<Benchmark_Type>Container\\s+-\\s+)?Multi-threaded\\s+-\\s+OpenMP:\\s+(?<Timing_ns>\\d+)\\s+ns.*?Number\\s+of\\s+threads:\\s+(?<Threads>\\d+)",
  "columns": ["Benchmark_Type", "Threads", "Timing_ns"],
  "map": {
    "Benchmark_Type": {
      "Container - ": "Containerized",
      "Container -": "Containerized",
      "__UNDEF__": "Non-Containerized"
    }
  }
}
```

The regex and command templates are part of the selected configuration's
contract. Update them together when changing benchmark output or command-line
behavior.

#### 2. Focused Kernel Profiling & Tuning: `sweep_cpu_tuning.pl`

After the broad parameter sweep identifies candidates, the focused tools on
`main` answer two different questions: `sweep_cpu_tuning.pl` collects repeated
timing and performance-counter evidence for a selected packed-container
configuration, while `run_numa_sweeps.sh` compares that workflow under four
dual-socket memory-placement cases. They currently remain on `main`:

- `scripts/sweep_cpu_tuning.pl`
- `scripts/run_numa_sweeps.sh`

Switch to `main` before using either workflow. `sweep_cpu_tuning.pl` must run
from the repository root because it verifies that `Makefile` is present there.

```bash
git switch main
make bench_omp GPU=NONE CC=clang

ELEVATE=always CORES=0-9 REPS=5 PERF_REPS=3 \
  perl ./scripts/sweep_cpu_tuning.pl
```

##### Focused Container-Kernel Sweep

`sweep_cpu_tuning.pl` is used to generate insights before CPU tuning of the containerized
intersection-count kernel for a given architecture. It really is a "live-cell imaging" of the library in action as it streams data across memory hierarchies. For each configuration it performs a clean rebuild,
runs `build/openmp_bit_container` with an explicit CPU affinity, and collects
`perf stat` profiles. It is intended to compare CPU tiles, K blocks,
outer-product microkernel shapes, unrolling, and the independent libpopcnt
scratch-buffer path without timing unrelated benchmark work.

Use it after `cpu_param_sweep.pl` narrows the candidate space or whenever
cache, execution, vectorization, TLB, NUMA, power, or scheduling evidence is
needed. Unlike `build/cpu_param_sweep`, this benchmark measures only the packed
`Bit_DB_T` intersection-count path; it does not compare ordinary-bitset and
container timings in the same process.

The default sweep evaluates the direct SIMD path (`LIBPOPCNT_MODES=0`). Include
both modes explicitly when comparing it with the libpopcnt path:

```bash
git switch main
LIBPOPCNT_MODES=0,1 ELEVATE=always \
  perl ./scripts/sweep_cpu_tuning.pl
```

To evaluate both algorithms, every default tuning parameter, and all 15
diagnostic profiles on one socket, run this from the repository root:

```bash
LIBPOPCNT_MODES=0,1 \
CORES=0-9 THREADS=10 \
REPS=5 PERF_REPS=3 \
RUN_LABEL=avx512 \
PERF_PROFILES=summary,cache-l1,cache-l2,cache-l3-dram,cache-stalls,buffers-pending,buffers-store,execution-uops,execution-ports,frontend,frequency,vectorization,tlb,uncore-numa,power-rapl \
ELEVATE=always \
./scripts/sweep_cpu_tuning.pl
```

This is a long, sequential measurement. Direct SIMD varies four CPU tiles,
four K blocks, four microkernel shapes, and three unroll factors for 192 build
configurations. The libpopcnt path varies the same tiles, blocks, and shapes
plus four scratch-buffer sizes for another 256. Together, the two modes produce
448 clean builds and $448 \times 15 = 6{,}720$ separate `perf stat` commands.

The repetition controls work at different levels. `PERF_REPS=3` becomes
`perf stat -r 3`, so those 6,720 commands launch 20,160 benchmark processes.
Each process receives `REPS=5` and performs one untimed warm-up followed by five
timed intersection calls. The script also runs the benchmark once outside the
profile loop to collect its primary timing output. Compiler speed, workload
size, PMU access, and host load determine the wall-clock duration, so use this
as a machine specific profiler that can help you understand why the library performs (or not) in the given machine.

Start with a small trial after changing machines, compilers, PMU permissions,
or event sets:

```bash
git switch main
MAX_CONFIGS=2 REPS=1 PERF_REPS=1 PERF_PROFILES=summary ELEVATE=always \
  perl ./scripts/sweep_cpu_tuning.pl
```

`MAX_CONFIGS=2` stops after two build configurations, `REPS=1` performs one
timed call per benchmark process, and `PERF_REPS=1` runs each profile once. For
this toolchain and permission check, `PERF_PROFILES=summary` keeps profiling to
the smallest general-purpose event set.

All sweep variables are environment variables. Comma-separated values define a
matrix; a single value fixes that dimension.

| Variable | Default | Description |
| --- | --- | --- |
| `LIBPOPCNT_MODES` | `0` | Algorithms to compare: `0` is direct SIMD; `1` is the libpopcnt scratch-buffer path. |
| `CPU_TILES` | `4,8,16,32` | Values compiled as `CPU_TILE`. |
| `K_BLOCKS` | `256,512,768,1024` | Values compiled as `BITVECTOR_TILE`. |
| `SHAPES` | `1x1,2x2,2x4,4x2` | Outer microkernel shapes, written as `ROWSxCOLS`. |
| `UNROLLS` | `1,2,4` | `OUTER_VEC_BLK` values, used for the direct-SIMD path. |
| `BUFFER_SIZES` | `128,512,1024,4096` | `BUFFER_SIZE` values, used for the libpopcnt path. |
| `CC` | `clang` | Compiler supplied to `make`. |
| `CORES` | `0-9` | CPU list supplied to `taskset -c`; choose physical cores where possible. |
| `BITS` | `65536` | Bitset length passed to `openmp_bit_container`. |
| `LEFT`, `RIGHT` | `1000`, `1000` | Left and right packed-container sizes. |
| `THREADS`, `REPS` | `10`, `5` | OpenMP thread count and timed repetitions. |
| `PERF_REPS` | `3` | Repetitions requested from `perf stat` for each profile. |
| `PERF_PROFILES` | profile set | Comma-separated profiles such as `summary`, `cache-l1`, `cache-l2`, `cache-l3-dram`, `cache-stalls`, `buffers-pending`, `buffers-store`, `execution-uops`, `execution-ports`, `frontend`, `frequency`, `vectorization`, `tlb`, `uncore-numa`, and `power-rapl`. Use `summary` while narrowing the matrix. |
| `PERF_EVENTS` | unset | Optional replacement event list for the `summary` profile. |
| `ELEVATE` | `auto` | `never`, `auto`, or `always`; elevation can be needed for performance counters or scheduling priority. |
| `PRIORITY` | `nice` | `normal`, `nice`, or real-time round-robin `rr`; elevated privileges are required where the operating system requires them. |
| `MAX_CONFIGS` | `0` | Stops after this many configurations; `0` means no limit. |
| `ARCH_TAG` | detected | Optional safe filename label for reports. |
| `RUN_LABEL` | unset | Optional label inserted into report and artifact names. |
| `NUMA_POLICY` | `default OS policy` | Descriptive policy text recorded in the Markdown report. |
| `NUMA_CMD` | unset | Optional `numactl` command prefixed to the benchmark process. |
| `RESULTS_DIR` | `tuning-results` | Directory for `summary-<run-tag>.csv` and `llm-summary-<run-tag>.md`. |
| `OUT_DIR` | `tuning-results/.work/<run-tag>` | Directory for per-configuration build, benchmark, and perf artifacts. |

###### Performance Profiles

Each name in `PERF_PROFILES` is a separate `perf stat` invocation with its own
event list and `*.perf.csv` artifact. The names describe diagnostic questions;
the script maps them to PMU events for generic Intel, hybrid Intel P-core, AMD
x86-64, generic AArch64, and Rockchip/Rock64-class systems.

| Profile | What it helps explain |
| --- | --- |
| `summary` | Timing rank, instructions per cycle, branch behavior, and general cache-miss rate. |
| `cache-l1` | Retired-load L1 hits and misses. |
| `cache-l2` | Retired-load L2 behavior after L1. |
| `cache-l3-dram` | Last-level-cache behavior and local DRAM demand-load misses. |
| `cache-stalls` | Cycles stalled around L1D, L2, and L3 miss activity. |
| `buffers-pending` | Fill-buffer saturation and pending-miss occupancy. |
| `buffers-store` | Store-buffer and store-queue pressure plus outstanding data-read depth. |
| `execution-uops` | Issued, executed, and retired micro-operations plus backend stalls. |
| `execution-ports` | Distribution of work across execution ports. |
| `frontend` | Undelivered micro-operations and low-delivery cycles. |
| `frequency` | APERF/MPERF behavior, including possible AVX-512 frequency changes. |
| `vectorization` | Packed SIMD work compared with scalar fallback indicators. |
| `tlb` | Data and instruction TLB loads and misses. |
| `uncore-numa` | Cross-socket, interconnect, and memory-controller traffic where supported. |
| `power-rapl` | Package and RAM energy or power counters where supported. |

The profile name remains stable across machines, but the underlying events do
not. Some maps use `cycles,instructions` when a detailed event is unavailable;
many Rockchip profiles and some Intel execution-port profiles intentionally
fall back this way. Check the generated event list before comparing unlike
architectures. `PERF_EVENTS` replaces the event list for `summary` only.

Profiles are kept separate rather than combined into one enormous event set.
An individual profile can still exceed the available hardware counters, so use
the running/scaling information from `perf` when interpreting multiplexed
counts. A missing event or permission affects that profile; the benchmark
timing remains available, and the profile CSV/log records what failed.

###### What `perf` Measures

The script runs:

```text
build/openmp_bit_container <bits> <left-bitsets> <right-bitsets> <threads> <repetitions>
```

The executable allocates and initializes packed containers, runs one untimed
`BitDB_inter_count_store_cpu` warm-up, performs the requested timed repetitions,
and prints each timing plus best, average, Gqword-pairs/s, and a result checksum.
`perf` wraps the complete process, so its counters include allocation,
initialization, warm-up, timed calls, checksum, and teardown. Use the C
nanosecond timings to rank the intersection kernel and the PMU profiles to
understand the wider process behavior.

Each run produces compact, architecture-labelled outputs such as
`tuning-results/summary-<run-tag>.csv` and
`tuning-results/llm-summary-<run-tag>.md`. The accompanying `.work/<run-tag>/`
directory holds per-configuration build logs, benchmark output, and perf CSV
files. A full matrix can be long-running, so reduce the matrix first and reserve
the complete profile set for selected candidates.

#### 3. Generic Dual-Socket NUMA Experiment: `run_numa_sweeps.sh`

`run_numa_sweeps.sh` implements a generic dual-socket experiment: establish a
local-memory baseline on each socket, then compare a dual-socket default
first-touch run with explicit memory interleaving. CPU affinity alone does not
choose where pages are allocated. Because the focused benchmark initializes
its shared input containers before OpenMP workers begin, ordinary Linux
first-touch placement can put many pages on one node and make work on the other
socket remote-memory heavy.

The checked-in script maps that method to a dual-socket Xeon E5-2697 v4 example
with 18 physical cores per socket. It locates the repository root itself and
may be started from another directory:

```bash
git switch main
bash ./scripts/run_numa_sweeps.sh
```

It requires `numactl`. The Xeon example runs these four comparable sweeps:

1. `socket0-local`: CPUs `0-17`, 18 threads, and allocation bound to NUMA node 0.
2. `socket1-local`: CPUs `18-35`, 18 threads, and allocation bound to NUMA node 1.
3. `dual-first-touch-spread`: CPUs `0-35`, 36 threads, spread OpenMP binding, and default Linux first-touch placement.
4. `dual-interleave`: CPUs `0-35`, 36 threads, spread OpenMP binding, and memory interleaved across nodes 0 and 1.

The single-socket runs provide local-memory baselines. Comparing the two
dual-socket runs helps distinguish an asymmetric first-touch placement effect
from the effect of explicit interleaving. Interleaving balances allocation; it
does not make every access local.

Before applying the method to another dual-socket machine, inspect its topology
and adapt the script's CPU lists, thread counts, NUMA-node IDs, and `ARCH_TAG`:

```bash
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE
```

| Adaptation point | Checked-in Xeon E5-2697 v4 example | Select for another host |
| --- | --- | --- |
| First socket CPU list | `0-17` | CPUs belonging to one socket and its chosen physical-core policy. |
| Second socket CPU list | `18-35` | CPUs belonging to the other socket. |
| Socket-local worker count | `18` | A count no greater than the selected socket's CPU capacity. |
| Dual-socket worker count | `36` | A count no greater than the combined selected capacity. |
| NUMA node IDs | `0`, `1` | The nodes backing the selected socket CPU lists. |
| Artifact architecture tag | `x86-64-intel-xeon-e5-2697-v4` | A stable description of the tested CPU/topology. |

The current wrapper does not discover topology or expose these values as
overrides, so manual adaptation is required. A future portable revision should
provide explicit CPU-list, NUMA-node, worker-count, and tag overrides; validate
nonempty/nonoverlapping online CPU lists and valid NUMA nodes before a run; and
offer a dry-run mode that prints all four resolved experiments. It should keep
the four experiment meanings stable while recording the resolved topology in
the output labels.

The runner forwards `OMP_PLACES`, `OMP_PROC_BIND`, `NUMA_CMD`, and the named
NUMA policy to the tuning script. `perf` access is governed by the host's
permissions and `kernel.perf_event_paranoid`; use `ELEVATE=always` only where
permitted by local administration policy.

### GPU Parameter Sweep (`gpuOpt`)

`scripts/gpu_param_sweep.pl` sweeps the experimental native benchmark matrix
through `Makefile_bench.mak`. It supports `--backend`, `--make-args`,
`--iterations`, `--out-dir`, `--summary`, `--log`, and `--dry-run`.

```bash
git switch gpuOpt

CUDA_VISIBLE_DEVICES=0 perl ./scripts/gpu_param_sweep.pl \
  --backend=NVIDIA \
  --make-args='CC=clang GPU_ARCH=sm_70'

ROCR_VISIBLE_DEVICES=0 perl ./scripts/gpu_param_sweep.pl \
  --backend=AMD \
  --make-args='CC=clang GPU_ARCH=gfx1010'
```

Choose the GPU with `--backend`; reserve `--make-args` for compiler,
architecture, and other Make variables.
By default, the script writes backend/architecture-labelled CSV and raw log
files in `benchmark_GPU_params/`. The native CUDA/HIP workflow, its result
files, and `plot_performance.R` belong to `gpuOpt` and are separate from the CPU
sweep suite. Plot the collected CSV files with:

```bash
Rscript ./scripts/plot_performance.R
```

### Script Inventory by Branch

The script trees are intentionally different. 

| Script or group | `main` | `gpuOpt` | `inteliGPU` | Purpose |
| --- | --- | --- | --- | --- |
| `generate_bug_report.sh` | Yes | Yes | Yes | Backend for `make bug_report`; collects build configuration, diagnostics, preprocessed source, and an optional backtrace. |
| `cpu_param_sweep.pl` + `benchmark_config_cpu.json` | Yes | No | No | JSON-driven broad CPU build/runtime sweep. |
| `cpu_profiling_analytics.R` | Yes | No | No | Intended analysis and plotting companion for broad CPU sweep CSV files; see the compatibility note below. |
| `sweep_cpu_tuning.pl` | Yes | No | No | Focused CPU kernel timing and `perf stat` profiling; writes its own CSV and Markdown reports. |
| `run_numa_sweeps.sh` | Yes | No | No | Runs four dual-socket scenarios through `sweep_cpu_tuning.pl`. |
| `gpu_param_sweep.pl` + `plot_performance.R` | No | Yes | No | Compatible GPU sweep and plotting pair for `benchmark_GPU_params/`. |
| Tracked `benchmark_GPU_params/` results | No | Yes | No | Historical GPU sweep CSV/log results kept with their producer and plotter. |
| `faiss_multigpu_benchmark.py` | No | Yes | No | Fixed-workload FAISS binary-index comparison with a measured CPU baseline and each detected CUDA GPU. |
| `faiss_multigpu_benchmark_nocpu.py` | No | Yes | No | Similar FAISS GPU comparison without a measured CPU baseline; reports devices relative to GPU 0. |
| `push_main_to_gpuOpt.sh`, `push_main_to_inteliGPU.sh` | Yes | No | No | Copy curated paths from `main` to the named destination branch. |
| `push_gpuOpt_to_main.sh`, `push_gpuOpt_to_inteliGPU.sh` | No | Yes | No | Mirror the same selective-copy workflow with `gpuOpt` as the source branch. |

The FAISS programs require Python, NumPy, and a FAISS build with GPU support.
They print fixed-workload timing summaries and do not feed either R script.

### Benchmark Producers and Analytics

The script names suggest several pairings, but the file formats decide which
ones work end to end:

- **GPU sweep and plot (`gpuOpt`):** `gpu_param_sweep.pl` writes architecture-labelled
  CSV files under `benchmark_GPU_params/`. `plot_performance.R` reads those
  files using the same `TILE_J`, `ILP`, workload, timing-type, and throughput
  columns. This is the working Perl-to-R pair.
- **Broad CPU sweep and analytics:** `cpu_param_sweep.pl` and
  `cpu_profiling_analytics.R` are intended companions: use the Perl script to generate the data and the R script to help you visualize them.
- **Focused CPU tuning:** `sweep_cpu_tuning.pl` is self-contained. It writes
  `summary-<run-tag>.csv`, `llm-summary-<run-tag>.md`, and per-configuration
  build/benchmark/perf files under `tuning-results/.work/`; no R script in this
  repository currently consumes that schema.
- **NUMA orchestration:** `run_numa_sweeps.sh` is not an analytics consumer. It
  invokes `sweep_cpu_tuning.pl` four times with socket-local, first-touch, and
  interleaved-memory settings, producing comparable tuning reports.

The R helpers use packages such as `data.table`, `ggplot2`, `this.path`, and
`bit64`; consult each script for its exact package list and output behavior.

### Branch Synchronization Helpers

Synchronization is available in both directions between `main` and `gpuOpt`,
and both source branches can update `inteliGPU`:

| Run from | Helper | Destination |
| --- | --- | --- |
| `main` | `scripts/push_main_to_gpuOpt.sh` | `gpuOpt` |
| `main` | `scripts/push_main_to_inteliGPU.sh` | `inteliGPU` |
| `gpuOpt` | `scripts/push_gpuOpt_to_main.sh` | `main` |
| `gpuOpt` | `scripts/push_gpuOpt_to_inteliGPU.sh` | `inteliGPU` |

Preview any synchronization without fetching, switching branches, staging,
committing, or pushing:

```bash
./scripts/push_main_to_gpuOpt.sh --dry-run
# Use the helper available on the current source branch for another direction.
```

All four helpers require a clean source worktree and both local branches. They
fetch and fast-forward the destination, copy the paths selected by that helper
with `git restore`, create a commit when content changed, push the destination,
and return to the starting branch. For one invocation, the selected source
paths are authoritative: corresponding destination edits are replaced rather
than merged, while files outside the selected set remain untouched.

The workflows mirror one another, and each helper copies the same explicitly
curated shared library/test/benchmark paths plus the complete `include/` tree.
Branch-exclusive CPU and GPU benchmark files stay with their owning branch.
Helpers targeting `main` or `inteliGPU` also remove stale gpuOpt-only benchmark
files; the helper targeting `gpuOpt` never removes them.

`README.md` is selected by every helper, so a sync also replaces the
destination branch's README with the source branch version.

The separate [benchmarking-bits](https://github.com/chrisarg/benchmarking-bits)
repository contains comparative C and Perl bitset/bitmap benchmarks. It is a
research companion rather than a dependency of this library. At the time of this writing (August 2026) this repository reflects the performance of an earlier version of `Bit` (the first release version).

## Constraints and Current Status

- **Capacity:** Bitsets have fixed, `int`-sized capacities.
- **Validation:** Most pointer, index, shape, and allocation checks use
  `assert`. Defining `NDEBUG` removes them; callers still provide valid indexes,
  equal-length operands, and correctly sized borrowed buffers.
- **Concurrency:** You synchronize shared mutation. GPU calls are synchronous
  in the current library path.
- **GPU residency:** Update and release flags control device mappings. GPU
  strategy selection belongs to the experimental benchmark layer, not the
  runtime library API.
- **Current research surfaces:** Intel OpenMP offload and native CUDA/HIP
  benchmarks remain experimental.
- **Generated files:** GPU binaries, LLVM intermediates, profiler reports, and
  benchmark results record builds and experiments; the public API is defined by
  the header and implementation.

## Design, Concurrency, and Performance Notes

### Concurrency and Execution

Individual bitsets are mutable buffers, so you coordinate concurrent access to
shared objects. CPU container calls use OpenMP internally; keep shared operands
and result buffers under one controlling call unless your application provides
its own synchronization.

GPU-facing container functions are synchronous. Device, update, and release
options control data residency across calls; they do not provide asynchronous
execution or cross-thread synchronization. The `gpuOpt` layout machinery may
retain prepared layouts, so keep ownership and lifetime boundaries explicit.

I have used the container API through its ordinary fork-join path: one thread
enters a call and OpenMP parallelizes the work inside it. Nested tasks, multiple
controlling threads sharing operands, and `fork` after OpenMP initialization
remain untested here. An application may use runtime tools such as
`omp_pause_resource_all` before `fork`; test that sequence with the OpenMP
implementation you deploy.

The implementation uses C preprocessor helpers and `_Pragma` to express a
family of OpenMP CPU and GPU regions without duplicating every variant by hand.
That is an implementation technique, not a public macro interface. The
benchmark sources are the practical reference for how those regions are mapped
and measured.

### Population Count and WWG

The codebase uses the name Wilkes-Wheeler-Gill (WWG) for a portable
sideways-addition population-count technique. Historical literature also calls
the technique Gillies-Miller sideways addition.[^wwg-history] The distinction is historical;
the relevant engineering point is that the arithmetic form offers a portable
fallback when a specific target or compiler path does not use a native popcount
instruction.

For GPU work, WWG is the default code path unless
`USE_BUILTIN_POPCOUNT=1` is selected at build time. Modern compiler/target
combinations may recognize either form efficiently, but generated instructions
and performance must be measured for the selected compiler, architecture, data
layout, and memory-transfer pattern. The GPU-only benchmark exists to make that
comparison explicit.

There is a useful compiler lesson hiding here: source spelling is not the same
thing as generated machine code. In a project investigation using Clang's
NVIDIA target, the hand-written WWG expression and `__builtin_popcountll`
produced byte-identical device PTX containing `popc.b64`. LLVM recognized the
classic SWAR pattern and canonicalized it to the hardware operation. That is a
specific observation, not a promise about every compiler, optimization level,
or AMD/NVIDIA target, but it explains why toggling `USE_BUILTIN_POPCOUNT` need
not change performance. Inspect generated code and benchmark the intended
binary before assigning speed to the source-level choice.

### Why Containers and OpenMP

The non-containerized bitset operations are straightforward to parallelize at
the application level. Packed containers additionally make it practical to
schedule many all-pairs count operations while controlling the storage layout.
CPU tiling, OpenMP scheduling, and GPU layout experiments are all attempts to
make locality and work distribution visible to the implementation rather than
leaving every choice to a generic loop nest.

This distinction is intentional. An application with an array of independent
`Bit_T` objects can write an OpenMP loop directly:

```c
#include "bit.h"
#include <assert.h>
#include <stdlib.h>

int main(void) {
  const int query_count = 2;
  const int reference_count = 3;
  Bit_T queries[2];
  Bit_T references[3];
  int *counts = calloc((size_t)query_count * reference_count, sizeof(*counts));
  assert(counts != NULL);

  for (int i = 0; i < query_count; ++i) {
    queries[i] = Bit_new(128);
    Bit_bset(queries[i], 10 + i);
  }
  for (int j = 0; j < reference_count; ++j) {
    references[j] = Bit_new(128);
    Bit_bset(references[j], 10 + j);
  }

#pragma omp parallel for collapse(2)
  for (int i = 0; i < query_count; ++i) {
    for (int j = 0; j < reference_count; ++j) {
      counts[i * reference_count + j] =
          Bit_inter_count(queries[i], references[j]);
    }
  }

  for (int i = 0; i < query_count; ++i) Bit_free(&queries[i]);
  for (int j = 0; j < reference_count; ++j) Bit_free(&references[j]);
  free(counts);
  return 0;
}
```

`Bit_DB_T` exists for the cases where I want the library to own that bulk
organization. Its contiguous storage lets the implementation tile the two
outer container dimensions and block the inner bit-vector reduction. The
`CPU_TILE`, `BITVECTOR_TILE`, outer-row/column shape, unroll, and scratch-buffer
settings are experiments in cache use, register pressure, and memory traffic;
they are tuning controls, not universal constants.

The internal `_Pragma` helpers serve the same purpose on the code-organization
side. They let one family of loops express CPU worksharing, SIMD reduction, GPU
teams, and mapping choices without maintaining several nearly identical
kernels. This is one of the places where the C preprocessor is earning its keep,
but those helpers remain private implementation machinery rather than an API
applications should depend on.

On a single socket, the relevant limits are often cache capacity and memory
bandwidth. On a multi-socket host, page placement and thread binding matter as
well; the `main`-only NUMA sweep documents one way to make those variables
measurable. On a GPU, transfer volume, residency, layout conversion, and launch
overhead can dominate a small or poorly shaped workload even when the inner
kernel is fast.

[^wwg-history]: Maurice V. Wilkes, David J. Wheeler, and Stanley Gill describe
  the Gillies-Miller method for sideways addition in *The Preparation of
  Programs for an Electronic Digital Computer*, 2nd ed., pp. 191-193 (1957).
  Wojciech Mula, Nathan Kurz, and Daniel Lemire later used the name
  “Wilkes-Wheeler-Gill” in “Faster Population Counts Using AVX2
  Instructions,” *The Computer Journal* 61(1), 2018.

## Dependencies, Inspiration, and Applications

This project incorporates or integrates the following open-source libraries:

- [libpopcnt](https://github.com/kimwalisch/libpopcnt), a BSD 2-Clause
  population-count library with architecture-specific implementations.
- [SIMDe](https://github.com/simd-everywhere/simde), a header-only SIMD
  portability layer used by the CPU implementation.

Several libraries and projects also informed the structure of this codebase and
the author's exploration of the C preprocessor and SIMD implementation work:

- [sse-popcount](https://github.com/WojciechMula/sse-popcount), including the
  Harley-Seal population-count work associated with Lemire, Kurz, and Mula.
- [cii](https://github.com/drh/cii), David Hanson's C Interfaces and
  Implementations library and the original `Bit_T` design.

### Applications

Bit is particularly useful for dense set and membership workloads such as:

- Bioinformatics and genomic data processing, including k-mer-like encodings.
- Network packet filtering and Bloom-filter-style membership tests.
- Dense data representation over large fixed domains.
- High-performance set operations and all-pairs intersection-count searches.

For genuinely sparse domains, a compressed representation such as a roaring
bitmap can be a better fit than this uncompressed library.

## Roadmap

- Continue validating CPU, NVIDIA, AMD, and Intel build paths.
- Extend SIMD-oriented CPU work across more set-operation paths while retaining
  portable fallbacks.
- Port or evaluate selected experimental `gpuOpt` count algorithms on the
  branches where they belong.
- Add set-operation metrics such as Jaccard similarity.
- Improve OS-agnostic build, profiling, and reproducibility workflows.
- Continue evaluating native CUDA/HIP backends alongside OpenMP offload.
- Investigate Unified Shared Memory where the target runtime supports it.

TPU and NPU support are not implemented and are not supported build targets.

## License

BSD 2-Clause License. See `LICENSE` for details.

## Author

Christos Argyropoulos (April 2025 -  May 2026)

## AI Disclosure and Scientific Publication Transparency Statement

This session is intended to document the involvement of AI in this project and a
roadmap to preserve, collect and characterize the involvement over time. In retrospect,
some of the steps (in particular recovery of history from other machines) should have
done much earlier than September 2026. The following few sections represent to the best of
my knowledge the use of AI in this project, which started of as a retype and extension of
the Bit T by David Hanson.

### AI-Assisted Work

GitHub Copilot and Google Gemini assisted with generating and refactoring
Makefile content, exploring test ideas for the OpenMP implementations, drafting
and refactoring templated C work used for the CUDA and HIP implementations, and
maintaining this README as the source evolved.

During development of the benchmarking framework and associated automation,
generative AI assisted with script refactoring, Perl automation boilerplate,
JSON configuration schemas, R authoring, OpenMP macro work, and documentation.
The following model-specific roles are author-confirmed and are described as
regular contributions, not as a complete per-file provenance record:

- Google Gemini 3.1: Perl automation, R authoring, and OpenMP macro work.
- Claude Sonnet 5 and Claude Sonnet 4.6: Makefile work.

GitHub Copilot is retained here as a development platform attribution. This
document does not infer its underlying model routing, per-turn model selection,
or relative model frequency from repository contents.

### Attribution Evidence and Limits

The repository does not use watermark analysis to identify authorship or assign
source code to a particular AI model. There is no universal source-code
watermark detector, and any provider-specific verification must use that
provider's supported process. Editing, formatting, copying, transformation, and
mixed human/AI work can make retrospective attribution incomplete or invalid.

Writing style, comments, formatting, compiled artifacts, commit wording, and
Git history are not sufficient evidence of a particular model's involvement.
Model-level claims in this disclosure are maintained from author-confirmed
records rather than inferred from source code.

### Recovering History From Other Machines

For work completed on other user-owned computers, collect first-party records
instead of attempting retrospective source attribution:

1. On each VS Code installation signed into the same account, enable
  `chat.sessionSync.enabled` and `github.copilot.chat.localIndex.enabled`.
2. Run `/chronicle reindex` on each machine to index retained local sessions and
  synchronize them where the account configuration permits it.
3. Preserve unpushed branches and reflogs from each checkout as work chronology,
  for example with `git branch -avv` and `git reflog show --all`.
4. Retrieve any account-level Copilot history or usage export available to the
  account owner, then reconcile it with author-maintained records.
5. Redact credentials, tokens, personal data, proprietary prompts, and other
  sensitive material before centralizing transcripts or exports.

These steps can recover only history still retained by the device or provider.
They do not recreate deleted sessions and may not expose the model routed for
each request.

### Author Responsibility

All core problem framing, architectural decisions (including decoupling
execution engines from target schemas), code validation, security reviews, and
scientific evaluations were performed directly by the author. The author
maintains responsibility for the accuracy, licensing, and integrity of all
submitted code and materials.
