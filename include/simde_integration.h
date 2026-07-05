/*
    Integrates the SIMDE library into the Bit project. This header file provides
   the necessary includes and configurations to enable SIMD operations across
   different architectures. It ensures that the appropriate SIMD headers are
   included and that the Bit project can leverage the performance benefits of
   SIMD instructions on supported platforms.

    * Author : Christos Argyropoulos
    * Created : July 4th 2026
    * Copyright : (c) 2025 - 2026
    * License : BSD-2
*/
#pragma once

#include <simde/simde-arch.h>
#include <simde/simde-common.h>
#include <simde/simde-features.h>

// Universal inclusion required for 128-bit and 256-bit popcount emulations
#include <simde/x86/avx512/popcnt.h>

/*
 * Detect architecture and map to the appropriate hardware register width.
 * AVX1 is merged into the SSE4.2 path because AVX1 lacks 256-bit integer
 * support.
 */
#if defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ARCH_X86_AVX512F)
#define BIT_SIMD_PATH_AVX512 1

#elif defined(SIMDE_X86_AVX2_NATIVE) || defined(SIMDE_ARCH_X86_AVX2)
#define BIT_SIMD_PATH_AVX2 1

#elif defined(SIMDE_X86_AVX_NATIVE) || defined(SIMDE_ARCH_X86_AVX) ||          \
    defined(SIMDE_X86_SSE4_2_NATIVE) || defined(SIMDE_ARCH_X86_SSE4_2) ||      \
    defined(SIMDE_ARCH_ARM_NEON) || defined(SIMDE_ARCH_AARCH64) ||             \
    defined(SIMDE_ARCH_WASM_SIMD128) || defined(SIMDE_ARCH_RISCV_V)
#define BIT_SIMD_PATH_128 1

#else
#define BIT_SIMD_PATH_SCALAR 1
#endif

// ------------------------------------------------------------------------
// Path Configurations
// ------------------------------------------------------------------------

#if defined(BIT_SIMD_PATH_AVX512)
#include <simde/x86/avx512.h>
#define VECTOR_TYPE simde__m512i
#define VECTOR_BYTES 64
#define VECTOR_QWORDS 8

#define VECTOR_UNALIGNED_LOAD simde_mm512_loadu_si512
#define VECTOR_ALIGNED_LOAD simde_mm512_load_si512
#define VECTOR_UNALIGNED_STORE simde_mm512_storeu_si512
#define VECTOR_ALIGNED_STORE simde_mm512_store_si512

#define BIT_AND(op1, op2) simde_mm512_and_epi64((op1), (op2))
#define BIT_OR(op1, op2) simde_mm512_or_epi64((op1), (op2))
#define BIT_XOR(op1, op2) simde_mm512_xor_epi64((op1), (op2))
#define BIT_AND_NOT(op1, op2) simde_mm512_andnot_epi64((op2), (op1))
#define BIT_NOT(op) simde_mm512_xor_epi64((op), simde_mm512_set1_epi64(-1))

#define SIMDe_POPCOUNT simde_mm512_popcnt_epi64
#define SIMDe_ZERO_VECTOR simde_mm512_setzero_si512()
#define SIMDe_VECTOR_ADD simde_mm512_add_epi64
#define SIMDe_STORE_VECTOR(ptr, vec)                                           \
  simde_mm512_storeu_si512((void *)(ptr), (vec))

#elif defined(BIT_SIMD_PATH_AVX2)
#include <simde/x86/avx2.h>
#define VECTOR_TYPE simde__m256i
#define VECTOR_BYTES 32
#define VECTOR_QWORDS 4

#define VECTOR_UNALIGNED_LOAD simde_mm256_loadu_si256
#define VECTOR_ALIGNED_LOAD simde_mm256_load_si256
#define VECTOR_UNALIGNED_STORE simde_mm256_storeu_si256
#define VECTOR_ALIGNED_STORE simde_mm256_store_si256

#define BIT_AND(op1, op2) simde_mm256_and_si256((op1), (op2))
#define BIT_OR(op1, op2) simde_mm256_or_si256((op1), (op2))
#define BIT_XOR(op1, op2) simde_mm256_xor_si256((op1), (op2))
#define BIT_AND_NOT(op1, op2) simde_mm256_andnot_si256((op2), (op1))
#define BIT_NOT(op) simde_mm256_xor_si256((op), simde_mm256_set1_epi64x(-1))

#define SIMDe_POPCOUNT simde_mm256_popcnt_epi64
#define SIMDe_ZERO_VECTOR simde_mm256_setzero_si256()
#define SIMDe_VECTOR_ADD simde_mm256_add_epi64
#define SIMDe_STORE_VECTOR(ptr, vec)                                           \
  simde_mm256_storeu_si256((simde__m256i *)(ptr), (vec))

#elif defined(BIT_SIMD_PATH_128)
// AVX1, SSE4.2, NEON, and RISC-V all map through this 128-bit execution path
#include <simde/x86/sse4.2.h>
#define VECTOR_TYPE simde__m128i
#define VECTOR_BYTES 16
#define VECTOR_QWORDS 2

#define VECTOR_UNALIGNED_LOAD simde_mm_loadu_si128
#define VECTOR_ALIGNED_LOAD simde_mm_load_si128
#define VECTOR_UNALIGNED_STORE simde_mm_storeu_si128
#define VECTOR_ALIGNED_STORE simde_mm_store_si128

#define BIT_AND(op1, op2) simde_mm_and_si128((op1), (op2))
#define BIT_OR(op1, op2) simde_mm_or_si128((op1), (op2))
#define BIT_XOR(op1, op2) simde_mm_xor_si128((op1), (op2))
#define BIT_AND_NOT(op1, op2) simde_mm_andnot_si128((op2), (op1))
#define BIT_NOT(op) simde_mm_xor_si128((op), simde_mm_set1_epi64x(-1))

#define SIMDe_POPCOUNT simde_mm_popcnt_epi64
#define SIMDe_ZERO_VECTOR simde_mm_setzero_si128()  
#define SIMDe_VECTOR_ADD simde_mm_add_epi64
#define SIMDe_STORE_VECTOR(ptr, vec)                                           \
  simde_mm_storeu_si128((simde__m128i *)(ptr), (vec))

#else
#define VECTOR_BYTES 0
#define VECTOR_QWORDS 0
#endif

// ------------------------------------------------------------------------
// Diagnostics
// ------------------------------------------------------------------------

#if defined(BIT_SIMD_DIAGNOSTICS) && (BIT_SIMD_DIAGNOSTICS)
#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER) ||    \
    defined(__INTEL_LLVM_COMPILER)
#define BIT_SIMD_PRAGMA_MESSAGE(msg) _Pragma(#msg)
#if defined(BIT_SIMD_PATH_AVX512)
BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: AVX-512 (SIMDe)"))
#elif defined(BIT_SIMD_PATH_AVX2)
BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: AVX2 (SIMDe)"))
#elif defined(BIT_SIMD_PATH_128)
BIT_SIMD_PRAGMA_MESSAGE(
    message("[bit] SIMD path selected: 128-bit AVX/SSE/NEON (SIMDe)"))
#else
BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: Scalar fallback"))
#endif
#undef BIT_SIMD_PRAGMA_MESSAGE
#endif
#endif

// ------------------------------------------------------------------------
// Loop Macro Configurations
// ------------------------------------------------------------------------

// Set loop unroll factor (U=4) to match the setop macro memory logic
#define VECTOR_UNROLL_FACTOR 4

// The block size represents the total number of uint64_t elements processed per
// iteration
#define VECTOR_BLOCK_SIZE (VECTOR_QWORDS * VECTOR_UNROLL_FACTOR)

// Correctly calculates the array index offset for a specific vector load/store
#define VECTOR_OFFSET(k) ((k) * VECTOR_QWORDS)