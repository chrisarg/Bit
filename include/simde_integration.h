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

/*
 * Prefer SIMDe "*_NATIVE" feature macros when available. They account for
 * SIMDE_NO_NATIVE and per-feature *_NO_NATIVE toggles.
 */
#if defined(SIMDE_X86_AVX512F_NATIVE) || defined(SIMDE_ARCH_X86_AVX512F)
        #define BIT_SIMD_PATH_AVX512 1
#elif defined(SIMDE_X86_AVX2_NATIVE) || defined(SIMDE_ARCH_X86_AVX2)
        #define BIT_SIMD_PATH_AVX2 1
#elif defined(SIMDE_X86_AVX_NATIVE) || defined(SIMDE_ARCH_X86_AVX)
        #define BIT_SIMD_PATH_AVX 1
#else
        #define BIT_SIMD_PATH_SCALAR 1
#endif

// Conditionally load only the necessary SIMDe intrinsic headers and define types
#if defined(BIT_SIMD_PATH_AVX512)
    #include <simde/x86/avx512.h>
    #define VECTOR_TYPE simde__m512i
#elif defined(BIT_SIMD_PATH_AVX2)
    #include <simde/x86/avx2.h>
    #define VECTOR_TYPE simde__m256i
#else // BIT_SIMD_PATH_AVX is the default fallback for AVX and scalar
    #include <simde/x86/avx.h>
    #define VECTOR_TYPE simde__m256i
#endif

#if defined(BIT_SIMD_DIAGNOSTICS) && (BIT_SIMD_DIAGNOSTICS)
#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
        #define BIT_SIMD_PRAGMA_MESSAGE(msg) _Pragma(#msg)
        #if defined(BIT_SIMD_PATH_AVX512)
            BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: AVX-512 (SIMDe)"))
        #elif defined(BIT_SIMD_PATH_AVX2)
            BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: AVX2 (SIMDe)"))
        #elif defined(BIT_SIMD_PATH_AVX)
            BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: AVX (SIMDe)"))
        #else
            BIT_SIMD_PRAGMA_MESSAGE(message("[bit] SIMD path selected: Scalar fallback"))
        #endif
        #undef BIT_SIMD_PRAGMA_MESSAGE
    #endif
#endif


