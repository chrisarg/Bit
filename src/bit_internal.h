/*
    Private implementation macros and shared configuration parameters for the
   Bit library. NOT part of the public API.

    * Author : Christos Argyropoulos
    * Created : April 1st 2025 (refactored June 2026)
    * Copyright : (c) 2025 - 2026
    * License : BSD-2
*/
#pragma once
#include "simde_integration.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

/* --- Shared low-level macro and bit math helpers --- */
#define STRINGIFY(x) #x

#define BPQW (sizeof(uint64_t) * 8)     // bits per qword
#define BPB (sizeof(unsigned char) * 8) // bits per byte
#define nqwords(len)                                                           \
  ((((len) + BPQW - 1) & (~(BPQW - 1))) / BPQW)          // ceil(len/BPQW)
#define nbytes(len) ((((len) + 8 - 1) & (~(8 - 1))) / 8) // ceil(len/8)

#define NO_SIMD /*NO SIMD*/

#define T Bit_T
#define T_DB Bit_DB_T

/* --- Universal bitwise alignment check (Alignment must be a power of 2) --- */
#define IS_ALIGNED_64(ptr) (((uintptr_t)(const void *)(ptr) & 63) == 0)
#define IS_ALIGNED_8(ptr) (((uintptr_t)(const void *)(ptr) & 7) == 0)

/* --- Architecture Detection via Pointer Size --- */
#if UINTPTR_MAX == 0xffffffff
#define ARCH_32BIT 1
#define ALIGN_CHECK(ptr) IS_ALIGNED_8(ptr)
#define ALIGNMENT 32
#elif UINTPTR_MAX == 0xffffffffffffffff
#define ARCH_32BIT 0
#define ALIGN_CHECK(ptr) IS_ALIGNED_64(ptr)
#define ALIGNMENT 64
#else
#error "Unsupported pointer size. Architecture must be 32-bit or 64-bit."
#endif

/* --- Tiling and cache parameters --- */
#ifndef CPU_TILE
#define CPU_TILE_BIT 32
#define CPU_TILE_BITS 32
#else
#define CPU_TILE_BIT CPU_TILE
#define CPU_TILE_BITS CPU_TILE
#endif

/* Outer Product Layout Settings. Change these to alter the microkernel */
#ifndef OUTER_ROW_NUM
#define OUTER_ROW_NUM 2
#endif

#ifndef OUTER_COL_NUM
#define OUTER_COL_NUM 2
#endif

/* Unroll factor for the SIMDe K-loop.
   U=2 is optimal for OUTER_ROW_NUM=2, OUTER_COL_NUM=2 (8 accumulators).
   If using OUTER_ROW_NUM=4, OUTER_COL_NUM=4 (16 accumulators), consider
   dropping to 1 to avoid spill */
#ifndef OUTER_VEC_BLK
#define OUTER_VEC_BLK 2
#endif

#ifndef BITVECTOR_TILE
#define K_BLOCK 1024
#else
#define K_BLOCK BITVECTOR_TILE
#endif

/* --- Default popcount buffer sizing --- */
#ifndef SETOP_BUFFER_SIZE
#ifndef BUFFER_SIZE
#define SETOP_BUFFER_SIZE 512
#else
#define SETOP_BUFFER_SIZE BUFFER_SIZE
#endif
#endif

/* Default in case the makefile was not used */
#ifndef USE_LIBPOPCNT
#define USE_LIBPOPCNT 1
#endif

/* --- Concrete representations of opaque types defined in bit.h --- */
struct T {
  unsigned int length;         // capacity of the bitset in bits
  unsigned int size_in_bytes;  // number of bytes of the 8 bit container
  unsigned int size_in_qwords; // number of qwords of the 64 bit container
  unsigned char *bytes;        // pointer to the first byte
  uint64_t *qwords;            // pointer to the first qword
  bool is_Bit_T_allocated;     // true if allocated by the library
};

struct T_DB {
  unsigned int nelem;          // number of bitsets in the packed container
  unsigned int length;         // capacity of the bitset in bits
  unsigned int size_in_bytes;  // number of bytes of the 8 bit set container
  unsigned int size_in_qwords; // number of qwords of the 64 bit set container
  unsigned char *bytes;        // pointer to the first byte
  uint64_t *qwords;            // pointer to the first qword
  bool is_Bit_T_allocated;     // true if allocated by the library
};

/* --- Fast inline WWG and Tree-Adder popcount algorithms --- */
#define C1_WWG UINT64_C(0X5555555555555555)
#define C2_WWG UINT64_C(0x3333333333333333)
#define C3_WWG UINT64_C(0x0F0F0F0F0F0F0F0F)
#define C4_WWG UINT64_C(0x0101010101010101)

static inline uint64_t count_WWG(uint64_t x) {
  x -= (x >> 1) & C1_WWG;
  x = ((x >> 2) & C2_WWG) + (x & C2_WWG);
  x = (x + (x >> 4)) & C3_WWG;
  x *= C4_WWG;

  return (x >> 56);
}

#define C1_TRADD UINT64_C(0xAAAAAAAAAAAAAAAA)
#define C2_TRADD UINT64_C(0xCCCCCCCCCCCCCCCC)
#define C3_TRADD UINT64_C(0xF0F0F0F0F0F0F0F0)
#define C4_TRADD UINT64_C(0xFF00FF00FF00FF00)
#define C5_TRADD UINT64_C(0x00FF00FF00FF00FF)
#define C6_TRADD UINT64_C(0xFF00FF00FF00FF00)
#define C7_TRADD UINT64_C(0x0000FFFF0000FFFF)
#define C8_TRADD UINT64_C(0xFFFF0000FFFF0000)
#define C9_TRADD UINT64_C(0x00000000FFFFFFFF)
#define C10_TRADD UINT64_C(0xFFFFFFFF00000000)

static inline uint64_t tree_adder(uint64_t v) {
  v = (v & C1_WWG) + ((v & C1_TRADD) >> 1);
  v = (v & C2_WWG) + ((v & C2_TRADD) >> 2);
  v = (v & C3_WWG) + ((v & C3_TRADD) >> 4);
  v = (v & C5_TRADD) + ((v & C6_TRADD) >> 8);
  v = (v & C7_TRADD) + ((v & C8_TRADD) >> 16);
  v = (v & C9_TRADD) + ((v & C10_TRADD) >> 32);
  return v;
}

/* CPU popcount dispatcher */
#define POPCOUNT(x) count_WWG((x))

/* ===========================================================================
   SECTION 1: OPENMP CPU PARALLELIZATION HELPERS
   ===========================================================================
 */

/* Collapse `levels` loops with the given schedule */
#define OMP_CPU_LOOP(levels, sched)                                            \
  _Pragma(STRINGIFY(omp parallel for collapse(levels) schedule(sched)))

#define OMP_CPU_LOOP_STATIC(levels, chunk)                                     \
  _Pragma(STRINGIFY(omp parallel for collapse(levels) schedule(static, chunk)))

/* SIMD vectorization directive (unaligned) */
#define OMP_CPU_SIMD _Pragma(STRINGIFY(omp simd))

/* SIMD with explicit alignment hint on listed pointers */
#define OMP_CPU_SIMD_ALIGN_BUFFER(alignment, ...)                              \
  _Pragma(STRINGIFY(omp simd aligned(__VA_ARGS__ : alignment)))

#ifndef NOGPU
/* Update a device array from the host (or vice versa) */
#define UPDATE_GPU_ARRAY(dir, array, index1, index2, dev_id)                   \
  _Pragma(                                                                     \
      STRINGIFY(omp target update dir(array [index1:index2]) device(dev_id)))

/* Map or unmap a device array via enter/exit data */
#define TARGET_GPU_ARRAY(point, dir, array, index1, index2, dev_id)            \
  _Pragma(STRINGIFY(omp target point data map(dir : array [index1:index2])     \
                        device(dev_id)))
#endif

/* --- End Section 1: OPENMP CPU PARALLELIZATION HELPERS --- */

/* ==========================================================================
   SECTION 2: PRIVATE IMPLEMENTATION MACROS
   File-local operational macros and helper wrappers.
   ========================================================================== */

/* Compute the last aligned index before the scalar remainder for a specific
 * chunk */
#define CHUNK_LIMIT(limit_var, k_b, k_max, BLOCK_SIZE)                         \
  size_t limit_var = k_b + ((k_max - k_b) / BLOCK_SIZE) * BLOCK_SIZE;

/* Compute the last aligned index before the scalar remainder */
#define LIMIT_STATEMENT(limit, bit_size_in_qwords, SETOP_BUFFER_SIZE)          \
  size_t limit = bit_size_in_qwords - bit_size_in_qwords % SETOP_BUFFER_SIZE;

/* --- End Section 2: PRIVATE IMPLEMENTATION MACROS --- */

/* ===========================================================================
   SECTION 3: SINGLE-BITSET SET OPERATION MACROS (Bit_T)
   ===========================================================================
 */

// Scalar bitwise operations
#define BIT_SCALAR_AND(op1, op2) ((op1) & (op2))
#define BIT_SCALAR_OR(op1, op2) ((op1) | (op2))
#define BIT_SCALAR_XOR(op1, op2) ((op1) ^ (op2))
#define BIT_SCALAR_AND_NOT(op1, op2) ((op1) & ~(op2))

/* Set operation that creates a new Bit_T result (modified from Hanson's book)
 */
#define setop_validate(sequal, snull, tnull)                                   \
  if (s == t) {                                                                \
    assert(s);                                                                 \
    return sequal;                                                             \
  } else if (s == NULL) {                                                      \
    assert(t);                                                                 \
    return snull;                                                              \
  } else if (t == NULL) {                                                      \
    return tnull;                                                              \
  } else {                                                                     \
    assert(s->length == t->length);                                            \
  }

#if !USE_LIBPOPCNT
#if BIT_SIMD_PATH_SCALAR
#define setop_count(op, s, t)                                                  \
  do {                                                                         \
    uint64_t count = 0;                                                        \
    unsigned int bit_size_in_qwords = s->size_in_qwords;                       \
    _Pragma(STRINGIFY(omp simd)) /* SIMD directive for the set operation */    \
        for (unsigned int i = 0; i < bit_size_in_qwords; i++) {                \
      count += POPCOUNT(BIT_SCALAR##op(s->qwords[i], t->qwords[i]));           \
    }                                                                          \
    return (int)count;                                                         \
  } while (0)
#else
#define setop_count(op, s, t)                                                  \
  do {                                                                         \
    uint64_t count = 0;                                                        \
    unsigned int bit_size_in_qwords = s->size_in_qwords;                       \
    size_t limit =                                                             \
        (bit_size_in_qwords / VECTOR_BLOCK_SIZE) * VECTOR_BLOCK_SIZE;          \
    size_t i = 0;                                                              \
                                                                               \
    VECTOR_TYPE sum0 = SIMDe_ZERO_VECTOR;                                      \
    VECTOR_TYPE sum1 = SIMDe_ZERO_VECTOR;                                      \
    VECTOR_TYPE sum2 = SIMDe_ZERO_VECTOR;                                      \
    VECTOR_TYPE sum3 = SIMDe_ZERO_VECTOR;                                      \
                                                                               \
    for (; i < limit; i += VECTOR_BLOCK_SIZE) {                                \
      /* Inline loads, op, and popcount to minimize live register state */     \
      sum0 = SIMDe_VECTOR_ADD(                                                 \
          sum0, SIMDe_POPCOUNT(BIT##op(                                        \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(0)]),      \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(0)]))));   \
                                                                               \
      sum1 = SIMDe_VECTOR_ADD(                                                 \
          sum1, SIMDe_POPCOUNT(BIT##op(                                        \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(1)]),      \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(1)]))));   \
                                                                               \
      sum2 = SIMDe_VECTOR_ADD(                                                 \
          sum2, SIMDe_POPCOUNT(BIT##op(                                        \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(2)]),      \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(2)]))));   \
                                                                               \
      sum3 = SIMDe_VECTOR_ADD(                                                 \
          sum3, SIMDe_POPCOUNT(BIT##op(                                        \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(3)]),      \
                    VECTOR_UNALIGNED_LOAD(                                     \
                        (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(3)]))));   \
    }                                                                          \
    /* Horizontal sum of the vector elements */                                \
    sum0 = SIMDe_VECTOR_ADD(sum0, sum1);                                       \
    sum2 = SIMDe_VECTOR_ADD(sum2, sum3);                                       \
    sum0 = SIMDe_VECTOR_ADD(sum0, sum2);                                       \
    uint64_t sum_array[VECTOR_QWORDS];                                         \
    SIMDe_STORE_VECTOR(sum_array, sum0);                                       \
    for (size_t j = 0; j < VECTOR_QWORDS; j++) {                               \
      count += sum_array[j];                                                   \
    }                                                                          \
    for (; i < bit_size_in_qwords; i++) {                                      \
      count += POPCOUNT(BIT_SCALAR##op(s->qwords[i], t->qwords[i]));           \
    }                                                                          \
    return (int)count;                                                         \
  } while (0)
#endif
#else
#define setop_count(op, s, t)                                                  \
  do {                                                                         \
    uint64_t count = 0;                                                        \
    unsigned int bit_size_in_qwords = s->size_in_qwords;                       \
    uint64_t setop_buffer[SETOP_BUFFER_SIZE]; /*buffer for popcount*/          \
    int limit = s->size_in_qwords - s->size_in_qwords % SETOP_BUFFER_SIZE;     \
    int i = 0;                                                                 \
    for (; i < limit; i += SETOP_BUFFER_SIZE) {                                \
      for (int j = 0; j < SETOP_BUFFER_SIZE; j++) {                            \
        setop_buffer[j] = BIT_SCALAR##op(s->qwords[i + j], t->qwords[i + j]);  \
      }                                                                        \
      count +=                                                                 \
          popcnt((void *)setop_buffer, SETOP_BUFFER_SIZE * sizeof(uint64_t));  \
    }                                                                          \
    for (; i < bit_size_in_qwords; i++) {                                      \
      count += POPCOUNT(BIT_SCALAR##op(s->qwords[i], t->qwords[i]));           \
    }                                                                          \
    return (int)count;                                                         \
  } while (0)
#endif

// unified macro for intersection, union, minus and difference operations
// note we can support scalar paths!
#if BIT_SIMD_PATH_SCALAR
#define setop(set, op, s, t)                                                   \
  do {                                                                         \
    _Pragma(STRINGIFY(omp simd)) /* SIMD directive for the set operation */    \
        for (unsigned int i = 0; i < bit_size_in_qwords; i++)                  \
            set->qwords[i] = BIT_SCALAR##op(s->qwords[i], t->qwords[i]);       \
  } while (0)
#else
#define setop(set, op, s, t)                                                   \
  do {                                                                         \
    size_t limit =                                                             \
        (s->size_in_qwords / VECTOR_BLOCK_SIZE) * VECTOR_BLOCK_SIZE;           \
    unsigned int bit_size_in_qwords = s->size_in_qwords;                       \
    size_t i = 0;                                                              \
    for (; i < limit; i += VECTOR_BLOCK_SIZE) {                                \
      /* Load First operand */                                                 \
      VECTOR_TYPE a0 = VECTOR_UNALIGNED_LOAD(                                  \
          (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(0)]);                    \
      VECTOR_TYPE b0 = VECTOR_UNALIGNED_LOAD(                                  \
          (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(0)]);                    \
      VECTOR_TYPE r0 = BIT##op(a0, b0);                                        \
      VECTOR_UNALIGNED_STORE(                                                  \
          (VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(0)], r0);              \
                                                                               \
      /* Load Second operand */                                                \
      a0 = VECTOR_UNALIGNED_LOAD(                                              \
          (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(1)]);                    \
      b0 = VECTOR_UNALIGNED_LOAD(                                              \
          (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(1)]);                    \
      r0 = BIT##op(a0, b0);                                                    \
      VECTOR_UNALIGNED_STORE(                                                  \
          (VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(1)], r0);              \
                                                                               \
      /* Load Third operand */                                                 \
      a0 = VECTOR_UNALIGNED_LOAD(                                              \
          (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(2)]);                    \
      b0 = VECTOR_UNALIGNED_LOAD(                                              \
          (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(2)]);                    \
      r0 = BIT##op(a0, b0);                                                    \
      VECTOR_UNALIGNED_STORE(                                                  \
          (VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(2)], r0);              \
                                                                               \
      /* Load Fourth operand */                                                \
      a0 = VECTOR_UNALIGNED_LOAD(                                              \
          (VECTOR_TYPE *)&s->qwords[i + VECTOR_OFFSET(3)]);                    \
      b0 = VECTOR_UNALIGNED_LOAD(                                              \
          (VECTOR_TYPE *)&t->qwords[i + VECTOR_OFFSET(3)]);                    \
      r0 = BIT##op(a0, b0);                                                    \
      VECTOR_UNALIGNED_STORE(                                                  \
          (VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(3)], r0);              \
    }                                                                          \
                                                                               \
    /* Scalar remainder loop */                                                \
    for (; i < bit_size_in_qwords; i++) {                                      \
      set->qwords[i] = BIT_SCALAR##op(s->qwords[i], t->qwords[i]);             \
    }                                                                          \
  } while (0)
#endif
/* --- End Section 3: SINGLE-BITSET SET OPERATION MACROS --- */

/* ===========================================================================
   SECTION 4: DB SET OPERATION MACROS — CPU
   ===========================================================================
 */

/* Validate two Bit_DB_T operands have the same bitset length */
#define SETOP_DB_CHECKS(bit, bits)                                             \
  assert(bit &&bits);                                                          \
  assert(bit->length == bits->length);

/* Extract raw pointers and dimensions from two Bit_DB_T operands */
#define SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords, \
                       num_targets, n)                                         \
  uint64_t *bit_qwords = bit->qwords;                                          \
  uint64_t *bits_qwords = bits->qwords;                                        \
  unsigned int bit_size_in_qwords = bit->size_in_qwords;                       \
  unsigned int num_targets = bit->nelem;                                       \
  unsigned int n = bits->nelem;

/* Accumulate popcount over a tile buffer (libpopcnt vs WWG dispatch) */
#if !USE_LIBPOPCNT

#define POPULATION_COUNT(count, setop_buffer, buffer_size)                     \
  /* SIMD SYNCHRONIZATION DIRECTIVE */                                         \
  OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, setop_buffer)                           \
  for (int k = 0; k < buffer_size; k++) {                                      \
    count += POPCOUNT(setop_buffer[k]);                                        \
  }

#else

#define POPULATION_COUNT(count, setop_buffer, buffer_size)                     \
  count += popcnt((void *)setop_buffer, buffer_size * sizeof(uint64_t));

#endif

/* --- 1x1 Microkernel (Strictly used for fringes and 1x1 fast path) --- */
#if USE_LIBPOPCNT || BIT_SIMD_PATH_SCALAR
#define setop_count_db_cpu_kernel(a_row, b_row, k_b, k_max, result, op,        \
                                  SIMD_DIRECTIVE, LOAD_MACRO)                  \
  do {                                                                         \
    _Alignas(ALIGNMENT) uint64_t setop_buffer[SETOP_BUFFER_SIZE];              \
    uint64_t count = 0;                                                        \
    size_t l = k_b;                                                            \
    CHUNK_LIMIT(limit, k_b, k_max, SETOP_BUFFER_SIZE)                          \
    for (; l < limit; l += SETOP_BUFFER_SIZE) {                                \
      SIMD_DIRECTIVE                                                           \
      for (int k = 0; k < SETOP_BUFFER_SIZE; k++) {                            \
        setop_buffer[k] = BIT_SCALAR##op(a_row[l + k], b_row[l + k]);          \
      }                                                                        \
      POPULATION_COUNT(count, setop_buffer, SETOP_BUFFER_SIZE)                 \
    }                                                                          \
    for (; l < k_max; l++) {                                                   \
      count += POPCOUNT(BIT_SCALAR##op(a_row[l], b_row[l]));                   \
    }                                                                          \
    result = (int)count;                                                       \
  } while (0)
#else
#define setop_count_db_cpu_kernel(a_row, b_row, k_b, k_max, result, op,        \
                                  SIMD_DIRECTIVE, LOAD_MACRO)                  \
  do {                                                                         \
    uint64_t count = 0;                                                        \
    size_t k_idx = k_b;                                                        \
    CHUNK_LIMIT(limit, k_b, k_max, VECTOR_BLOCK_SIZE)                          \
    VECTOR_TYPE sum0 = SIMDe_ZERO_VECTOR;                                      \
    for (; k_idx < limit; k_idx += VECTOR_BLOCK_SIZE) {                        \
      sum0 = SIMDe_VECTOR_ADD(                                                 \
          sum0,                                                                \
          SIMDe_POPCOUNT(BIT##op(                                              \
              LOAD_MACRO((VECTOR_TYPE *)&a_row[k_idx + VECTOR_OFFSET(0)]),     \
              LOAD_MACRO((VECTOR_TYPE *)&b_row[k_idx + VECTOR_OFFSET(0)]))));  \
                                                                               \
      sum0 = SIMDe_VECTOR_ADD(                                                 \
          sum0,                                                                \
          SIMDe_POPCOUNT(BIT##op(                                              \
              LOAD_MACRO((VECTOR_TYPE *)&a_row[k_idx + VECTOR_OFFSET(1)]),     \
              LOAD_MACRO((VECTOR_TYPE *)&b_row[k_idx + VECTOR_OFFSET(1)]))));  \
                                                                               \
      sum0 = SIMDe_VECTOR_ADD(                                                 \
          sum0,                                                                \
          SIMDe_POPCOUNT(BIT##op(                                              \
              LOAD_MACRO((VECTOR_TYPE *)&a_row[k_idx + VECTOR_OFFSET(2)]),     \
              LOAD_MACRO((VECTOR_TYPE *)&b_row[k_idx + VECTOR_OFFSET(2)]))));  \
                                                                               \
      sum0 = SIMDe_VECTOR_ADD(                                                 \
          sum0,                                                                \
          SIMDe_POPCOUNT(BIT##op(                                              \
              LOAD_MACRO((VECTOR_TYPE *)&a_row[k_idx + VECTOR_OFFSET(3)]),     \
              LOAD_MACRO((VECTOR_TYPE *)&b_row[k_idx + VECTOR_OFFSET(3)]))));  \
    }                                                                          \
    uint64_t sum_array[VECTOR_QWORDS];                                         \
    SIMDe_STORE_VECTOR(sum_array, sum0);                                       \
    for (size_t j_idx = 0; j_idx < VECTOR_QWORDS; j_idx++) {                   \
      count += sum_array[j_idx];                                               \
    }                                                                          \
    for (; k_idx < k_max; k_idx++) {                                           \
      count += POPCOUNT(BIT_SCALAR##op(a_row[k_idx], b_row[k_idx]));           \
    }                                                                          \
    result = (int)count;                                                       \
  } while (0)
#endif

/* --- Parameterized Generic Outer Product Microkernel --- */
#if USE_LIBPOPCNT || BIT_SIMD_PATH_SCALAR
#define setop_count_db_cpu_kernel_outer(a_rows, b_rows, k_b, k_max, results,   \
                                        op, SIMD_DIRECTIVE, LOAD_MACRO)        \
  do {                                                                         \
    /* Per-output contiguous buffer for libpopcnt bulk reduction. */           \
    const int BUF_SZ = SETOP_BUFFER_SIZE;                                      \
    _Alignas(ALIGNMENT)                                                        \
        uint64_t setop_buffer[OUTER_ROW_NUM][OUTER_COL_NUM][BUF_SZ];           \
    uint64_t c[OUTER_ROW_NUM][OUTER_COL_NUM] = {0};                            \
    size_t l = k_b;                                                            \
    CHUNK_LIMIT(limit, k_b, k_max, BUF_SZ)                                     \
    for (; l < limit; l += BUF_SZ) {                                           \
      SIMD_DIRECTIVE                                                           \
      for (int k = 0; k < BUF_SZ; k++) {                                       \
        uint64_t a_values[OUTER_ROW_NUM];                                      \
        uint64_t b_values[OUTER_COL_NUM];                                      \
        for (int x = 0; x < OUTER_ROW_NUM; ++x)                                \
          a_values[x] = a_rows[x][l + k];                                      \
                                                                               \
        for (int y = 0; y < OUTER_COL_NUM; ++y)                                \
          b_values[y] = b_rows[y][l + k];                                      \
                                                                               \
        for (int x = 0; x < OUTER_ROW_NUM; ++x)                                \
          for (int y = 0; y < OUTER_COL_NUM; ++y)                              \
            setop_buffer[x][y][k] = BIT_SCALAR##op(a_values[x], b_values[y]);  \
      }                                                                        \
      for (int x = 0; x < OUTER_ROW_NUM; x++) {                                \
        for (int y = 0; y < OUTER_COL_NUM; y++) {                              \
          POPULATION_COUNT(c[x][y], setop_buffer[x][y], BUF_SZ)                \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (; l < k_max; l++) {                                                   \
      for (int x = 0; x < OUTER_ROW_NUM; x++) {                                \
        for (int y = 0; y < OUTER_COL_NUM; y++) {                              \
          c[x][y] += POPCOUNT(BIT_SCALAR##op(a_rows[x][l], b_rows[y][l]));     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int x = 0; x < OUTER_ROW_NUM; x++) {                                  \
      for (int y = 0; y < OUTER_COL_NUM; y++) {                                \
        results[x][y] = (int)c[x][y];                                          \
      }                                                                        \
    }                                                                          \
  } while (0)
#else
#define setop_count_db_cpu_kernel_outer(a_rows, b_rows, k_b, k_max, results,   \
                                        op, SIMD_DIRECTIVE, LOAD_MACRO)        \
  do {                                                                         \
    uint64_t c[OUTER_ROW_NUM][OUTER_COL_NUM] = {0};                            \
    size_t k_idx = k_b;                                                        \
    /* FIX: Multiply the unroll factor by the vector word width */             \
    const size_t step_size = (size_t)(OUTER_VEC_BLK * VECTOR_QWORDS);          \
    CHUNK_LIMIT(limit, k_b, k_max, step_size)                                  \
    VECTOR_TYPE sum[OUTER_ROW_NUM][OUTER_COL_NUM];                             \
    for (int x = 0; x < OUTER_ROW_NUM; x++) {                                  \
      for (int y = 0; y < OUTER_COL_NUM; y++) {                                \
        sum[x][y] = SIMDe_ZERO_VECTOR;                                         \
      }                                                                        \
    }                                                                          \
    for (; k_idx < limit; k_idx += step_size) {                                \
      for (int u = 0; u < OUTER_VEC_BLK; u++) {                                \
        VECTOR_TYPE a_vectors[OUTER_ROW_NUM];                                  \
        VECTOR_TYPE b_vectors[OUTER_COL_NUM];                                  \
                                                                               \
        for (int x = 0; x < OUTER_ROW_NUM; x++) {                              \
          a_vectors[x] =                                                       \
              LOAD_MACRO((VECTOR_TYPE *)&a_rows[x][k_idx + VECTOR_OFFSET(u)]); \
        }                                                                      \
        for (int y = 0; y < OUTER_COL_NUM; y++) {                              \
          b_vectors[y] =                                                       \
              LOAD_MACRO((VECTOR_TYPE *)&b_rows[y][k_idx + VECTOR_OFFSET(u)]); \
        }                                                                      \
                                                                               \
        for (int x = 0; x < OUTER_ROW_NUM; x++) {                              \
          for (int y = 0; y < OUTER_COL_NUM; y++) {                            \
            sum[x][y] = SIMDe_VECTOR_ADD(                                      \
                sum[x][y],                                                     \
                SIMDe_POPCOUNT(BIT##op(a_vectors[x], b_vectors[y])));          \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    /* Extract Phase */                                                        \
    uint64_t sum_array[OUTER_ROW_NUM][OUTER_COL_NUM][VECTOR_QWORDS];           \
    for (int x = 0; x < OUTER_ROW_NUM; x++) {                                  \
      for (int y = 0; y < OUTER_COL_NUM; y++) {                                \
        SIMDe_STORE_VECTOR(sum_array[x][y], sum[x][y]);                        \
        for (size_t v = 0; v < VECTOR_QWORDS; v++) {                           \
          c[x][y] += sum_array[x][y][v];                                       \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    /* Scalar Fringe */                                                        \
    for (; k_idx < k_max; k_idx++) {                                           \
      for (int x = 0; x < OUTER_ROW_NUM; x++) {                                \
        for (int y = 0; y < OUTER_COL_NUM; y++) {                              \
          c[x][y] +=                                                           \
              POPCOUNT(BIT_SCALAR##op(a_rows[x][k_idx], b_rows[y][k_idx]));    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    for (int x = 0; x < OUTER_ROW_NUM; x++) {                                  \
      for (int y = 0; y < OUTER_COL_NUM; y++) {                                \
        results[x][y] = (int)c[x][y];                                          \
      }                                                                        \
    }                                                                          \
  } while (0)
#endif

/* MACRO ARCHITECTURE AND LOOP DISPATCH */
#if (OUTER_ROW_NUM == 1 && OUTER_COL_NUM == 1)

/* Fast-path Architecture for 1x1 dimensions (Zero fringes, direct 1x1 kernel
 * dispatch) */
#define OMP_CPU_TILE_START_OUTER                                               \
  OMP_CPU_LOOP(1, dynamic)                                                     \
  for (int i_b = 0; i_b < num_targets; i_b += CPU_TILE_BIT) {                  \
    for (int j_b = 0; j_b < n; j_b += CPU_TILE_BITS) {                         \
                                                                               \
      int i_max = (i_b + CPU_TILE_BIT < num_targets) ? i_b + CPU_TILE_BIT      \
                                                     : num_targets;            \
      int j_max = (j_b + CPU_TILE_BITS < n) ? j_b + CPU_TILE_BITS : n;         \
                                                                               \
      for (int i = i_b; i < i_max; i++) {                                      \
        for (int j = j_b; j < j_max; j++) {                                    \
          counts[(uint64_t)i * n + j] = 0;                                     \
        }                                                                      \
      }                                                                        \
                                                                               \
      for (size_t k_b = 0; k_b < bit_size_in_qwords; k_b += K_BLOCK) {         \
        size_t k_max = (k_b + K_BLOCK < bit_size_in_qwords)                    \
                           ? k_b + K_BLOCK                                     \
                           : bit_size_in_qwords;                               \
                                                                               \
        for (int i = i_b; i < i_max; i++) {                                    \
          const uint64_t *restrict a_row =                                     \
              bit_qwords + (uint64_t)i * bit_size_in_qwords;                   \
          for (int j = j_b; j < j_max; j++) {                                  \
            const uint64_t *restrict b_row =                                   \
                bits_qwords + (uint64_t)j * bit_size_in_qwords;

#define OMP_CPU_TILE_END_OUTER(op, SIMD_DIR, LOAD_MACRO)                       \
  int result = 0;                                                              \
  setop_count_db_cpu_kernel(a_row, b_row, k_b, k_max, result, op, SIMD_DIR,    \
                            LOAD_MACRO);                                       \
  counts[(uint64_t)i * n + j] += result;                                       \
  }                                                                            \
  }                                                                            \
  }                                                                            \
  }                                                                            \
  }

#else

/* Generic Tiled Architecture for dimensions > 1 (Outer product arrays + fringe
 * handling) */
#define OMP_CPU_TILE_START_OUTER                                               \
  OMP_CPU_LOOP(1, dynamic)                                                     \
  for (int i_b = 0; i_b < num_targets; i_b += CPU_TILE_BIT) {                  \
    for (int j_b = 0; j_b < n; j_b += CPU_TILE_BITS) {                         \
                                                                               \
      int i_max = (i_b + CPU_TILE_BIT < num_targets) ? i_b + CPU_TILE_BIT      \
                                                     : num_targets;            \
      int j_max = (j_b + CPU_TILE_BITS < n) ? j_b + CPU_TILE_BITS : n;         \
                                                                               \
      for (int i = i_b; i < i_max; i++) {                                      \
        for (int j = j_b; j < j_max; j++) {                                    \
          counts[(uint64_t)i * n + j] = 0;                                     \
        }                                                                      \
      }                                                                        \
                                                                               \
      for (size_t k_b = 0; k_b < bit_size_in_qwords; k_b += K_BLOCK) {         \
        size_t k_max = (k_b + K_BLOCK < bit_size_in_qwords)                    \
                           ? k_b + K_BLOCK                                     \
                           : bit_size_in_qwords;                               \
                                                                               \
        int i = i_b;                                                           \
        for (; i <= i_max - OUTER_ROW_NUM; i += OUTER_ROW_NUM) {               \
          const uint64_t *restrict a_rows[OUTER_ROW_NUM];                      \
          for (int x = 0; x < OUTER_ROW_NUM; x++) {                            \
            a_rows[x] = bit_qwords + (uint64_t)(i + x) * bit_size_in_qwords;   \
          }                                                                    \
                                                                               \
          int j = j_b;                                                         \
          for (; j <= j_max - OUTER_COL_NUM; j += OUTER_COL_NUM) {             \
            const uint64_t *restrict b_rows[OUTER_COL_NUM];                    \
            for (int y = 0; y < OUTER_COL_NUM; y++) {                          \
              b_rows[y] =                                                      \
                  bits_qwords + (uint64_t)(j + y) * bit_size_in_qwords;        \
            }                                                                  \
            int results[OUTER_ROW_NUM][OUTER_COL_NUM];

#define OMP_CPU_TILE_END_OUTER(op, SIMD_DIR, LOAD_MACRO)                       \
  setop_count_db_cpu_kernel_outer(a_rows, b_rows, k_b, k_max, results, op,     \
                                  SIMD_DIR, LOAD_MACRO);                       \
                                                                               \
  for (int x = 0; x < OUTER_ROW_NUM; x++) {                                    \
    for (int y = 0; y < OUTER_COL_NUM; y++) {                                  \
      counts[(uint64_t)(i + x) * n + (j + y)] += results[x][y];                \
    }                                                                          \
  }                                                                            \
  }                                                                            \
  /* J-FRINGE: Resolve the remaining columns using 1x1 kernel */               \
  for (; j < j_max; j++) {                                                     \
    const uint64_t *restrict b_row_f =                                         \
        bits_qwords + (uint64_t)j * bit_size_in_qwords;                        \
    for (int x = 0; x < OUTER_ROW_NUM; x++) {                                  \
      int rf = 0;                                                              \
      setop_count_db_cpu_kernel(a_rows[x], b_row_f, k_b, k_max, rf, op,        \
                                SIMD_DIR, LOAD_MACRO);                         \
      counts[(uint64_t)(i + x) * n + j] += rf;                                 \
    }                                                                          \
  }                                                                            \
  }                                                                            \
  /* I-FRINGE: Resolve the remaining rows using 1x1 kernel */                  \
  for (; i < i_max; i++) {                                                     \
    const uint64_t *restrict a_row_f =                                         \
        bit_qwords + (uint64_t)i * bit_size_in_qwords;                         \
    for (int j_f = j_b; j_f < j_max; j_f++) {                                  \
      const uint64_t *restrict b_row_f =                                       \
          bits_qwords + (uint64_t)j_f * bit_size_in_qwords;                    \
      int rff = 0;                                                             \
      setop_count_db_cpu_kernel(a_row_f, b_row_f, k_b, k_max, rff, op,         \
                                SIMD_DIR, LOAD_MACRO);                         \
      counts[(uint64_t)i * n + j_f] += rff;                                    \
    }                                                                          \
  }                                                                            \
  }                                                                            \
  }                                                                            \
  }

#endif

/* Top-level DB CPU set-operation dispatch (architecture and alignment aware) */
#define setop_count_db_cpu(bit, bits, counts, op, opts)                        \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
                                                                               \
  bool aligned = ALIGN_CHECK(bit_qwords) && ALIGN_CHECK(bits_qwords);          \
  int numthreads = opts.num_cpu_threads;                                       \
  if (numthreads <= 0) {                                                       \
    numthreads = omp_get_max_threads();                                        \
  }                                                                            \
  omp_set_num_threads(numthreads);                                             \
                                                                               \
  if (ARCH_32BIT) {                                                            \
    OMP_CPU_TILE_START_OUTER                                                   \
    OMP_CPU_TILE_END_OUTER(op,                                                 \
                           OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, setop_buffer), \
                           VECTOR_UNALIGNED_LOAD)                              \
  } else {                                                                     \
    if (aligned) {                                                             \
      OMP_CPU_TILE_START_OUTER                                                 \
      OMP_CPU_TILE_END_OUTER(                                                  \
          op, OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, setop_buffer),              \
          VECTOR_ALIGNED_LOAD)                                                 \
    } else {                                                                   \
      OMP_CPU_TILE_START_OUTER                                                 \
      OMP_CPU_TILE_END_OUTER(                                                  \
          op, OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, setop_buffer),              \
          VECTOR_UNALIGNED_LOAD)                                               \
    }                                                                          \
  }

/* --- End Section 4: DB SET OPERATION MACROS — CPU --- */

/* ===========================================================================
   SECTION 5: DB SET OPERATION MACROS — GPU
   ===========================================================================
 */
#ifndef NOGPU

/* Ensure both operands and counts are present on the target device */
#define SETOP_INIT_GPU(bit, bits, counts, opts)                                \
  const int _setop_dev_id = (opts).device_id;                                  \
  const int _setop_upd_1st = (opts).upd_1st_operand;                           \
  const int _setop_upd_2nd = (opts).upd_2nd_operand;                           \
  uint64_t *_setop_bit_qwords = (bit)->qwords;                                 \
  uint64_t *_setop_bits_qwords = (bits)->qwords;                               \
  int *_setop_counts = (counts);                                               \
  const size_t _setop_bit_span = (size_t)(bit)->size_in_qwords * (bit)->nelem; \
  const size_t _setop_bits_span =                                              \
      (size_t)(bits)->size_in_qwords * (bits)->nelem;                          \
  const size_t _setop_counts_span = (size_t)(bit)->nelem * (bits)->nelem;      \
  if (omp_target_is_present(_setop_bit_qwords, _setop_dev_id)) {               \
    if (_setop_upd_1st) {                                                      \
      UPDATE_GPU_ARRAY(to, _setop_bit_qwords, 0, _setop_bit_span,              \
                       _setop_dev_id)                                          \
    }                                                                          \
  } else {                                                                     \
    TARGET_GPU_ARRAY(enter, to, _setop_bit_qwords, 0, _setop_bit_span,         \
                     _setop_dev_id)                                            \
  }                                                                            \
  if (omp_target_is_present(_setop_bits_qwords, _setop_dev_id)) {              \
    if (_setop_upd_2nd) {                                                      \
      UPDATE_GPU_ARRAY(to, _setop_bits_qwords, 0, _setop_bits_span,            \
                       _setop_dev_id)                                          \
    }                                                                          \
  } else {                                                                     \
    TARGET_GPU_ARRAY(enter, to, _setop_bits_qwords, 0, _setop_bits_span,       \
                     _setop_dev_id)                                            \
  }                                                                            \
  if (!omp_target_is_present(_setop_counts, _setop_dev_id)) {                  \
    TARGET_GPU_ARRAY(enter, to, _setop_counts, 0, _setop_counts_span,          \
                     _setop_dev_id)                                            \
  }

/* Launch a target region with `num_targets` teams */
#define OMP_GPU_TEAMS(num_targets, dev_id)                                     \
  _Pragma(STRINGIFY(omp target teams distribute num_teams(num_targets)         \
                        device(dev_id)))

/* Open a parallel region with `n` threads inside a teams region */
#define OMP_GPU_PARALLEL(n) _Pragma(STRINGIFY(omp parallel num_threads(n)))

/* SIMD reduction inside a GPU parallel region */
#define OMP_GPU_SIMD_REDUCTION(reduction_type, reduction_var)                  \
  _Pragma(STRINGIFY(omp simd reduction(reduction_type : reduction_var)))

/* Workshare loop division without an implicit barrier */
#define OMP_GPU_FOR_NOWAIT _Pragma(STRINGIFY(omp for nowait))

/* Release or delete a device array after use */
#define SETOP_FINALIZE_GPU(action, buffer, index1, index2, dev_id)             \
  if (omp_target_is_present(buffer, dev_id)) {                                 \
    _Pragma(STRINGIFY(omp target exit data map(                                \
        action : buffer [index1:index2]) device(dev_id)))                      \
  }

/* Full GPU DB set-operation kernel (team-parallel, SIMD inner loop) */
#define setop_count_db_gpu(bit, bits, counts, op, opts)                        \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
  SETOP_INIT_GPU(bit, bits, counts, opts)                                      \
  OMP_GPU_TEAMS(num_targets, opts.device_id)                                   \
  for (int k = 0; k < num_targets; k++) {                                      \
    uint64_t shift_k = k * bit_size_in_qwords;                                 \
    OMP_GPU_PARALLEL(n) {                                                      \
      OMP_GPU_FOR_NOWAIT                                                       \
      for (unsigned int i = 0; i < n; i++) {                                   \
        uint64_t shift_i = i * bit_size_in_qwords;                             \
        int total_sum_for_i = 0;                                               \
        OMP_GPU_SIMD_REDUCTION(+, total_sum_for_i)                             \
        for (unsigned int j = 0; j < bit_size_in_qwords; j++) {                \
          uint64_t x = bit_qwords[shift_k + j] op bits_qwords[shift_i + j];    \
          total_sum_for_i += (uint32_t)POPCOUNT_GPU(x);                        \
        }                                                                      \
        counts[k * n + i] = total_sum_for_i;                                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  _Pragma(STRINGIFY(omp target exit data map(                                  \
      from : counts [0:num_targets * n]))) if (opts.release_1st_operand) {     \
    SETOP_FINALIZE_GPU(release, bit->qwords, 0,                                \
                       bit_size_in_qwords * num_targets, opts.device_id)       \
  }                                                                            \
  if (opts.release_2nd_operand) {                                              \
    SETOP_FINALIZE_GPU(release, bits->qwords, 0, bit_size_in_qwords * n,       \
                       opts.device_id)                                         \
  }                                                                            \
  if (opts.release_counts) {                                                   \
    SETOP_FINALIZE_GPU(release, counts, 0, num_targets * n, opts.device_id)    \
  }

#endif /* NOGPU */

/* --- End Section 5: DB SET OPERATION MACROS — GPU --- */
