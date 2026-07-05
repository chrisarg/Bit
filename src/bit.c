/*
    A simple uncompressed bitset implementation based on the ADT
    in David Hanson's "C Interfaces and Implementations" book.
    Suitably extended to allow memory for the bitsets to be allocated
    outside the library.


    * Author : Christos Argyropoulos
    * Created : April 1st 2025
    * Copyright : (c) 2025
    * License : BSD-2
*/

/* ===========================================================================
   SECTION 1: INCLUDES
   Standard library headers first, then project headers, then conditional.
   ===========================================================================
 */

#include "bit.h"               // Contains your public API declarations
#include "omp.h"               // For OpenMP parallelization
#include "simde_integration.h" // For SIMD operations
#include <assert.h>            // For assert() validation
#include <limits.h>            // For INT_MAX
#include <stdatomic.h>         // For atomic operations
#include <stdbool.h>           // For bool type (is_Bit_T_allocated)
#include <stdint.h>            // For uintptr_t and UINT64_C macros
#include <stdio.h>             // For printf (if needed for debugging)
#include <stdlib.h>            // For malloc, free
#include <string.h>            // For memset
/*---------------------------------------------------------------------------
  Environmental and configuration macros/defines and enums
----------------------------------------------------------------------------*/
#ifndef USE_LIBPOPCNT
#define USE_LIBPOPCNT 1
#endif

#if USE_LIBPOPCNT
#include "libpopcnt.h"
#else
#endif

/* --- End Section 1: INCLUDES --- */

/* ===========================================================================
   SECTION 2: COMPILE-TIME CONFIGURATION
   Architecture detection, alignment constants, cache-tile tuning,
   type aliases, and GPU-layout state codes.
   ===========================================================================
 */

// Universal bitwise alignment check (Alignment must be a power of 2)
#define IS_ALIGNED_64(ptr) (((uintptr_t)(const void *)(ptr) & 63) == 0)
#define IS_ALIGNED_8(ptr) (((uintptr_t)(const void *)(ptr) & 7) == 0)

// Architecture Detection via Pointer Size
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

/*
  Tune these tiles to fit in L2/L3 cache. e.g., 32x32 or 32x64 for your
  OpenMP CPU implementation.
*/
#ifndef CPU_TILE
#define CPU_TILE_BIT 32
#define CPU_TILE_BITS 32
#else
#define CPU_TILE_BIT CPU_TILE
#define CPU_TILE_BITS CPU_TILE
#endif

#define T Bit_T
#define T_DB Bit_DB_T

/* --- End Section 2: COMPILE-TIME CONFIGURATION --- */

/* ===========================================================================
   SECTION 3: INFRASTRUCTURAL MACROS
   Low-level bit-math helpers and popcount dispatch.
   These must be defined before bit_internal.h is included.
   ===========================================================================
 */

#define STRINGIFY(x) #x // Macro to convert a macro argument to a string

#define BPQW (sizeof(uint64_t) * 8)     // bits per qword
#define BPB (sizeof(unsigned char) * 8) // bits per byte
#define nqwords(len)                                                           \
  ((((len) + BPQW - 1) & (~(BPQW - 1))) / BPQW)          // ceil(len/BPQW)
#define nbytes(len) ((((len) + 8 - 1) & (~(8 - 1))) / 8) // ceil(len/8)

// Buffer size for popcount operations over DB bitsets
#define SETOP_BUFFER_SIZE 1024

// CPU popcount — dispatches to count_WWG (forward-declared in Section 7)
#define POPCOUNT(x) count_WWG((x))

// No-op marker to disable SIMD vectorization in CPU code when needed
#define NO_SIMD /*NO SIMD*/

/* --- End Section 3: INFRASTRUCTURAL MACROS --- */

/* ===========================================================================
   SECTION 4: PRIVATE IMPLEMENTATION MACROS
   All large OpenMP and set-operation macros live in a separate header to
   reduce clutter. They depend on macros defined in Sections 2 and 3 above.
   ===========================================================================
 */

#include "bit_internal.h"

/* --- End Section 4: PRIVATE IMPLEMENTATION MACROS --- */

/* ===========================================================================
   SECTION 5: INTERNAL DATA STRUCTURES AND ENUMS
   Concrete definitions of the opaque types declared in bit.h, plus internal
   bookkeeping structures.
   ===========================================================================
 */

/* --- 5a. Single bitset (Bit_T) ---
 The ADT provides access to the bitset as bytes,or qwords,
 anticipating optimization of the code *down the road*,
 including loading of externally allocated buffers into it
 The interface is effectively the one for Bit_T presented by D. Hanson
 in C Interfaces and Implementations, ISBN 0-201-49841-3 Ch13, 1997.
 Changes made by the author include
 1) the introduction of the set_count operations to avoid
 forming the intermediate Bit_T
 2) using optimized popcount functions and defaulting to the
    WWG algorithm for popcount if the user elects not to use the
    libpopcnt library.
*/
struct T {
  int length;              // capacity of the bitset in bits
  int size_in_bytes;       // number of bytes of the 8 bit container
  int size_in_qwords;      // number of qwords of the 64 bit container
  bool is_Bit_T_allocated; // true if allocated by the library
  unsigned char *bytes;    // pointer to the first byte
  uint64_t *qwords;        // pointer to the first qword
};

/* --- 5b. Packed bitset database (Bit_DB_T) ---
 The ADT provides access to containers of bitsets that pack fixed number of
  the bitset data in a single container for locality of memory access when
  processing large number of bitsets.
*/
struct T_DB {
  int nelem;               // number of bitsets in the packed container
  int length;              // capacity of the bitset in bits
  int size_in_bytes;       // number of bytes of the 8 bit set container
  int size_in_qwords;      // number of qwords of the 64 bit set container
  bool is_Bit_T_allocated; // true if allocated by the library
  unsigned char *bytes;    // pointer to the first byte
  uint64_t *qwords;        // pointer to the first qword
};

/* --- 5c. GPU allocation state tracker and flags --- */
/*
  Single linked list structure to track GPU allocations to manage
  operations that require synchronization (e.g., transposition)
  of Bitset containers in the GPU.
*/

typedef enum {
  STATE_ROW_MAJOR = 0,
  STATE_COL_MAJOR = 1,
} GPUDataLayout;

typedef enum { FLAG_NONE = 0 } GPUDataFlags;

// GPU state transition structure
typedef struct {
  GPUDataLayout from;
  GPUDataFlags to;
  int kernel_id;
} StateTransition;

// This is Thread-Safe Ready
typedef struct GPUAllocationState {
  const void *host_ptr;
  int device_id;

  // The Partitioned Bitfield: holds BOTH GPUDataLayout and GPUDataFlags
  _Atomic uint32_t state_word;

  // --- MULTITHREADING EXTENSIBILITY ---
  // Even in multi-processing, these ensure absolute safety if you
  // ever spawn threads inside your processes later.
  _Atomic int transition_lock; // 0 = Free, 1 = Currently transposing
  _Atomic int
      active_users; // Reader count to prevent layout shifts while in use

  struct GPUAllocationState *next;
} GPUAllocationState;

/* --- End Section 5: INTERNAL DATA STRUCTURES AND ENUMS --- */

/* ===========================================================================
   SECTION 6: STATIC DATA
   File-scope constants and thread-safety primitives through FSMs.
   ===========================================================================
 */

// Byte masks used by the bit-range API (Bit_set, Bit_clear, Bit_not)
static unsigned const char msbmask[] = {
    0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80,
};

static unsigned const char lsbmask[] = {0x01, 0x03, 0x07, 0x0F,
                                        0x1F, 0x3F, 0x7F, 0xFF};

/* --- End Section 6: STATIC DATA --- */

/* ===========================================================================
   SECTION 7: INTERNAL FUNCTION FORWARD DECLARATIONS
   All static helpers are declared here so they can be used in any order
   below without requiring specific definition ordering.
   ===========================================================================
 */

// Bitset copy helper (used by single-bitset set-operation macros)
// Forward declarations
static T copy(T t);
static inline uint64_t count_WWG(uint64_t x);
static void GPU_transpose_kernel(uint64_t *bits, size_t rows, size_t columns,
                                 int device_id);
static void *portable_aligned_calloc(size_t alignment, size_t size);
static inline uint64_t tree_adder(uint64_t v);

/* TODO: add new CPU/GPU helper forward declarations here */

/* --- End Section 7: INTERNAL FUNCTION FORWARD DECLARATIONS --- */

/* ===========================================================================
   SECTION 8: INTERNAL HELPER FUNCTION DEFINITIONS
   Low-level helpers: popcount algorithms, memory allocation, bitset copy.
   ===========================================================================
 */

/* --- 8a. Bitset copy --- */

static T copy(T t) {
  T set;
  assert(t);
  set = Bit_new(t->length);
  if (t->length > 0) {
    memcpy(set->bytes, t->bytes, t->size_in_bytes);
  }
  return set;
}

/* --- 8b. Wilks-Wheeler-Gill (WWG) popcount ---
   Highly portable and fast.
   References:
     https://en.wikipedia.org/wiki/The_Preparation_of_Programs_for_an_Electronic_Digital_Computer
     https://arxiv.org/abs/1611.07612
     https://github.com/kimwalisch/libpopcnt/tree/master
*/
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

/* --- 8c. Tree-adder popcount ---
   Reference: https://metacpan.org/pod/Bit::Fast
*/
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

/* --- 8d. Portable aligned calloc ---
   Allocates `size` bytes aligned to `alignment` and zeroes the memory.
   Does not rely on platform-specific APIs (posix_memalign, _aligned_malloc);
   instead uses pointer arithmetic for portability.
*/

static void *portable_aligned_calloc(size_t alignment, size_t size) {

  assert(alignment > sizeof(void *));
  void *ptr = NULL;
  // Fallback using malloc + offset
  size_t offset = alignment - 1 + sizeof(void *);
  void *original = malloc(size + offset);
  if (!original)
    return NULL;
  // Align the pointer to the specified alignment

  uintptr_t storage_addr = (uintptr_t)original + offset;
  storage_addr &= ~(alignment - 1); // Align down
  void *aligned_ptr = (void *)storage_addr;
  void **store_ptr = (void **)aligned_ptr - 1;
  *store_ptr = original; // Store the original pointer

  ptr = aligned_ptr;

  // Zero the allocated memory if allocation was successful
  if (ptr) {
    memset(ptr, 0, size);
  }

  return ptr;
}

/* --- End Section 8: INTERNAL HELPER FUNCTION DEFINITIONS --- */

/* ===========================================================================
   SECTION 9: GPU KERNEL HELPERS
   Device-side data transformation routines and popcounts.
   ===========================================================================
 */

// Make popcount functions available on GPU device targets
#pragma omp declare target(count_WWG)
#pragma omp declare target(tree_adder)
// GPU popcount alias — change this line to swap the GPU popcount implementation
#define POPCOUNT_GPU count_WWG

/* --- End Section 9: GPU KERNEL HELPERS --- */

/* ===========================================================================
   SECTION 10: PUBLIC API — SINGLE BITSET (Bit_T)
   ===========================================================================
 */

/* --- 10a. Lifecycle: create, destroy, load from external buffer --- */

T Bit_new(int length) {
  assert(length > 0);
  assert(length < INT_MAX); // limit to 2^30 bits
  T set = malloc(sizeof(*set));
  set->length = length;

  set->size_in_qwords = nqwords(length);
  set->size_in_bytes = set->size_in_qwords * BPQW / BPB;

  set->qwords = calloc(set->size_in_bytes, sizeof(unsigned char));
  assert(set->qwords != NULL);

  set->bytes = (unsigned char *)set->qwords;

  set->is_Bit_T_allocated = true; // allocated by the library
  return set;
}

// return a pointer to the original buffer (if externally loaded) or NULL
// otherwise
void *Bit_free(T *set) {
  assert(set && *set);
  void *original_location = (void *)(*set)->qwords;
  if ((*set)->is_Bit_T_allocated) {
    original_location = NULL;
    free((*set)->qwords);
    (*set)->qwords = NULL;
    (*set)->bytes = NULL; // set bytes to NULL after freeing qwords
  }
  free(*set);
  *set = NULL;
  return original_location;
}

T Bit_load(int length, void *buffer) {
  assert(length > 0);
  assert(length < INT_MAX); // limit to 2^30 bits
  assert(buffer != NULL);

  T set = malloc(sizeof(*set));
  set->length = length;

  set->size_in_qwords = nqwords(length);
  set->size_in_bytes = set->size_in_qwords * BPQW / BPB;

  set->bytes = (unsigned char *)buffer;
  set->qwords = (uint64_t *)buffer; // set qwords to point to the buffer
  set->is_Bit_T_allocated = false;  // not allocated by the library
  return set;
}

int Bit_extract(T set, void *buffer) {
  assert(set);
  assert(buffer != NULL);
  // Copy the bytes from the bitset to the buffer
  memcpy(buffer, set->bytes, set->size_in_bytes);
  return set->size_in_bytes; // return the number of bytes written
}

/* --- 10b. Properties --- */

int Bit_length(T set) {
  assert(set);
  return set->length;
}

int Bit_count(T set) {
  assert(set);
  int length = 0;
#if !USE_LIBPOPCNT
  size_t limit = (set->size_in_qwords / VECTOR_BLOCK_SIZE) * VECTOR_BLOCK_SIZE;
  size_t i = 0;
  VECTOR_TYPE sum0 = SIMDe_ZERO_VECTOR;
  VECTOR_TYPE sum1 = SIMDe_ZERO_VECTOR;
  VECTOR_TYPE sum2 = SIMDe_ZERO_VECTOR;
  VECTOR_TYPE sum3 = SIMDe_ZERO_VECTOR;
  for (; i < limit; i += VECTOR_BLOCK_SIZE) {
    VECTOR_TYPE a0 =
        VECTOR_UNALIGNED_LOAD((VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(0)]);
    VECTOR_TYPE a1 =
        VECTOR_UNALIGNED_LOAD((VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(1)]);
    VECTOR_TYPE a2 =
        VECTOR_UNALIGNED_LOAD((VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(2)]);
    VECTOR_TYPE a3 =
        VECTOR_UNALIGNED_LOAD((VECTOR_TYPE *)&set->qwords[i + VECTOR_OFFSET(3)]);

    sum0 = SIMDe_VECTOR_ADD(sum0, SIMDe_POPCOUNT(a0));
    sum1 = SIMDe_VECTOR_ADD(sum1, SIMDe_POPCOUNT(a1));
    sum2 = SIMDe_VECTOR_ADD(sum2, SIMDe_POPCOUNT(a2));
    sum3 = SIMDe_VECTOR_ADD(sum3, SIMDe_POPCOUNT(a3));
  }
  // Horizontal sum of the vector elements
  sum0 = SIMDe_VECTOR_ADD(sum0, sum1);
  sum2 = SIMDe_VECTOR_ADD(sum2, sum3);
  sum0 = SIMDe_VECTOR_ADD(sum0, sum2);
  uint64_t sum_array[VECTOR_QWORDS];
  SIMDe_STORE_VECTOR(sum_array, sum0);
  for (size_t j = 0; j < VECTOR_QWORDS; j++) {
    length += sum_array[j];
  }

  // Handle remaining elements
  for (; i < set->size_in_qwords; i++) {
    length += POPCOUNT(set->qwords[i]);
  }
#else
  length = (int)popcnt(set->bytes, set->size_in_bytes);
#endif
  return length;
}

int Bit_buffer_size(int length) {
  assert(length > 0);
  return nqwords(length) * BPQW / BPB;
}

/* --- 10c. Member operations (set, clear, get, map individual bits) --- */

void Bit_aset(T set, int indices[], int n) {
  assert(set);
  assert(indices);
  for (int i = 0; i < n; i++) {
    assert(indices[i] >= 0 && indices[i] < set->length);
    set->bytes[indices[i] / BPB] |= 1 << (indices[i] % BPB);
  }
}
void Bit_aclear(T set, int indices[], int n) {
  assert(set);
  assert(indices);
  for (int i = 0; i < n; i++) {
    assert(indices[i] >= 0 && indices[i] < set->length);
    set->bytes[indices[i] / BPB] &= ~(1 << (indices[i] % BPB));
  }
}
void Bit_bset(T set, int index) {
  assert(set);
  assert(index >= 0 && index < set->length);
  set->bytes[index / BPB] |= 1 << (index % BPB);
}

void Bit_bclear(T set, int index) {
  assert(set);
  assert(index >= 0 && index < set->length);
  set->bytes[index / BPB] &= ~(1 << (index % BPB));
}

void Bit_clear(T set, int lo, int hi) {
  assert(set);
  assert(0 <= lo && hi < set->length);
  assert(lo <= hi);
  if (lo / 8 < hi / 8) {
    // clear the most significant bits in byte lo/8
    set->bytes[lo / 8] &= ~msbmask[lo % 8];
    // clear the least significant bits in byte hi/8
    set->bytes[hi / 8] &= ~lsbmask[hi % 8];
    // clear the bits in between
    for (int i = lo / 8 + 1; i < hi / 8; i++)
      set->bytes[i] = 0;

  } else // lo and hi are in the same byte
    set->bytes[lo / 8] &= ~(msbmask[lo % 8] & lsbmask[hi % 8]);
}
int Bit_get(T set, int index) {
  assert(set);
  assert(index >= 0 && index < set->length);
  return ((set->bytes[index / BPB] >> (index % BPB)) & 1);
}

void Bit_map(T set, void apply(int n, int bit, void *cl), void *cl) {
  assert(set);
  for (int i = 0; i < set->length; i++) {
    apply(i, ((set->bytes[i / BPB] >> (i % BPB)) & 1), cl);
  }
}

void Bit_not(T set, int lo, int hi) {
  assert(set);
  assert(0 <= lo && hi < set->length);
  assert(lo <= hi);
  if (lo / 8 < hi / 8) {
    // clear the most significant bits in byte lo/8
    set->bytes[lo / 8] ^= msbmask[lo % 8];
    // clear the least significant bits in byte hi/8
    set->bytes[hi / 8] ^= lsbmask[hi % 8];
    // clear the bits in between
    for (int i = lo / 8 + 1; i < hi / 8; i++)
      set->bytes[i] = ~set->bytes[i];

  } else // lo and hi are in the same byte
    set->bytes[lo / 8] ^= (msbmask[lo % 8] & lsbmask[hi % 8]);
}
int Bit_put(T set, int index, int bit) {
  int prev;
  assert(set);
  assert(bit == 0 || bit == 1);
  assert(0 <= index && index < set->length);
  prev = ((set->bytes[index / BPB] >> (index % BPB)) & 1);
  if (bit == 1)
    set->bytes[index / BPB] |= 1 << (index % BPB);
  else
    set->bytes[index / BPB] &= ~(1 << (index % BPB));
  return prev;
}

void Bit_set(T set, int lo, int hi) {
  assert(set);
  assert(0 <= lo && hi < set->length);
  assert(lo <= hi);
  if (lo / 8 < hi / 8) {
    // set the most significant bits in byte lo/8
    set->bytes[lo / 8] |= msbmask[lo % 8];
    // clear the least significant bits in byte hi/8
    set->bytes[hi / 8] |= lsbmask[hi % 8];
    // clear the bits in between
    for (int i = lo / 8 + 1; i < hi / 8; i++)
      set->bytes[i] = 0xFF;

  } else // lo and hi are in the same byte
    set->bytes[lo / 8] |= (msbmask[lo % 8] & lsbmask[hi % 8]);
}
/* --- 10d. Comparisons --- */

int Bit_eq(T s, T t) {
  assert(s && t);
  assert(s->length == t->length);
  for (int i = s->size_in_qwords; --i >= 0;)
    if (s->qwords[i] != t->qwords[i])
      return 0;
  return 1;
}

int Bit_leq(T s, T t) {
  assert(s && t);
  assert(s->length == t->length);
  for (int i = s->size_in_qwords; --i >= 0;)
    if ((s->qwords[i] & ~t->qwords[i]) != 0)
      return 0;
  return 1;
}

int Bit_lt(T s, T t) {
  assert(s && t);
  assert(s->length == t->length);
  int lt = 0;
  for (int i = s->size_in_qwords; --i >= 0;)
    if ((s->qwords[i] & ~t->qwords[i]) != 0)
      return 0;
    else if ((s->qwords[i] & t->qwords[i]) != 0)
      lt |= 1;
  return lt;
}
/* --- 10e. Set operations (return a new Bit_T) --- */

T Bit_diff(T s, T t) {
  setop_validate(Bit_new(s->length), copy(t), copy(s));
  T set = Bit_new(s->length);
  setop(set, _XOR, s, t);
  return set;
}
T Bit_minus(T s, T t) {
  setop_validate(Bit_new(s->length), Bit_new(t->length), copy(s));
  T set = Bit_new(s->length);
  setop(set, _AND_NOT, s, t);
  return set;
}
T Bit_inter(T s, T t) {
  setop_validate(copy(t), Bit_new(t->length), Bit_new(s->length));
  T set = Bit_new(s->length);
  setop(set, _AND, s, t);
  return set;
}

T Bit_union(T s, T t) {
  setop_validate(copy(t), copy(t), copy(s));
  T set = Bit_new(s->length);
  setop(set, _OR, s, t);
  return set;
}

/* --- 10f. Set operations (return population count of result) --- */

int Bit_diff_count(T s, T t) {
  setop_validate(0, Bit_count(t), Bit_count(s));
  setop_count(_XOR, s, t);
}
int Bit_minus_count(T s, T t) {
  setop_validate(0, 0, Bit_count(s));
  setop_count(_AND_NOT, s, t);
}
int Bit_inter_count(T s, T t) {
  setop_validate(Bit_count(t), 0, 0);
  setop_count(_AND, s, t);
}
int Bit_union_count(T s, T t) {
  setop_validate(Bit_count(t), Bit_count(t), Bit_count(s));
  setop_count(_OR, s, t);
}

void print_Bit_configuration(void) {

  printf("CPU_TILE : %d, GPU_TILE_J: %d, GPU_ILP: %d\n", CPU_TILE_BIT,
         GPU_TILE_J, GPU_ILP);
  printf("Using LIBPOPCNT: %s\n", (USE_LIBPOPCNT == 1) ? "Yes" : "No");
  printf("CPU_TILE_BIT %d, CPU_TILE_BITS: %d\n", CPU_TILE_BIT, CPU_TILE_BITS);
}

/* --- End Section 10: PUBLIC API — SINGLE BITSET --- */

/* ===========================================================================
   SECTION 11: PUBLIC API — BITSET DATABASE (Bit_DB_T)
   ===========================================================================
 */

/* --- 11a. Lifecycle: create, destroy, load --- */

T_DB BitDB_new(int length, int num_of_bitsets) {
  assert(length > 0);
  assert(num_of_bitsets > 0);
  assert(num_of_bitsets < INT_MAX); // limit to 2^30 bitsets
  assert(length < INT_MAX);         // limit to 2^30 bits

  T_DB set = malloc(sizeof(*set));
  set->length = length;
  set->nelem = num_of_bitsets;

  set->size_in_qwords = nqwords(length);
  set->size_in_bytes = set->size_in_qwords * BPQW / BPB;

  size_t size_in_bytes = (size_t)set->size_in_bytes * num_of_bitsets;

  // Allocate aligned memory for the bitsets in the database
  set->qwords = portable_aligned_calloc(ALIGNMENT, size_in_bytes);
  assert(set->qwords != NULL);

  set->bytes = (unsigned char *)set->qwords;
  set->is_Bit_T_allocated = true; // allocated by the library
  return set;
}

// return a pointer to the original buffer (if externally loaded) or NULL
// otherwise
void *BitDB_free(T_DB *set) {
  assert(set && *set);
  void *original_location = (void *)(*set)->qwords;
  // complex deallocation logic to handle aligned allocation and external
  // buffers
  if ((*set)->is_Bit_T_allocated) {
    original_location = (void *)((void **)(*set)->qwords - 1);
    void *original_block = *(void **)original_location;
    free(original_block);
    original_block = original_location = NULL;
    (*set)->qwords = NULL;
    (*set)->bytes = NULL; // set bytes to NULL after freeing qwords
  }
  free(*set);
  *set = NULL;
  return original_location;
}

T_DB BitDB_load(int length, int num_of_bitsets, void *buffer) {
  assert(length > 0);
  assert(num_of_bitsets > 0);
  assert(num_of_bitsets < INT_MAX); // limit to 2^30 bitsets
  assert(length < INT_MAX);         // limit to 2^30 bits
  assert(buffer != NULL);

  T_DB set = malloc(sizeof(*set));
  set->length = length;
  set->nelem = num_of_bitsets;

  set->size_in_qwords = nqwords(length);
  set->size_in_bytes = set->size_in_qwords * BPQW / BPB;

  set->bytes = (unsigned char *)buffer;
  set->qwords = (uint64_t *)buffer; // set qwords to point to the buffer
  set->is_Bit_T_allocated = false;  // not allocated by the library
  return set;
}

/* --- 11b. Properties --- */

int BitDB_length(T_DB set) {
  assert(set);
  return set->length;
}

int BitDB_nelem(T_DB set) {
  assert(set);
  return set->nelem;
}

int BitDB_count_at(T_DB set, int index) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  int count = 0;
#if !USE_LIBPOPCNT
  uint64_t *qwords = set->qwords + index * set->size_in_qwords;
  for (int i = 0; i < set->size_in_qwords; i++)
    count += POPCOUNT(qwords[i]);
#else
  count =
      (int)popcnt(set->bytes + index * set->size_in_bytes, set->size_in_bytes);
#endif
  return count;
}

int *BitDB_count(T_DB set) {
  assert(set);
  int *counts = malloc(set->nelem * sizeof(int));
  assert(counts != NULL);
#if !USE_LIBPOPCNT
  uint64_t *qwords = set->qwords;
  for (int i = 0; i < set->nelem; i++, qwords += set->size_in_qwords) {
    int count = 0;
    for (int j = 0; j < set->size_in_qwords; j++)
      count += POPCOUNT(qwords[j]);
    counts[i] = count;
  }
#else
  unsigned char *bytes = set->bytes;
  for (int i = 0; i < set->nelem; i++, bytes += set->size_in_bytes)
    counts[i] = (int)popcnt(bytes, set->size_in_bytes);

#endif
  return counts;
}

/* --- 11c. Element access and bulk operations --- */

void BitDB_clear_at(T_DB set, int index) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memset(set->bytes + shift, 0, set->size_in_bytes);
}

void BitDB_clear(T_DB set) {
  assert(set);
  size_t size_in_bytes = (size_t)set->nelem;
  size_in_bytes *= set->size_in_bytes; // calculate the total size
  memset(set->bytes, 0, size_in_bytes);
}

T BitDB_get_from(T_DB set, int index) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  T bitset = Bit_new(set->length);
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  // Copy the bytes from the set to the new bitset
  memcpy(bitset->bytes, set->bytes + shift, set->size_in_bytes);
  return bitset;
}

void BitDB_put_at(T_DB set, int index, T bitset) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  assert(bitset);
  assert(bitset->length == set->length);
  // Copy the bytes from the bitset to the set
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memcpy(set->bytes + shift, bitset->bytes, set->size_in_bytes);
}

void BitDB_extract_from(T_DB set, int index, void *buffer) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  assert(buffer != NULL);
  // Copy the bytes from the set to the buffer
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memcpy(buffer, set->bytes + shift, set->size_in_bytes);
}

void BitDB_replace_at(T_DB set, int index, void *buffer) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  assert(buffer != NULL);
  // Copy the bytes from the buffer to the set
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memcpy(set->bytes + shift, buffer, set->size_in_bytes);
}

/* --- 11d. CPU set operations (allocate and return counts buffer) --- */

int *BitDB_inter_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_inter_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_inter_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {

  setop_count_db_cpu(bit, bits, counts, _AND, opts);
}

int *BitDB_inter_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_inter_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_inter_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, &, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _AND, opts)
#endif
}

int *BitDB_union_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_union_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_union_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, _OR, opts);
}

int *BitDB_union_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_union_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_union_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_union_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, |, opts);
#else
  setop_count_db_cpu(bit, bits, counts, _OR, opts);
#endif
}

int *BitDB_diff_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_diff_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_diff_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, _XOR, opts);
}

int *BitDB_diff_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_diff_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_diff_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_diff_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, ^, opts)
#else
  setop_count_db_cpu(bit, bits, counts, _XOR, opts)
#endif
}

int *BitDB_minus_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  BitDB_minus_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_minus_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, _AND_NOT, opts);
}

int *BitDB_minus_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
#ifndef NOGPU
  BitDB_minus_count_store_gpu(bit, bits, counts, opts);
#else
  BitDB_minus_count_store_cpu(bit, bits, counts, opts);
#endif
  return counts;
}

void BitDB_minus_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                 SETOP_COUNT_OPTS opts) {
#ifndef NOGPU
  setop_count_db_gpu(bit, bits, counts, &~, opts)
#else
  setop_count_db_cpu(bit, bits, counts, _AND_NOT, opts);
#endif
}

/* --- End Section 11: PUBLIC API — BITSET DATABASE --- */
