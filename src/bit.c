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

#include "bit_internal.h"

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
static void *portable_aligned_calloc(size_t alignment, size_t size);

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

/* --- 8b. Portable aligned calloc ---
   Allocates `size` bytes aligned to `alignment` and zeroes the memory.
   Does not rely on platform-specific APIs (posix_memalign, _aligned_malloc);
   instead uses pointer arithmetic for portability.
*/

static void *portable_aligned_calloc(size_t alignment, size_t size) {

  assert(alignment > sizeof(void *));
  void *ptr = NULL;
  // Fallback using malloc + offset
  size_t offset;
  size_t allocation_size;
  if (alignment > SIZE_MAX - sizeof(void *))
    return NULL;
  offset = alignment - 1 + sizeof(void *);
  if (offset > SIZE_MAX - size)
    return NULL;
  allocation_size = size + offset;
  void *original = malloc(allocation_size);
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
   SECTION 10: PUBLIC API — SINGLE BITSET (Bit_T)
   ===========================================================================
 */

/* --- 10a. Lifecycle: create, destroy, load from external buffer --- */

T Bit_new(size_t length) {
  assert(length > 0);
  size_t qwords;
  size_t size_in_bytes;
  if (!bit_qwords_for_bits(length, &qwords) ||
      !bit_size_mul(qwords, sizeof(uint64_t), &size_in_bytes))
    return NULL;
  T set = malloc(sizeof(*set));
  if (!set)
    return NULL;
  set->length = length;

  set->size_in_qwords = qwords;
  set->size_in_bytes = size_in_bytes;

  set->qwords = calloc(set->size_in_bytes, sizeof(unsigned char));
  if (!set->qwords) {
    free(set);
    return NULL;
  }

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

T Bit_load(size_t length, void *buffer) {
  assert(length > 0);
  assert(buffer != NULL);
  size_t qwords;
  size_t size_in_bytes;
  if (!bit_qwords_for_bits(length, &qwords) ||
      !bit_size_mul(qwords, sizeof(uint64_t), &size_in_bytes))
    return NULL;

  T set = malloc(sizeof(*set));
  if (!set)
    return NULL;
  set->length = length;

  set->size_in_qwords = qwords;
  set->size_in_bytes = size_in_bytes;

  set->bytes = (unsigned char *)buffer;
  set->qwords = (uint64_t *)buffer; // set qwords to point to the buffer
  set->is_Bit_T_allocated = false;  // not allocated by the library
  return set;
}

size_t Bit_extract(T set, void *buffer) {
  assert(set);
  assert(buffer != NULL);
  // Copy the bytes from the bitset to the buffer
  memcpy(buffer, set->bytes, set->size_in_bytes);
  return set->size_in_bytes; // return the number of bytes written
}

/* --- 10b. Properties --- */

size_t Bit_length(T set) {
  assert(set);
  return set->length;
}

uint64_t Bit_count(T set) {
  assert(set);
  uint64_t length = 0;
#if !USE_LIBPOPCNT
#if BIT_SIMD_PATH_SCALAR
  /* Scalar fallback (no vector types available) */
  for (unsigned int k = 0; k < set->size_in_qwords; k++) {
    length += POPCOUNT(set->qwords[k]);
  }
#else
  /* Dispatch on pointer alignment, mirroring setop_count_db_cpu */
  if (ALIGN_CHECK(set->qwords)) {
    bit_count_body(length, set, set->size_in_qwords, VECTOR_ALIGNED_LOAD);
  } else {
    bit_count_body(length, set, set->size_in_qwords, VECTOR_UNALIGNED_LOAD);
  }
#endif
#else
  length = (int)popcnt(set->bytes, set->size_in_bytes);
#endif
  return length;
}

size_t Bit_buffer_size(size_t length) {
  assert(length > 0);
  size_t qwords;
  size_t size_in_bytes;
  if (!bit_qwords_for_bits(length, &qwords) ||
      !bit_size_mul(qwords, sizeof(uint64_t), &size_in_bytes))
    return 0;
  return size_in_bytes;
}

/* --- 10c. Member operations (set, clear, get, map individual bits) --- */

void Bit_aset(T set, size_t indices[], size_t n) {
  assert(set);
  assert(indices);
  for (size_t i = 0; i < n; i++) {
    assert(indices[i] < set->length);
    set->bytes[indices[i] / BPB] |= 1 << (indices[i] % BPB);
  }
}
void Bit_aclear(T set, size_t indices[], size_t n) {
  assert(set);
  assert(indices);
  for (size_t i = 0; i < n; i++) {
    assert(indices[i] < set->length);
    set->bytes[indices[i] / BPB] &= ~(1 << (indices[i] % BPB));
  }
}
void Bit_bset(T set, size_t index) {
  assert(set);
  assert(index < set->length);
  set->bytes[index / BPB] |= 1 << (index % BPB);
}

void Bit_bclear(T set, size_t index) {
  assert(set);
  assert(index < set->length);
  set->bytes[index / BPB] &= ~(1 << (index % BPB));
}

void Bit_clear(T set, size_t lo, size_t hi) {
  assert(set);
  assert(hi < set->length);
  assert(lo <= hi);
  if (lo / 8 < hi / 8) {
    // clear the most significant bits in byte lo/8
    set->bytes[lo / 8] &= ~msbmask[lo % 8];
    // clear the least significant bits in byte hi/8
    set->bytes[hi / 8] &= ~lsbmask[hi % 8];
    // clear the bits in between
    for (size_t i = lo / 8 + 1; i < hi / 8; i++)
      set->bytes[i] = 0;

  } else // lo and hi are in the same byte
    set->bytes[lo / 8] &= ~(msbmask[lo % 8] & lsbmask[hi % 8]);
}
int Bit_get(T set, size_t index) {
  assert(set);
  assert(index < set->length);
  return ((set->bytes[index / BPB] >> (index % BPB)) & 1);
}

void Bit_map(T set, void apply(size_t n, int bit, void *cl), void *cl) {
  assert(set);
  for (size_t i = 0; i < set->length; i++) {
    apply(i, ((set->bytes[i / BPB] >> (i % BPB)) & 1), cl);
  }
}

void Bit_not(T set, size_t lo, size_t hi) {
  assert(set);
  assert(hi < set->length);
  assert(lo <= hi);
  if (lo / 8 < hi / 8) {
    // clear the most significant bits in byte lo/8
    set->bytes[lo / 8] ^= msbmask[lo % 8];
    // clear the least significant bits in byte hi/8
    set->bytes[hi / 8] ^= lsbmask[hi % 8];
    // clear the bits in between
    for (size_t i = lo / 8 + 1; i < hi / 8; i++)
      set->bytes[i] = ~set->bytes[i];

  } else // lo and hi are in the same byte
    set->bytes[lo / 8] ^= (msbmask[lo % 8] & lsbmask[hi % 8]);
}
int Bit_put(T set, size_t index, int bit) {
  int prev;
  assert(set);
  assert(bit == 0 || bit == 1);
  assert(index < set->length);
  prev = ((set->bytes[index / BPB] >> (index % BPB)) & 1);
  if (bit == 1)
    set->bytes[index / BPB] |= 1 << (index % BPB);
  else
    set->bytes[index / BPB] &= ~(1 << (index % BPB));
  return prev;
}

void Bit_set(T set, size_t lo, size_t hi) {
  assert(set);
  assert(hi < set->length);
  assert(lo <= hi);
  if (lo / 8 < hi / 8) {
    // set the most significant bits in byte lo/8
    set->bytes[lo / 8] |= msbmask[lo % 8];
    // clear the least significant bits in byte hi/8
    set->bytes[hi / 8] |= lsbmask[hi % 8];
    // clear the bits in between
    for (size_t i = lo / 8 + 1; i < hi / 8; i++)
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

uint64_t Bit_diff_count(T s, T t) {
  setop_validate(0, Bit_count(t), Bit_count(s));
  setop_count(_XOR, s, t);
}
uint64_t Bit_minus_count(T s, T t) {
  setop_validate(0, 0, Bit_count(s));
  setop_count(_AND_NOT, s, t);
}
uint64_t Bit_inter_count(T s, T t) {
  setop_validate(Bit_count(t), 0, 0);
  setop_count(_AND, s, t);
}
uint64_t Bit_union_count(T s, T t) {
  setop_validate(Bit_count(t), Bit_count(t), Bit_count(s));
  setop_count(_OR, s, t);
}

#ifndef NOGPU
extern void _Bit_gpu_configuration(void);
#endif

void print_Bit_configuration(void) {
  printf("==========================================\n");
  printf("        System Bit Configuration          \n");
  printf("==========================================\n");

  // Using fixed-width specifiers for clean alignment (e.g., %-20s)
  printf(" %-20s : %d\n", "CPU_TILE_BIT", CPU_TILE_BIT);
  printf(" %-20s : %d\n", "CPU_TILE_BITS", CPU_TILE_BITS);
  printf(" %-20s : %d\n", "GPU_TILE_J", GPU_TILE_J);
  printf(" %-20s : %d\n", "GPU_ILP", GPU_ILP);
  printf(" %-20s : %d\n", "K_BLOCK", K_BLOCK);
  printf(" %-20s : %d\n", "SETOP_BUFFER_SIZE", SETOP_BUFFER_SIZE);
  printf(" %-20s : %d\n", "OUTER_ROW_NUM", OUTER_ROW_NUM);
  printf(" %-20s : %d\n", "OUTER_COL_NUM", OUTER_COL_NUM);
  printf(" %-20s : %d\n", "OUTER_VEC_BLK", OUTER_VEC_BLK);

  printf("------------------------------------------\n");
  printf(" %-20s : %s\n", "Using LIBPOPCNT", USE_LIBPOPCNT ? "Yes" : "No");
#ifndef NOGPU
  _Bit_gpu_configuration();
#endif

#ifdef _OPENMP
  printf(" %-20s : %d\n", "OpenMP version", _OPENMP);
#endif
  printf("==========================================\n");
}
/* --- End Section 10: PUBLIC API — SINGLE BITSET --- */

/* ===========================================================================
   SECTION 11: PUBLIC API — BITSET DATABASE (Bit_DB_T)
   ===========================================================================
 */

/* --- 11a. Lifecycle: create, destroy, load --- */

T_DB BitDB_new(size_t length, size_t num_of_bitsets) {
  assert(length > 0);
  assert(num_of_bitsets > 0);
  size_t qwords;
  size_t size_in_bytes;
  size_t total_size;
  if (!bit_qwords_for_bits(length, &qwords) ||
      !bit_size_mul(qwords, sizeof(uint64_t), &size_in_bytes) ||
      !bit_size_mul(size_in_bytes, num_of_bitsets, &total_size))
    return NULL;

  T_DB set = malloc(sizeof(*set));
  if (!set)
    return NULL;
  set->length = length;
  set->nelem = num_of_bitsets;

  set->size_in_qwords = qwords;
  set->size_in_bytes = size_in_bytes;

  // Allocate aligned memory for the bitsets in the database
  set->qwords = portable_aligned_calloc(ALIGNMENT, total_size);
  if (!set->qwords) {
    free(set);
    return NULL;
  }

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

T_DB BitDB_load(size_t length, size_t num_of_bitsets, void *buffer) {
  assert(length > 0);
  assert(num_of_bitsets > 0);
  assert(buffer != NULL);
  size_t qwords;
  size_t size_in_bytes;
  if (!bit_qwords_for_bits(length, &qwords) ||
      !bit_size_mul(qwords, sizeof(uint64_t), &size_in_bytes))
    return NULL;

  T_DB set = malloc(sizeof(*set));
  if (!set)
    return NULL;
  set->length = length;
  set->nelem = num_of_bitsets;

  set->size_in_qwords = qwords;
  set->size_in_bytes = size_in_bytes;

  set->bytes = (unsigned char *)buffer;
  set->qwords = (uint64_t *)buffer; // set qwords to point to the buffer
  set->is_Bit_T_allocated = false;  // not allocated by the library
  return set;
}

/* --- 11b. Properties --- */

size_t BitDB_length(T_DB set) {
  assert(set);
  return set->length;
}

size_t BitDB_nelem(T_DB set) {
  assert(set);
  return set->nelem;
}

uint64_t BitDB_count_at(T_DB set, size_t index) {
  assert(set);
  assert(index < set->nelem);
  uint64_t count = 0;
#if !USE_LIBPOPCNT
  size_t offset;
  if (!bit_size_mul(index, set->size_in_qwords, &offset))
    return 0;
  uint64_t *qwords = set->qwords + offset;
  for (size_t i = 0; i < set->size_in_qwords; i++)
    count += POPCOUNT(qwords[i]);
#else
  size_t offset;
  if (!bit_size_mul(index, set->size_in_bytes, &offset))
    return 0;
  count = popcnt(set->bytes + offset, set->size_in_bytes);
#endif
  return count;
}

uint64_t *BitDB_count(T_DB set) {
  assert(set);
  size_t count_bytes;
  if (!bit_size_mul(set->nelem, sizeof(uint64_t), &count_bytes))
    return NULL;
  uint64_t *counts = malloc(count_bytes);
  if (!counts)
    return NULL;
#if !USE_LIBPOPCNT
  uint64_t *qwords = set->qwords;
  for (size_t i = 0; i < set->nelem; i++, qwords += set->size_in_qwords) {
    uint64_t count = 0;
    for (size_t j = 0; j < set->size_in_qwords; j++)
      count += POPCOUNT(qwords[j]);
    counts[i] = count;
  }
#else
  unsigned char *bytes = set->bytes;
  for (size_t i = 0; i < set->nelem; i++, bytes += set->size_in_bytes)
    counts[i] = popcnt(bytes, set->size_in_bytes);

#endif
  return counts;
}

/* --- 11c. Element access and bulk operations --- */

void BitDB_clear_at(T_DB set, size_t index) {
  assert(set);
  assert(index < set->nelem);
  size_t shift;
  if (!bit_size_mul(index, set->size_in_bytes, &shift))
    return;
  memset(set->bytes + shift, 0, set->size_in_bytes);
}

void BitDB_clear(T_DB set) {
  assert(set);
  size_t size_in_bytes;
  if (!bit_size_mul(set->nelem, set->size_in_bytes, &size_in_bytes))
    return;
  memset(set->bytes, 0, size_in_bytes);
}

T BitDB_get_from(T_DB set, size_t index) {
  assert(set);
  assert(index < set->nelem);
  T bitset = Bit_new(set->length);
  if (!bitset)
    return NULL;
  size_t shift;
  if (!bit_size_mul(index, set->size_in_bytes, &shift)) {
    Bit_free(&bitset);
    return NULL;
  }
  // Copy the bytes from the set to the new bitset
  memcpy(bitset->bytes, set->bytes + shift, set->size_in_bytes);
  return bitset;
}

void BitDB_put_at(T_DB set, size_t index, T bitset) {
  assert(set);
  assert(index < set->nelem);
  assert(bitset);
  assert(bitset->length == set->length);
  // Copy the bytes from the bitset to the set
  size_t shift;
  if (!bit_size_mul(index, set->size_in_bytes, &shift))
    return;
  memcpy(set->bytes + shift, bitset->bytes, set->size_in_bytes);
}

void BitDB_extract_from(T_DB set, size_t index, void *buffer) {
  assert(set);
  assert(index < set->nelem);
  assert(buffer != NULL);
  // Copy the bytes from the set to the buffer
  size_t shift;
  if (!bit_size_mul(index, set->size_in_bytes, &shift))
    return;
  memcpy(buffer, set->bytes + shift, set->size_in_bytes);
}

void BitDB_replace_at(T_DB set, size_t index, void *buffer) {
  assert(set);
  assert(index < set->nelem);
  assert(buffer != NULL);
  // Copy the bytes from the buffer to the set
  size_t shift;
  if (!bit_size_mul(index, set->size_in_bytes, &shift))
    return;
  memcpy(set->bytes + shift, buffer, set->size_in_bytes);
}

/* --- 11d. CPU set operations (allocate and return counts buffer) --- */

uint64_t *BitDB_inter_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
  BitDB_inter_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_inter_count_store_cpu(T_DB bit, T_DB bits, uint64_t *counts,
                                 SETOP_COUNT_OPTS opts) {

  setop_count_db_cpu(bit, bits, counts, _AND, opts);
}

uint64_t *BitDB_union_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
  BitDB_union_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_union_count_store_cpu(T_DB bit, T_DB bits, uint64_t *counts,
                                 SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, _OR, opts);
}

uint64_t *BitDB_diff_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
  BitDB_diff_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_diff_count_store_cpu(T_DB bit, T_DB bits, uint64_t *counts,
                                SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, _XOR, opts);
}

uint64_t *BitDB_minus_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {
  size_t result_count;
  if (!bit_size_mul(bit->nelem, bits->nelem, &result_count) ||
      result_count > SIZE_MAX / sizeof(uint64_t))
    return NULL;
  uint64_t *counts = calloc(result_count, sizeof(*counts));
  if (!counts)
    return NULL;
  BitDB_minus_count_store_cpu(bit, bits, counts, opts);
  return counts;
}

void BitDB_minus_count_store_cpu(T_DB bit, T_DB bits, uint64_t *counts,
                                 SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, _AND_NOT, opts);
}

/* --- End Section 11: PUBLIC API — BITSET DATABASE --- */
