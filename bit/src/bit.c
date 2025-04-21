/*
    A simple uncompressed bitset implementation based on the ADT
    in David Hanson's "C Interfaces and Implementations" book.
    Suitably extended to allow memory for the bitsets to be allocated
    outside the library.


    * Author : Christos Argyropoulos
    * Created : April 1st 2025
    * Copyright : (c) 2025
    * License : Free
*/
#include "bit.h"     // Contains your public API declarations
#include <assert.h>  // For assert() validation
#include <limits.h>  // For INT_MAX
#include <stdbool.h> // For bool type (is_Bit_T_allocated)
#include <stdint.h>  // For uintptr_t and UINT64_C macros
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset

#include <stdio.h> // For printf (if needed for debugging)

#define T Bit_T


// Detect architecture and set appropriate alignment
#if defined(__LP64__) || defined(__x86_64__) || defined(__amd64__) ||          \
    defined(__aarch64__) || defined(_WIN64) || defined(__ia64__) ||            \
    defined(__powerpc64__)
// 64-bit architecture
#define ALIGNMENT 64
#else
// 32-bit architecture
#define ALIGNMENT 32
#endif



#ifdef BUILTIN_POPCOUNT
#define POPCOUNT(x) __builtin_popcountll(x)
#else
#define POPCOUNT(x) count_WWG((x))
#endif


static_assert(ALIGNMENT >= 2 * sizeof(void *),
              "Alignment must be greater or equal to 2 x* sizeof(void *)");

// Declarations of helper functions that are not part of the public API
static T copy(T t);
static inline uint64_t count_WWG(unsigned long long input_num);
static void *portable_aligned_calloc(size_t alignment, size_t size);

#if (ALIGNMENT % 32 || ALIGNMENT % 64)
  #define MM_LOAD_256(x) _mm256_loadu_si256((__m256i *)(x))
#else
  #define MM_LOAD_256(x) _mm256_load_si256((__m256i *)(x))
#endif

#if ALIGNMENT % 64
  #define MM_LOAD_512(x) _mm512_loadu_si512((__m512i *)(x))
#else
  #define MM_LOAD_512(x) _mm512_load_si512((__m512i *)(x))
#endif

#ifdef USE_LIBPOPCNT
#include "libpopcnt.h"
//#include "popcnt-sse-harley-seal.h"
//#define popcnt(x, size)   popcnt_SSE_harley_seal((x), (size))
#endif
// masks used for some of the bit functions in the public API
unsigned const char msbmask[] = {
    0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80,
};

unsigned const char lsbmask[] = {0x01, 0x03, 0x07, 0x0F,
                                 0x1F, 0x3F, 0x7F, 0xFF};

// Bitset structure
/*
 The ADT provides access to the bitset as bytes,or qwords,
 anticipating optimization of the code *down the road*, 
 including loading of externally allocated buffers into it
 The interface is effectively the one for Bit_T presented by D. Hanson
 in C Interfaces and Implementations, ISBN 0-201-49841-3 Ch13, 1997.
 Changes made by the author include the use of aligned allocations
 for the buffer and the introduction of the set_count operations
 to avoid forming the intermediate Bit_T
*/
struct T {
  int length;
  size_t size_in_bytes;       // number of 8 bit container
  size_t size_in_qwords;      // number of 64 bit container
  bool is_Bit_T_allocated;    // true if allocated by the library
  unsigned char *bytes;       // pointer to the first byte
  unsigned long long *qwords; // pointer to the first qword
};
#define BPQW (sizeof(unsigned long long) * 8) // bits per qword
#define BPB (sizeof(unsigned char) * 8)       // bits per byte
#define nqwords(len)                                                           \
  ((((len) + BPQW - 1) & (~(BPQW - 1))) / BPQW)          // ceil(len/QBPW)
#define nbytes(len) ((((len) + 8 - 1) & (~(8 - 1))) / 8) // ceil(len/QBPW)

/*---------------------------------------------------------------------------*/
// Functions that create, free and obtain the properties of the bitset.
T Bit_new(int length) {
  assert(length > 0);
  assert(length < INT_MAX); // limit to 2^30 bits
  // alignment must be a multiple of sizeof(unsigned long long) or zero
  T set = malloc(sizeof(*set));
  set->length = length;

  set->size_in_qwords = nqwords(length);
  set->size_in_bytes = set->size_in_qwords * BPQW / BPB;

  set->qwords = portable_aligned_calloc(ALIGNMENT, set->size_in_bytes);
  assert(set->qwords != NULL);

  set->bytes = (unsigned char *)set->qwords;

  set->is_Bit_T_allocated = true; // allocated by the library
  return set;
}

void Bit_free(T set) {
  assert(set);
  assert(set->is_Bit_T_allocated);
  void *original_location = (void *)set->qwords - sizeof(void *);
  void *original_block = *(void **)original_location;
  free(original_block);
  free(set);
}

void Bit_free_safe(T *set) {

  assert(set && *set);
  assert((*set)->is_Bit_T_allocated);
  void *original_location = (void *)(*set)->qwords - sizeof(void *);
  void *original_block = *(void **)original_location;
  free(original_block);
  *((void **)original_location) = NULL; // clear the pointer
  free(*set);
  *set = NULL;
}
/*---------------------------------------------------------------------------*/
//     Functions that obtain the properties of a bitset:
int Bit_length(T set) {
  assert(set);
  return set->length;
}

int Bit_count(T set) {
  assert(set);
  int length = 0;
  #ifndef USE_LIBPOPCNT
  for (size_t i = 0; i < nqwords(set->length); i++) {
    length += POPCOUNT(set->qwords[i]);
  }
  #else
  length = (int)popcnt(set->bytes, set->size_in_bytes);
  #endif
  return length;
}

int size_of_Bit_T(void) { return sizeof(struct T); }

int Bit_size(int length) {
  assert(length > 0);
  return sizeof(struct T) + nqwords(length) * BPQW / BPB + ALIGNMENT - 1 +
         sizeof(void *);
}

int Bit_buffer_size(int length) {
  assert(length > 0);
  return nqwords(length) * BPQW / BPB + ALIGNMENT - 1 + sizeof(void *);
}

/*---------------------------------------------------------------------------*/
// Functions that manipulate an individual bitset (member operations):
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
    set->bytes[lo / 8] &= ~ msbmask[lo % 8];
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
/*---------------------------------------------------------------------------*/
// Functions that compare two bitsets

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
    if ((s->qwords[i] &~t->qwords[i]) != 0)
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
/*---------------------------------------------------------------------------*/
// Functions that operate on two bitsets (and create a new one)

#define setop(sequal, snull, tnull, op)                                        \
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
    T set = Bit_new(s->length);                                                \
    for (int i = s->size_in_qwords; --i >= 0;) {                                \
      set->qwords[i] = s->qwords[i] op t->qwords[i];                           \
    }                                                                          \
    return set;                                                                \
  }

T Bit_diff(T s, T t) { setop(Bit_new(s->length), copy(t), copy(s), ^); }
T Bit_minus(T s, T t) {
  setop(Bit_new(s->length), Bit_new(t->length), copy(s), &~);
}
T Bit_inter(T s, T t) {
  setop(copy(t), Bit_new(t->length), Bit_new(s->length), &);
}
T Bit_union(T s, T t) { setop(copy(t), copy(t), copy(s), |); }

/*---------------------------------------------------------------------------*/
// Functions that operate on two bitsets (and return the population count of
// the result)

#define setop_count(sequal, snull, tnull, op)                                  \
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
    int answer = 0;                                                            \
    for (int i = s->size_in_qwords; --i >= 0;) {                                \
      answer += POPCOUNT(s->qwords[i] op t->qwords[i]);                        \
    }                                                                          \
    return answer;                                                             \
  }

int Bit_diff_count(T s, T t) { setop_count(0, Bit_count(t), Bit_count(s), ^); }
int Bit_minus_count(T s, T t) { setop_count(0, 0, Bit_count(s), &~); }
int Bit_inter_count(T s, T t) { setop_count(Bit_count(t), 0, 0, &); }
int Bit_union_count(T s, T t) {
  setop_count(Bit_count(t), Bit_count(t), Bit_count(s), |);
}

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
// Static functions that are not part of the public interface
/*
  Note this is a generic aligned calloc function.
*/
void *portable_aligned_calloc(size_t alignment, size_t size) {

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
  void **store_ptr = (void **)(aligned_ptr - sizeof(void *));
  *store_ptr = original; // Store the original pointer

  ptr = aligned_ptr;

  // Zero the allocated memory if allocation was successful
  if (ptr) {
    memset(ptr, 0, size);
  }

  return ptr;
}

static T copy(T t) {
  T set;
  assert(t);
  set = Bit_new(t->length);
  if (t->length > 0) {
    memcpy(set->bytes, t->bytes, t->size_in_bytes);
  }
  return set;
}

/*
  Wilks Wheeler Gill function : highly portable and fast
  https://en.wikipedia.org/wiki/The_Preparation_of_Programs_for_an_Electronic_Digital_Computer
  https://arxiv.org/abs/1611.07612
  https://github.com/kimwalisch/libpopcnt/tree/master
  Note that WWG implementation of bitset intersection is rather portable,
  but can be optimized further.
*/

static inline uint64_t count_WWG(unsigned long long x) {
#define C1_WWG UINT64_C(0X5555555555555555)
#define C2_WWG UINT64_C(0x3333333333333333)
#define C3_WWG UINT64_C(0x0F0F0F0F0F0F0F0F)
#define C4_WWG UINT64_C(0x0101010101010101)

  x -= (x >> 1) & C1_WWG;
  x = ((x >> 2) & C2_WWG) + (x & C2_WWG);
  x = (x + (x >> 4)) & C3_WWG;
  x *= C4_WWG;

  return (x >> 56);
}
