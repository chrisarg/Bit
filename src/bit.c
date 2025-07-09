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
#include "bit.h" // Contains your public API declarations
#include "omp.h"
#include <assert.h>  // For assert() validation
#include <limits.h>  // For INT_MAX
#include <stdbool.h> // For bool type (is_Bit_T_allocated)
#include <stdint.h>  // For uintptr_t and UINT64_C macros
#include <stdio.h>   // For printf (if needed for debugging)
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset

#ifdef USE_LIBPOPCNT
#include "libpopcnt.h"
#define BUFFER_SIZE 1024 /* Buffer size for popcnt operations */
#endif

#define T Bit_T
#define T_DB Bit_T_DB
/*---------------------------------------------------------------------------*/
// Bitset structure
/*
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
  int length;                 // capacity of the bitset in bits
  int size_in_bytes;          // number of bytes of the 8 bit container
  int size_in_qwords;         // number of qwords of the 64 bit container
  bool is_Bit_T_allocated;    // true if allocated by the library
  unsigned char *bytes;       // pointer to the first byte
  unsigned long long *qwords; // pointer to the first qword
};

// Bitset DB structure
/*
 The ADT provides access to containers of bitsets that pack fixed number of
  the bitset data in a single container for locality of memory access when
  processing large number of bitsets.
*/
struct T_DB {
  int nelem;                  // number of bitsets in the packed container
  int length;                 // capacity of the bitset in bits
  int size_in_bytes;          // number of bytes of the 8 bit set container
  int size_in_qwords;         // number of qwords of the 64 bit set container
  bool is_Bit_T_allocated;    // true if allocated by the library
  unsigned char *bytes;       // pointer to the first byte
  unsigned long long *qwords; // pointer to the first qword
};
/*---------------------------------------------------------------------------*/
/*
  Macros
*/

#define STRINGIFY(x) #x // Macro to convert a macro argument to a string

#define BPQW (sizeof(unsigned long long) * 8) // bits per qword
#define BPB (sizeof(unsigned char) * 8)       // bits per byte
#define nqwords(len)                                                           \
  ((((len) + BPQW - 1) & (~(BPQW - 1))) / BPQW)          // ceil(len/QBPW)
#define nbytes(len) ((((len) + 8 - 1) & (~(8 - 1))) / 8) // ceil(len/QBPW)

#define POPCOUNT(x) count_WWG((x))

// Macro that implements the set operations on two bitsets (from Hanson's book)
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
    for (int i = 0; i < s->size_in_qwords; i++) {                              \
      set->qwords[i] = s->qwords[i] op t->qwords[i];                           \
    }                                                                          \
    return set;                                                                \
  }

// Macro that implements the setop count operations on two bitsets
#ifndef USE_LIBPOPCNT
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
    uint64_t count = 0;                                                        \
    for (int i = 0; i < s->size_in_qwords; i++) {                              \
      count += POPCOUNT(s->qwords[i] op t->qwords[i]);                         \
    }                                                                          \
    return (int)count;                                                         \
  }
#else
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
    uint64_t count = 0;                                                        \
    unsigned long long setop_buffer[BUFFER_SIZE]; /*buffer for popcount*/      \
    int limit = s->size_in_qwords - s->size_in_qwords % BUFFER_SIZE;           \
    int i = 0;                                                                 \
    for (; i < limit; i += BUFFER_SIZE) {                                      \
      for (int j = 0; j < BUFFER_SIZE; j++) {                                  \
        setop_buffer[j] = s->qwords[i + j] op t->qwords[i + j];                \
      }                                                                        \
      count += popcnt((void *)setop_buffer,                                    \
                      BUFFER_SIZE * sizeof(unsigned long long));               \
    }                                                                          \
    for (; i < s->size_in_qwords; i++) {                                       \
      count += POPCOUNT(s->qwords[i] op t->qwords[i]);                         \
    }                                                                          \
    return (int)count;                                                         \
  }
#endif

// Checks that 2 DB bitsets are valid and have the same length
#define SETOP_DB_CHECKS(bit, bits)                                             \
  assert(bit &&bits);                                                          \
  assert(bit->length == bits->length);

// Initializes the variables used in the set operations on two DB bitsets
#define SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords, \
                       num_targets, n)                                         \
  unsigned long long *bit_qwords = bit->qwords;                                \
  unsigned long long *bits_qwords = bits->qwords;                              \
  int bit_size_in_qwords = bit->size_in_qwords;                                \
  int num_targets = bit->nelem;                                                \
  int n = bits->nelem;

// Initializes the threaded set operations on two DB bitsets
#define SETOP_DB_INIT_CPU(opts)                                                \
  LIMIT_STATEMENT                                                              \
  int numthreads = opts.num_cpu_threads;                                       \
  if (numthreads <= 0) {                                                       \
    numthreads = omp_get_max_threads();                                        \
  }                                                                            \
  omp_set_num_threads(numthreads);

// conditional macro that executes the set operation count on two DB bitsets
#ifndef USE_LIBPOPCNT
#define LIMIT_STATEMENT
#define setop_count_db_cpu_kernel(bits, bit_qwords, bits_qwords,               \
                                  bit_size_in_qwords, counts, op, n)           \
  int count = 0;                                                               \
  for (int k = 0; k < bit_size_in_qwords; k++) {                               \
    count += POPCOUNT(bit_qwords[i * bit_size_in_qwords + k] op                \
                          bits_qwords[j * bit_size_in_qwords + k]);            \
  }                                                                            \
  counts[i * n + j] = count;
#else
#define LIMIT_STATEMENT                                                        \
  int limit = bit_size_in_qwords - bit_size_in_qwords % BUFFER_SIZE;

#define setop_count_db_cpu_kernel(bits, bit_qwords, bits_qwords,               \
                                  bit_size_in_qwords, counts, op, n)           \
  uint64_t count = 0;                                                          \
  unsigned long long setop_buffer[BUFFER_SIZE];                                \
  int l = 0;                                                                   \
  for (; l < limit; l += BUFFER_SIZE) {                                        \
    for (int k = 0; k < BUFFER_SIZE; k++) {                                    \
      setop_buffer[k] = bit_qwords[i * bit_size_in_qwords + l + k] op          \
          bits_qwords[j * bit_size_in_qwords + l + k];                         \
    }                                                                          \
    count += popcnt((void *)setop_buffer,                                      \
                    BUFFER_SIZE * sizeof(unsigned long long));                 \
  }                                                                            \
  for (; l < bit_size_in_qwords; l++) {                                        \
    count += POPCOUNT(bit_qwords[i * bit_size_in_qwords + l] op                \
                          bits_qwords[j * bit_size_in_qwords + l]);            \
  }                                                                            \
  counts[i * n + j] = (int)count;
#endif

// simple OMP macro to parallelize the loop
#define OMP_CPU_LOOP(levels, sched)                                            \
_Pragma(STRINGIFY(omp parallel for collapse(levels) schedule(sched)))

// CPU SETOP on two bitsets
#define setop_count_db_cpu(bit, bits, counts, op, opts)                        \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
  SETOP_DB_INIT_CPU(opts)                                                      \
  OMP_CPU_LOOP(2, guided)                                                      \
  for (int i = 0; i < num_targets; i++) {                                      \
    for (int j = 0; j < n; j++) {                                              \
      setop_count_db_cpu_kernel(bits, bit_qwords, bits_qwords,                 \
                                bit_size_in_qwords, counts, op, n)             \
    }                                                                          \
  }                                                                            \
  return counts;

// Macros for GPU operations

#define UPDATE_GPU_ARRAY(dir, array, index1, index2, dev_id)                   \
  _Pragma(                                                                     \
      STRINGIFY(omp target update dir(array [index1:index2]) device(dev_id)))

#define TARGET_GPU_ARRAY(point, dir, array, index1, index2, dev_id)            \
  _Pragma(STRINGIFY(omp target point data map(dir                              \
                                              : array [index1:index2])         \
                        device(dev_id)))

#define SETOP_INIT_GPU(bit, bits, counts, opts)                                \
  if (omp_target_is_present(bit->qwords, opts.device_id)) {                    \
    if (opts.upd_1st_operand) {                                                \
      UPDATE_GPU_ARRAY(to, bit->qwords, 0, bit->size_in_qwords * bit->nelem,   \
                       opts.device_id)                                         \
    }                                                                          \
  } else {                                                                     \
    TARGET_GPU_ARRAY(enter, tofrom, bit->qwords, 0,                            \
                     bit->size_in_qwords * bit->nelem, opts.device_id)         \
  }                                                                            \
  if (omp_target_is_present(bits->qwords, opts.device_id)) {                   \
    if (opts.upd_2nd_operand) {                                                \
      UPDATE_GPU_ARRAY(to, bits->qwords, 0,                                    \
                       bits->size_in_qwords * bits->nelem, opts.device_id)     \
    } else {                                                                   \
      TARGET_GPU_ARRAY(enter, tofrom, bits->qwords, 0,                         \
                       bits->size_in_qwords * bits->nelem, opts.device_id)     \
    }                                                                          \
  } else {                                                                     \
    TARGET_GPU_ARRAY(enter, tofrom, bits->qwords, 0,                           \
                     bits->size_in_qwords * bits->nelem, opts.device_id)       \
  }                                                                            \
  if (!omp_target_is_present(counts, opts.device_id)) {                        \
    TARGET_GPU_ARRAY(enter, to, counts, 0, bit->nelem * bits->nelem,           \
                     opts.device_id)                                           \
  }

#define OMP_GPU_TEAMS(num_targets, dev_id)                                     \
  _Pragma(STRINGIFY(omp target teams distribute num_teams(num_targets)         \
                        device(dev_id)))

#define OMP_GPU_PARALLEL(n) _Pragma(STRINGIFY(omp parallel num_threads(n)))

#define OMP_GPU_SIMD_REDUCTION(reduction_type, reduction_var)                  \
  _Pragma(STRINGIFY(omp simd reduction(reduction_type : reduction_var)))
  
#define OMP_GPU_FOR_NOWAIT _Pragma(STRINGIFY(omp for nowait))

#define SETOP_FINALIZE_GPU(action, buffer, index1, index2, dev_id)             \
  if (omp_target_is_present(buffer, dev_id)) {                                 \
    _Pragma(STRINGIFY(omp target exit data                                     \
      map(action : buffer [index1:index2]) device(dev_id)))                    \
  }


#define setop_count_db_gpu(bit, bits, counts, op, opts)                        \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                  num_targets, n)                                              \
  SETOP_INIT_GPU(bit, bits, counts, opts)                                      \
  OMP_GPU_TEAMS(num_targets, opts.device_id)                                   \
  for (int k = 0; k < num_targets; k++) {                                      \
    uint64_t shift_k = k * bit_size_in_qwords;                                 \
    OMP_GPU_PARALLEL(n)                                                        \
    {                                                                          \
      OMP_GPU_FOR_NOWAIT                                                       \
      for (unsigned int i = 0; i < n; i++) {                                   \
        uint64_t shift_i = i * bit_size_in_qwords;                             \
        int total_sum_for_i = 0;                                               \
        OMP_GPU_SIMD_REDUCTION(+, total_sum_for_i)                             \
        for (unsigned int j = 0; j < bit_size_in_qwords; j++) {                \
          unsigned long long x =                                               \
              bit_qwords[shift_k + j] op bits_qwords[shift_i + j];             \
          total_sum_for_i += (uint32_t)count_WWG(x);                           \
        }                                                                      \
        counts[k * n + i] = total_sum_for_i;                                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  _Pragma(STRINGIFY(omp target exit data map(from : counts[0:num_targets*n]))) \
  if(opts.release_1st_operand) {                                               \
    SETOP_FINALIZE_GPU(release, bit->qwords, 0,                                \
      bit_size_in_qwords * num_targets, opts.device_id)                        \
  }                                                                            \
  if(opts.release_2nd_operand) {                                               \
    SETOP_FINALIZE_GPU(release, bits->qwords, 0, bit_size_in_qwords * n,       \
                       opts.device_id)                                         \
  }                                                                            \
  if (opts.release_counts) {                                                   \
    SETOP_FINALIZE_GPU(release, counts, 0, num_targets * n, opts.device_id)    \
  }                                                                            \
  return counts;
/*---------------------------------------------------------------------------*/
/*
  Static Data
*/

// masks used for some of the bit functions in the public API
unsigned const char msbmask[] = {
    0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80,
};

unsigned const char lsbmask[] = {0x01, 0x03, 0x07, 0x0F,
                                 0x1F, 0x3F, 0x7F, 0xFF};
/*---------------------------------------------------------------------------*/
/*
  Static Functions
 */

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
#define C1_WWG UINT64_C(0X5555555555555555)
#define C2_WWG UINT64_C(0x3333333333333333)
#define C3_WWG UINT64_C(0x0F0F0F0F0F0F0F0F)
#define C4_WWG UINT64_C(0x0101010101010101)
static inline uint64_t count_WWG(unsigned long long x) {
  x -= (x >> 1) & C1_WWG;
  x = ((x >> 2) & C2_WWG) + (x & C2_WWG);
  x = (x + (x >> 4)) & C3_WWG;
  x *= C4_WWG;

  return (x >> 56);
}

// create it at the device as well
#pragma omp declare target(count_WWG)
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

  set->qwords = calloc(set->size_in_bytes, sizeof(unsigned char));
  assert(set->qwords != NULL);

  set->bytes = (unsigned char *)set->qwords;

  set->is_Bit_T_allocated = true; // allocated by the library
  return set;
}

void *Bit_free(T *set) {

  assert(set && *set);
  void *original_location = NULL;
  if ((*set)->is_Bit_T_allocated) {
    original_location = (void *)(*set)->qwords;
    free((*set)->qwords);
    (*set)->qwords = NULL;
    (*set)->bytes = NULL; // set bytes to NULL after freeing qwords
  }
  free(*set);
  *set = NULL;
  return original_location; // return the original location of the buffer or
  // NULL
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
  set->qwords =
      (unsigned long long *)buffer; // set qwords to point to the buffer
  set->is_Bit_T_allocated = false;  // not allocated by the library
  return set;
}

extern int Bit_extract(T set, void *buffer) {
  assert(set);
  assert(buffer != NULL);
  // Copy the bytes from the bitset to the buffer
  memcpy(buffer, set->bytes, set->size_in_bytes);
  return set->size_in_bytes; // return the number of bytes written
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

int Bit_buffer_size(int length) {
  assert(length > 0);
  return nqwords(length) * BPQW / BPB;
}

/*---------------------------------------------------------------------------*/
// Functions that manipulate an individual bitset (member operations):
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
/*---------------------------------------------------------------------------*/
// Functions that operate on two bitsets (and create a new one)

T Bit_diff(T s, T t) { setop(Bit_new(s->length), copy(t), copy(s), ^); }
T Bit_minus(T s, T t) {
  setop(Bit_new(s->length), Bit_new(t->length), copy(s), &~);
}
T Bit_inter(T s, T t){setop(copy(t), Bit_new(t->length),
                            Bit_new(s->length), &)} T Bit_union(T s, T t) {
  setop(copy(t), copy(t), copy(s), |)
}

/*---------------------------------------------------------------------------*/
// Functions that operate on two bitsets (and return the population count of
// the result)

int Bit_diff_count(T s, T t) { setop_count(0, Bit_count(t), Bit_count(s), ^); }
int Bit_minus_count(T s, T t) { setop_count(0, 0, Bit_count(s), &~); }
int Bit_inter_count(T s, T t) { setop_count(Bit_count(t), 0, 0, &); }
int Bit_union_count(T s, T t) {
  setop_count(Bit_count(t), Bit_count(t), Bit_count(s), |);
}

/*---------------------------------------------------------------------------*/

// Functions that create, free and obtain the properties of the Bit_DB.
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

  set->qwords = calloc(size_in_bytes, sizeof(unsigned char));
  assert(set->qwords != NULL);

  set->bytes = (unsigned char *)set->qwords;
  set->is_Bit_T_allocated = true; // allocated by the library
  return set;
}
void *BitDB_free(T_DB *set) {
  assert(set && *set);
  void *original_location = NULL;
  if ((*set)->is_Bit_T_allocated) {
    original_location = (void *)(*set)->qwords;
    free((*set)->qwords);
    (*set)->qwords = NULL;
    (*set)->bytes = NULL; // set bytes to NULL after freeing qwords
  }
  free(*set);
  *set = NULL;
  // return the original location of the buffer or NULL
  return original_location;
}

extern int BitDB_length(T_DB set) {
  assert(set);
  return set->length;
}

extern int BitDB_nelem(T_DB set) {
  assert(set);
  return set->nelem;
}

extern int BitDB_count_at(T_DB set, int index) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  int count = 0;
#ifndef USE_LIBPOPCNT
  unsigned long long *qwords = set->qwords + index * set->size_in_qwords;
  for (int i = 0; i < set->size_in_qwords; i++)
    count += POPCOUNT(qwords[i]);
#else
  count =
      (int)popcnt(set->bytes + index * set->size_in_bytes, set->size_in_bytes);
#endif
  return count;
}

extern int *BitDB_count(T_DB set) {
  assert(set);
  int *counts = malloc(set->nelem * sizeof(int));
  assert(counts != NULL);
#ifndef USE_LIBPOPCNT
  unsigned long long *qwords = set->qwords;
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

extern void BitDB_clear_at(T_DB set, int index) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memset(set->bytes + shift, 0, set->size_in_bytes);
}

extern void BitDB_clear(T_DB set) {
  assert(set);
  size_t size_in_bytes = (size_t)set->nelem;
  size_in_bytes *= set->size_in_bytes; // calculate the total size
  memset(set->bytes, 0, size_in_bytes);
}

extern T BitDB_get_from(T_DB set, int index) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  T bitset = Bit_new(set->length);
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  // Copy the bytes from the set to the new bitset
  memcpy(bitset->bytes, set->bytes + shift, set->size_in_bytes);
  return bitset;
}

extern void BitDB_put_at(T_DB set, int index, T bitset) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  assert(bitset);
  assert(bitset->length == set->length);
  // Copy the bytes from the bitset to the set
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memcpy(set->bytes + shift, bitset->bytes, set->size_in_bytes);
}

extern int BitDB_extract_from(T_DB set, int index, void *buffer) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  assert(buffer != NULL);
  // Copy the bytes from the set to the buffer
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memcpy(buffer, set->bytes + shift, set->size_in_bytes);
  return set->size_in_bytes; // return the length of the bitset
}

extern void BitDB_replace_at(T_DB set, int index, void *buffer) {
  assert(set);
  assert(index >= 0 && index < set->nelem);
  assert(buffer != NULL);
  // Copy the bytes from the buffer to the set
  size_t shift = (size_t)index;
  shift *= set->size_in_bytes; // calculate the offset
  memcpy(set->bytes + shift, buffer, set->size_in_bytes);
}

/*---------------------------------------------------------------------------*/
// Functions that operate on two BitDB containers (and return the population
// count of the result)

// Macros used to reduce un-necessary code duplication

extern int *BitDB_inter_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_cpu(bit, bits, counts, opts);
}

extern int *BitDB_inter_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                        SETOP_COUNT_OPTS opts) {

  setop_count_db_cpu(bit, bits, counts, &, opts)
}

extern int *BitDB_inter_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_gpu(bit, bits, counts, opts);
}
extern int *BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                        SETOP_COUNT_OPTS opts) {
  setop_count_db_gpu(bit, bits, counts, &, opts)
}

extern int *BitDB_union_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_cpu(bit, bits, counts, opts);
}

extern int *BitDB_union_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                        SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, |, opts);
}

extern int *BitDB_union_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_gpu(bit, bits, counts, opts);
}

extern int *BitDB_union_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                        SETOP_COUNT_OPTS opts) {
  setop_count_db_gpu(bit, bits, counts, |, opts)
}

extern int *BitDB_diff_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_cpu(bit, bits, counts, opts);
}

extern int *BitDB_diff_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                       SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, ^, opts)
}

extern int *BitDB_diff_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_gpu(bit, bits, counts, opts);
}

extern int *BitDB_diff_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                       SETOP_COUNT_OPTS opts) {
  setop_count_db_gpu(bit, bits, counts, ^, opts)
}

extern int *BitDB_minus_count_cpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_cpu(bit, bits, counts, opts);
}

extern int *BitDB_minus_count_store_cpu(T_DB bit, T_DB bits, int *counts,
                                        SETOP_COUNT_OPTS opts) {
  setop_count_db_cpu(bit, bits, counts, &~, opts)
}

extern int *BitDB_minus_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int *counts = (int *)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_gpu(bit, bits, counts, opts);
}

extern int *BitDB_minus_count_store_gpu(T_DB bit, T_DB bits, int *counts,
                                        SETOP_COUNT_OPTS opts) {
  setop_count_db_gpu(bit, bits, counts, &~, opts)
}

void Bit_debug(T set)
{
  assert(set);
  uintptr_t p1 = (uintptr_t)set;
 // printf("set: %xllu, *set2:  %xllu, set2: %xllu\n", p1, p2, p3);
  printf("set: %llx\n", p1);
  printf("Count : %d\n", Bit_count(set));
}
/*---------------------------------------------------------------------------*/

/*
  Code that was eventually converted to Preprocessor macros with arguments
  or used in earlier versions of the library
*/

/*
  static inline void SETOP_INIT_GPU(T_DB bit, T_DB bits, int *counts,
                                   SETOP_COUNT_OPTS opts) {
  if (omp_target_is_present(bit->qwords, opts.device_id)) {
    if (opts.upd_1st_operand) {
#pragma omp target update to(                                                  \
        bit->qwords[0 : bit->size_in_qwords * bit->nelem])                     \
    device(opts.device_id)
    }
  } else {
#pragma omp target enter data map(                                             \
        tofrom : bit->qwords[0 : bit->size_in_qwords * bit->nelem])            \
    device(opts.device_id)
  }

  if (omp_target_is_present(bits->qwords, opts.device_id)) {
    if (opts.upd_2nd_operand) {
#pragma omp target update to(                                                  \
        bits->qwords[0 : bits->size_in_qwords * bits->nelem])                  \
    device(opts.device_id)
    }
  } else {
#pragma omp target enter data map(                                             \
        tofrom : bits->qwords[0 : bits->size_in_qwords * bits->nelem])         \
    device(opts.device_id)
  }

  if (!omp_target_is_present(counts, opts.device_id)) {
#pragma omp target enter data map(to : counts[0 : bit->nelem * bits->nelem])   \
    device(opts.device_id)
  }
}

extern int* BitDB_inter_count_gpu(T_DB bit, T_DB bits, SETOP_COUNT_OPTS opts) {

  int* counts = (int*)calloc(bit->nelem * bits->nelem, sizeof(int));
  assert(counts != NULL);
  return BitDB_inter_count_store_gpu(bit, bits, counts, opts);
}
extern int* BitDB_inter_count_store_gpu(T_DB bit, T_DB bits, int* counts,
  SETOP_COUNT_OPTS opts) {

    SETOP_DB_CHECKS(bit, bits)
    SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,
                    num_targets, n)
    SETOP_INIT_GPU(bit, bits, counts, opts)

  #pragma omp target teams distribute num_teams(num_targets) \
      device(opts.device_id)
    for (int k = 0; k < num_targets; k++) {
      uint64_t shift_k = k * bit_size_in_qwords;
  #pragma omp parallel num_threads(n)
      {
  #pragma omp for nowait
        for (unsigned int i = 0; i < n; i++) {
          uint64_t shift_i = i * bit_size_in_qwords;
          uint32_t total_sum_for_i = 0;
  #pragma omp simd reduction(+ : total_sum_for_i)
          for (unsigned int j = 0; j < bit_size_in_qwords; j++) {
            unsigned long long x =
                bit_qwords[shift_k + j] & bits_qwords[shift_i + j];
            total_sum_for_i += (uint32_t)count_WWG(x);
          }
          counts[k * n + i] = total_sum_for_i;
        }
      }
    }
  #pragma omp target update from(counts[0 : n * num_targets])

    return counts;

}

*/
/*
  Note this is a generic aligned calloc function.
  It was used in the very first version of the library
  to allocate aligned memory for the bitsets.
  It is not used in the current version, but kept for reference.

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

*/

/*
Nicely formated macros (for when vscode formatting misbehaves)

#define setop_count_db_gpu(bit, bits, counts, op, opts)                        \
  SETOP_DB_CHECKS(bit, bits)                                                  \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,      \
                  num_targets, n)                                              \
  SETOP_INIT_GPU(bit, bits, counts, opts)                                     \
  OMP_GPU_TEAMS(num_targets, opts.device_id)                                   \
  for (int k = 0; k < num_targets; k++) {                                      \
    uint64_t shift_k = k * bit_size_in_qwords;                                 \
    OMP_GPU_PARALLEL(n)                                                        \
    {                                                                          \
      OMP_GPU_FOR_NOWAIT                                                       \
      for (unsigned int i = 0; i < n; i++) {                                   \
        uint64_t shift_i = i * bit_size_in_qwords;                             \
        int total_sum_for_i = 0;                                               \
        OMP_GPU_SIMD_REDUCTION(+, total_sum_for_i)                             \
        for (unsigned int j = 0; j < bit_size_in_qwords; j++) {                \
          unsigned long long x =                                               \
              bit_qwords[shift_k + j] op bits_qwords[shift_i + j];             \
          total_sum_for_i += (uint32_t)count_WWG(x);                           \
        }                                                                      \
        counts[k * n + i] = total_sum_for_i;                                   \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  _Pragma(STRINGIFY(omp target exit data                                       \
    map(from : counts[0:num_targets])))                                        \
  return counts;
*/