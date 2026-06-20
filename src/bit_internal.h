/*
    Private implementation macros for the Bit library.
    NOT part of the public API — include only from src/bit.c.

    Requires that the following macros are already defined before this
    header is included (all defined in Section 3 of bit.c):
      STRINGIFY, ALIGNMENT, SETOP_BUFFER_SIZE, POPCOUNT,
      OMP_CPU_SIMD_ALIGN_BUFFER, ALIGN_CHECK, ARCH_32BIT,
      CPU_TILE_BIT, CPU_TILE_BITS

    Note: POPCOUNT_GPU is defined later in Section 8 of bit.c (after
    count_WWG is declared and compiled for GPU target), so macros in
    Section D that reference it are fine — they expand lazily at the
    call sites in Section 11.

    * Author : Christos Argyropoulos
    * Created : April 1st 2025 (refactored June 2026)
    * Copyright : (c) 2025
    * License : BSD-2
*/
#pragma once

/* ===========================================================================
   SECTION A: OPENMP CPU PARALLELIZATION HELPERS
   =========================================================================== */

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

/* --- End Section A: OPENMP CPU PARALLELIZATION HELPERS --- */

/* ===========================================================================
   SECTION B: SINGLE-BITSET SET OPERATION MACROS (Bit_T)
   =========================================================================== */

/* Set operation that creates a new Bit_T result (from Hanson's book) */
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

/* Set operation that returns the population count of the result */
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
    uint64_t setop_buffer[SETOP_BUFFER_SIZE]; /*buffer for popcount*/          \
    int limit = s->size_in_qwords - s->size_in_qwords % SETOP_BUFFER_SIZE;     \
    int i = 0;                                                                 \
    for (; i < limit; i += SETOP_BUFFER_SIZE) {                                \
      for (int j = 0; j < SETOP_BUFFER_SIZE; j++) {                            \
        setop_buffer[j] = s->qwords[i + j] op t->qwords[i + j];                \
      }                                                                        \
      count +=                                                                 \
          popcnt((void *)setop_buffer, SETOP_BUFFER_SIZE * sizeof(uint64_t));  \
    }                                                                          \
    for (; i < s->size_in_qwords; i++) {                                       \
      count += POPCOUNT(s->qwords[i] op t->qwords[i]);                         \
    }                                                                          \
    return (int)count;                                                         \
  }
#endif

/* --- End Section B: SINGLE-BITSET SET OPERATION MACROS --- */

/* ===========================================================================
   SECTION C: DB SET OPERATION MACROS — CPU
   =========================================================================== */

/* Validate two Bit_DB_T operands have the same bitset length */
#define SETOP_DB_CHECKS(bit, bits)                                             \
  assert(bit && bits);                                                         \
  assert(bit->length == bits->length);

/* Extract raw pointers and dimensions from two Bit_DB_T operands */
#define SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords, \
                       num_targets, n)                                         \
  uint64_t *bit_qwords = bit->qwords;                                          \
  uint64_t *bits_qwords = bits->qwords;                                        \
  int bit_size_in_qwords = bit->size_in_qwords;                                \
  int num_targets = bit->nelem;                                                \
  int n = bits->nelem;

/* Compute the popcount remainder limit and configure the OMP thread count */
#define SETOP_DB_INIT_CPU(opts)                                                \
  LIMIT_STATEMENT(limit, bit_size_in_qwords, SETOP_BUFFER_SIZE)                \
  int numthreads = opts.num_cpu_threads;                                       \
  if (numthreads <= 0) {                                                       \
    numthreads = omp_get_max_threads();                                        \
  }                                                                            \
  omp_set_num_threads(numthreads);

/* Accumulate popcount over a tile buffer (libpopcnt vs WWG dispatch) */
#ifndef USE_LIBPOPCNT

#define POPULATION_COUNT(count, setop_buffer, buffer_size)                     \
  /* SIMD SYNCHRONIZATION DIRECTIVE */                                         \
  OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, setop_buffer)                           \
  for (int k = 0; k < SETOP_BUFFER_SIZE; k++) {                                \
    count += POPCOUNT(setop_buffer[k]);                                        \
  }

#else

#define POPULATION_COUNT(count, setop_buffer, buffer_size)                     \
  count += popcnt((void *)setop_buffer, buffer_size * sizeof(uint64_t));

#endif

/* Compute the last aligned index before the scalar remainder */
#define LIMIT_STATEMENT(limit, bit_size_in_qwords, SETOP_BUFFER_SIZE)          \
  int limit = bit_size_in_qwords - bit_size_in_qwords % SETOP_BUFFER_SIZE;

/* Open the tiled double loop over (i_b, j_b) tile origins */
#define OMP_CPU_TILE_START                                                     \
  OMP_CPU_LOOP(2, static)                                                      \
  for (int i_b = 0; i_b < num_targets; i_b += CPU_TILE_BIT) {                  \
    for (int j_b = 0; j_b < n; j_b += CPU_TILE_BITS) {                         \
                                                                               \
      /* Determine boundary limits for this tile */                            \
      int i_max = (i_b + CPU_TILE_BIT < num_targets) ? i_b + CPU_TILE_BIT      \
                                                     : num_targets;            \
      int j_max = (j_b + CPU_TILE_BITS < n) ? j_b + CPU_TILE_BITS : n;         \
                                                                               \
      /* Compute all pairs within the tile */                                  \
      for (int i = i_b; i < i_max; i++) {                                      \
        /* Pre-calculate the starting pointer for row i of Matrix A */         \
        const uint64_t *restrict a_row =                                       \
            bit_qwords + (uint64_t)i * bit_size_in_qwords;                     \
                                                                               \
        for (int j = j_b; j < j_max; j++) {                                    \
          /* Pre-calculate the starting pointer for row j of Matrix B */       \
          const uint64_t *restrict b_row =                                     \
              bits_qwords + (uint64_t)j * bit_size_in_qwords;                  \
/* Here goes code that differentiates the execution paths */

/* Close the tiled double loop — matches the four { in OMP_CPU_TILE_START */
#define OMP_CPU_TILE_END                                                       \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/* Top-level DB CPU set-operation dispatch (architecture and alignment aware) */
#define setop_count_db_cpu(bit, bits, counts, op, opts)                        \
  SETOP_DB_CHECKS(bit, bits)                                                   \
  SETOP_VAR_INIT(bit, bits, bit_qwords, bits_qwords, bit_size_in_qwords,       \
                 num_targets, n)                                               \
                                                                               \
  /*Check Alignment of the buffers to select the codepath*/                    \
  bool aligned = ALIGN_CHECK(bit_qwords) && ALIGN_CHECK(bits_qwords);          \
  SETOP_DB_INIT_CPU(opts)                                                      \
  if (ARCH_32BIT) {                                                            \
    OMP_CPU_TILE_START                                                         \
    setop_count_db_cpu_kernel(                                                 \
        a_row, b_row, bit_size_in_qwords, counts[(uint64_t)i * n + j], op,     \
        OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, a_row, b_row, setop_buffer))      \
    OMP_CPU_TILE_END                                                           \
  } else {                                                                     \
    if (aligned) {                                                             \
      OMP_CPU_TILE_START                                                       \
      setop_count_db_cpu_kernel(                                               \
          a_row, b_row, bit_size_in_qwords, counts[(uint64_t)i * n + j], op,   \
          OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, a_row, b_row, setop_buffer))    \
      OMP_CPU_TILE_END                                                         \
    } else {                                                                   \
      OMP_CPU_TILE_START                                                       \
      setop_count_db_cpu_kernel(                                               \
          a_row, b_row, bit_size_in_qwords, counts[(uint64_t)i * n + j], op,   \
          OMP_CPU_SIMD_ALIGN_BUFFER(ALIGNMENT, setop_buffer))                  \
      OMP_CPU_TILE_END                                                         \
    }                                                                          \
  }

/* Inner kernel: buffered set-op + popcount for one (a_row, b_row) pair */
#define setop_count_db_cpu_kernel(a_row, b_row, bit_size_in_qwords, result,    \
                                  op, SIMD_DIRECTIVE)                          \
  _Alignas(ALIGNMENT) uint64_t setop_buffer[SETOP_BUFFER_SIZE];                \
  uint64_t count = 0;                                                          \
  int l = 0;                                                                   \
                                                                               \
  for (; l < limit; l += SETOP_BUFFER_SIZE) {                                  \
    /* SIMD SYNCHRONIZATION DIRECTIVE */                                       \
    SIMD_DIRECTIVE                                                             \
    for (int k = 0; k < SETOP_BUFFER_SIZE; k++) {                              \
      setop_buffer[k] = a_row[l + k] op b_row[l + k];                          \
    }                                                                          \
    POPULATION_COUNT(count, setop_buffer, SETOP_BUFFER_SIZE)                   \
  } /* Handle the scalar remainder */                                          \
  for (; l < bit_size_in_qwords; l++) {                                        \
    count += POPCOUNT(a_row[l] op b_row[l]);                                   \
  }                                                                            \
  result = (int)count;

/* --- End Section C: DB SET OPERATION MACROS — CPU --- */

/* ===========================================================================
   SECTION D: DB SET OPERATION MACROS — GPU
   =========================================================================== */
#ifndef NOGPU

/* Update a device array from the host (or vice versa) */
#define UPDATE_GPU_ARRAY(dir, array, index1, index2, dev_id)                   \
  _Pragma(                                                                     \
      STRINGIFY(omp target update dir(array [index1:index2]) device(dev_id)))

/* Map or unmap a device array via enter/exit data */
#define TARGET_GPU_ARRAY(point, dir, array, index1, index2, dev_id)            \
  _Pragma(STRINGIFY(omp target point data map(dir : array [index1:index2])     \
                        device(dev_id)))

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

/* --- End Section D: DB SET OPERATION MACROS — GPU --- */
