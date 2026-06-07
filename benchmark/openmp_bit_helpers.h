#ifndef OPENMP_BIT_HELPERS_H
#define OPENMP_BIT_HELPERS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void compute_cpu_popcount_reference(
    const uint64_t *queries,
    const uint64_t *refs,
    size_t bitset_bits,
    size_t num_queries,
    size_t num_refs,
    uint32_t *cpu_results);

void compute_cpu_popcount_reference_32bit(
    const uint32_t *queries,
    const uint32_t *refs,
    size_t bitset_bits,
    size_t num_queries,
    size_t num_refs,
    uint32_t *cpu_results);

void compare_gpu_to_cpu_results(
    const int *gpu_results,
    const uint32_t *cpu_results,
    size_t num_queries,
    size_t num_refs,
    size_t *agreements,
    size_t *disagreements,
    uint32_t *max_result);

void compute_mean_stddev(
    const double *values,
    size_t count,
    double *mean,
    double *stddev);

void compute_int64_mean_stddev(
    const int64_t *values,
    size_t count,
    double *mean,
    double *stddev);

#ifdef __cplusplus
}
#endif

#endif // OPENMP_BIT_HELPERS_H
