#include "openmp_bit_helpers.h"
#include <stdint.h>
#include <stddef.h>
#include <math.h>

// Use OpenMP for the CPU reference computations in the helper module.
// The corresponding object file is now built with host-only OpenMP flags.


void compute_cpu_popcount_reference(const uint64_t *queries,
                                    const uint64_t *refs,
                                    size_t bitset_bits,
                                    size_t num_queries,
                                    size_t num_refs,
                                    uint32_t *cpu_results) {
    if (num_queries == 0 || num_refs == 0) {
        return;
    }
    const size_t words_per_bitset = (bitset_bits + 63) / 64;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t qi = 0; qi < num_queries; ++qi) {
        for (size_t ri = 0; ri < num_refs; ++ri) {
            uint32_t count = 0;
            for (size_t wi = 0; wi < words_per_bitset; ++wi) {
                uint64_t combined = queries[qi * words_per_bitset + wi] &
                    refs[ri * words_per_bitset + wi];
                count += (uint32_t)__builtin_popcountll(combined);
            }
            cpu_results[qi * num_refs + ri] = count;
        }
    }
}
void compute_cpu_popcount_reference_32bit(
    const uint32_t *queries,
    const uint32_t *refs,
    size_t bitset_bits,
    size_t num_queries,
    size_t num_refs,
    uint32_t *cpu_results)
{
    if (num_queries == 0 || num_refs == 0) {
        return;
    }
    const size_t words_per_bitset = (bitset_bits + 31) / 32;
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t qi = 0; qi < num_queries; ++qi) {
        for (size_t ri = 0; ri < num_refs; ++ri) {
            uint32_t count = 0;
            for (size_t wi = 0; wi < words_per_bitset; ++wi) {
                uint32_t combined = queries[qi * words_per_bitset + wi] &
                    refs[ri * words_per_bitset + wi];
                count += (uint32_t)__builtin_popcount(combined);
            }
            cpu_results[qi * num_refs + ri] = count;
        }
    }
}

void compare_gpu_to_cpu_results(
    const int *gpu_results,
    const uint32_t *cpu_results,
    size_t num_queries,
    size_t num_refs,
    size_t *agreements,
    size_t *disagreements,
    uint32_t *max_result)
{
    size_t agree = 0, disagree = 0;
    uint32_t maxval = 0;
    for (size_t i = 0; i < num_queries * num_refs; ++i) {
        if ((uint32_t)gpu_results[i] == cpu_results[i]) {
            ++agree;
        } else {
            ++disagree;
        }
        if ((uint32_t)gpu_results[i] > maxval) {
            maxval = (uint32_t)gpu_results[i];
        }
    }
    if (agreements) *agreements = agree;
    if (disagreements) *disagreements = disagree;
    if (max_result) *max_result = maxval;
}

void compute_mean_stddev(const double *values,
                         size_t count,
                         double *mean,
                         double *stddev)
{
    if (mean) *mean = 0.0;
    if (stddev) *stddev = 0.0;
    if (count == 0 || values == NULL) {
        return;
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += values[i];
    }
    double avg = sum / (double)count;
    double variance = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = values[i] - avg;
        variance += diff * diff;
    }
    variance /= (double)count;
    if (mean) *mean = avg;
    if (stddev) *stddev = sqrt(variance);
}

void compute_int64_mean_stddev(const int64_t *values,
                              size_t count,
                              double *mean,
                              double *stddev)
{
    if (mean) *mean = 0.0;
    if (stddev) *stddev = 0.0;
    if (count == 0 || values == NULL) {
        return;
    }

    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += (double)values[i];
    }
    double avg = sum / (double)count;
    double variance = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double diff = (double)values[i] - avg;
        variance += diff * diff;
    }
    variance /= (double)count;
    if (mean) *mean = avg;
    if (stddev) *stddev = sqrt(variance);
}
