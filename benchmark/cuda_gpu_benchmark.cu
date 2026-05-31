/*
 * CUDA Unified GPU Benchmark: All-Pairs Popcount Intersection
 * 
 * Standalone CUDA benchmark (no dependency on Bit library for GPU code)
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#include "gpu_kernels.h"
#include "gpu_harness.h"

static void summarize_results(const char *test, int64_t timeElapsed,
                              int iteration, int result, float speedup) {
    printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
    printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
    printf("GPU iteration: %3d \t", iteration);
    printf("Result: %d\t", result);
    printf("Speedup factor: %.2f\n", speedup);
}

void compute_cpu_popcount_reference(
    const uint64_t *queries,
    const uint64_t *refs,
    size_t bitset_bits,
    size_t num_queries,
    size_t num_refs,
    uint32_t *cpu_results)
{
    const size_t words_per_bitset = (bitset_bits + 63) / 64;
    
    for (size_t qi = 0; qi < num_queries; ++qi) {
        for (size_t ri = 0; ri < num_refs; ++ri) {
            uint32_t count = 0;
            for (size_t wi = 0; wi < words_per_bitset; ++wi) {
                uint64_t combined = 
                    queries[qi * words_per_bitset + wi] &
                    refs[ri * words_per_bitset + wi];
                count += (uint32_t)__builtin_popcountll(combined);
            }
            cpu_results[qi * num_refs + ri] = count;
        }
    }
}

static void set_bit_range(uint64_t *words, size_t lo, size_t hi) {
    for (size_t bit = lo; bit <= hi; ++bit) {
        words[bit / 64] |= (1ull << (bit % 64));
    }
}

int main(int argc, char *argv[]) {
        if (argc != 5 && argc != 6 && argc != 7) {
        fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference bitsets> <gpu_iterations> [<gpu_id> [<ref_layout>]]\n",
            argv[0]);
        fprintf(stderr, "Example: %s 1024 1000 1000000 10 0 transpose\n", argv[0]);
        fprintf(stderr,
            "This will create 1000 bitsets of size 1024 and run 10 GPU-only containerized intersection-count iterations on GPU 0.\n");
        return EXIT_FAILURE;
        }

    int size = atoi(argv[1]);
    int num_of_bits = atoi(argv[2]);
    int num_of_ref_bits = atoi(argv[3]);
    int gpu_iterations = atoi(argv[4]);
    int gpu_id = 0;
    if (argc == 6) {
        if (strcmp(argv[5], "transpose") == 0 || strcmp(argv[5], "transposed") == 0) {
            // leave gpu_id at default 0, use transpose layout
        } else {
            gpu_id = atoi(argv[5]);
        }
    }
    RefLayout ref_layout = RefLayout::RowMajor;
    if (argc == 6) {
        if (strcmp(argv[5], "transpose") == 0 || strcmp(argv[5], "transposed") == 0) {
            ref_layout = RefLayout::Transposed;
        }
    } else if (argc == 7) {
        gpu_id = atoi(argv[5]);
        if (strcmp(argv[6], "transpose") == 0 || strcmp(argv[6], "transposed") == 0) {
            ref_layout = RefLayout::Transposed;
        } else if (strcmp(argv[6], "row") == 0 || strcmp(argv[6], "row-major") == 0) {
            ref_layout = RefLayout::RowMajor;
        } else {
            fprintf(stderr, "Invalid ref_layout '%s'. Use row or transpose.\n", argv[6]);
            return EXIT_FAILURE;
        }
    }

    if (size <= 0 || num_of_bits <= 0 || num_of_ref_bits <= 0 || gpu_iterations <= 0) {
        fprintf(stderr, "Error: size, number of bits, number of ref bits, and GPU iterations must be positive integers.\n");
        return EXIT_FAILURE;
    }
    if (gpu_iterations > 1024) {
        fprintf(stderr, "Warning: gpu_iterations capped to 1024\n");
        gpu_iterations = 1024;
    }
    if (size < 128) {
        fprintf(stderr, "Warning: size increased to 128\n");
        size = 128;
    }

    cudaSetDevice(gpu_id);

#ifndef NDEBUG
    printf("Debug mode is enabled.\n");
#else
    printf("Debug mode is disabled.\n");
#endif

    printf("Starting GPU-only benchmark\n");

    const size_t words_per_bitset = (size_t)(size + 63) / 64;
    const size_t query_words = words_per_bitset * (size_t)num_of_bits;
    const size_t ref_words = words_per_bitset * (size_t)num_of_ref_bits;

    uint64_t *h_queries = (uint64_t *)calloc(query_words, sizeof(uint64_t));
    uint64_t *h_refs = (uint64_t *)calloc(ref_words, sizeof(uint64_t));
    assert(h_queries && h_refs);

    for (int i = 0; i < num_of_bits; ++i) {
        set_bit_range(h_queries + (size_t)i * words_per_bitset, (size_t)size / 2, (size_t)size - 1);
    }
    for (int i = 0; i < num_of_ref_bits; ++i) {
        set_bit_range(h_refs + (size_t)i * words_per_bitset, (size_t)size / 2, (size_t)size - 1);
    }
    if (num_of_bits > 0 && size / 2 > 0) {
        set_bit_range(h_queries + 0 * words_per_bitset, (size_t)size / 2 - 1, (size_t)size / 2 + 5);
    }
    if (num_of_ref_bits > 0) {
        set_bit_range(h_refs + 0 * words_per_bitset, (size_t)size / 2, (size_t)size / 2 + 5);
    }

    GPUBenchmarkResult wwg_result = benchmark_popcount_intersection_gpu(
        h_queries, h_refs, nullptr, (size_t)size,
        (size_t)num_of_bits, (size_t)num_of_ref_bits, gpu_iterations, "WWG");
        ref_layout);

    double avg_algorithm_time = 0.0;
    double avg_total_operation_time = 0.0;
    for (int i = 0; i < gpu_iterations; ++i) {
        avg_algorithm_time += wwg_result.per_iter_kernel_ms[i] * 1.0e6;
        avg_total_operation_time += wwg_result.per_iter_total_ms[i] * 1.0e6;
    }
    avg_algorithm_time /= gpu_iterations;
    avg_total_operation_time /= gpu_iterations;

    const double db1_bytes_per_iter = (double)words_per_bitset * 8.0 * num_of_bits;
    const double results_bytes_per_iter = (double)num_of_bits * num_of_ref_bits * sizeof(int);
    const double db2_bytes_resident = (double)words_per_bitset * 8.0 * num_of_ref_bits;
    const double payload_per_iteration = (db1_bytes_per_iter + results_bytes_per_iter) / (1024.0 * 1024.0 * 1024.0);

    puts("GPU Algorithm Timing:");
    for (int i = 0; i < gpu_iterations; ++i) {
        int64_t ns = (int64_t)llround(wwg_result.per_iter_kernel_ms[i] * 1.0e6);
        summarize_results("Container - GPU - CUDA", ns, i + 1,
                          (int)wwg_result.per_iter_results[i],
                          (float)(wwg_result.per_iter_kernel_ms[0] / wwg_result.per_iter_kernel_ms[i]));
    }

    puts("GPU Algorithm + PCIe Timings:");
    for (int i = 0; i < gpu_iterations; ++i) {
        int64_t ns = (int64_t)llround(wwg_result.per_iter_total_ms[i] * 1.0e6);
        summarize_results("Container - GPU - CUDA", ns, i + 1,
                          (int)wwg_result.per_iter_results[i],
                          (float)(wwg_result.per_iter_total_ms[0] / wwg_result.per_iter_total_ms[i]));
    }

    puts("CPU Overhead Timings:");
    for (int i = 0; i < gpu_iterations; ++i) {
        int64_t ns = (int64_t)llround(wwg_result.per_iter_cpu_overhead_ms[i] * 1.0e6);
        summarize_results("Container - GPU - CUDA", ns, i + 1,
                          (int)wwg_result.per_iter_results[i],
                          (float)(wwg_result.per_iter_cpu_overhead_ms[0] / wwg_result.per_iter_cpu_overhead_ms[i]));
    }

    puts("\nPer-Iteration Data Movement Breakdown:");
    printf("  db1 (queries) uploaded:   %.6lf GB\n", db1_bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
    printf("  results downloaded:       %.6lf GB\n", results_bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
    printf("  db2 (reference, resident): %.6lf GB (NOT transferred)\n", db2_bytes_resident / (1024.0 * 1024.0 * 1024.0));
    printf("  Total per-iteration:      %.6lf GB\n", payload_per_iteration);

    puts("\nEstimated Throughput (iterations 1-N, steady-state):");
    printf("GPU compute throughput:      %.3lf GB/s\n", payload_per_iteration / ((double)avg_algorithm_time / 1E9));
    printf("Total operation throughput:  %.3lf GB/s\n", payload_per_iteration / ((double)avg_total_operation_time / 1E9));
    puts("\nNote: Total operation throughput includes GPU compute time, data staging,");
    puts("      and PCIe transfers combined, representing user-perceived performance.");

    double cuda_compute_gbps = payload_per_iteration / ((double)avg_algorithm_time / 1E9);
            printf("CUDA_SUMMARY,backend=CUDA,method=WWG,bitset_bits=%d,nelem=%d,iterations=%d,avg_ns=%.3f,gbps=%.6f,max=%u\n",
                size, num_of_bits, gpu_iterations,
                (double)avg_algorithm_time, (double)cuda_compute_gbps,
                (unsigned int)wwg_result.per_iter_results.back());

    free(wwg_result.gpu_results);
    free(h_queries);
    free(h_refs);

    return EXIT_SUCCESS;
}
