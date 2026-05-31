/*
 * HIP Unified GPU Benchmark: All-Pairs Popcount Intersection
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <math.h>
#include <vector>
#include <random>
#include <ctime>
#include "gpu_kernels.h"
#include "gpu_harness.h"
#include "openmp_bit_helpers.h"

static void summarize_results(const char *test, int64_t timeElapsed,
                              int iteration, int result, float speedup) {
    printf("Total time for %-35s: %15ld ns\t", test, timeElapsed);
    printf("Searches per second : %0.2f\t", (float)1E9 / timeElapsed);
    printf("GPU iteration: %3d \t", iteration);
    printf("Result: %d\t", result);
    printf("Speedup factor: %.2f\n", speedup);
}

template <typename T, typename Engine>
static void fill_random_bitsets(T *bitsets,
                                size_t num_bitsets,
                                size_t words_per_bitset,
                                Engine &rng)
{
    std::uniform_int_distribution<T> distribution(0);
    const size_t total_words = num_bitsets * words_per_bitset;
    for (size_t i = 0; i < total_words; ++i) {
        bitsets[i] = distribution(rng);
    }
}

template <typename T>
static void set_bit_range(T *words, size_t lo, size_t hi) {
    const size_t bits_per_word = sizeof(T) * 8;
    for (size_t bit = lo; bit <= hi; ++bit) {
        words[bit / bits_per_word] |= (T(1) << (bit % bits_per_word));
    }
}

static bool parse_word_bits(const char *arg, unsigned int *word_bits) {
    if (strcmp(arg, "32") == 0) {
        *word_bits = 32u;
        return true;
    }
    if (strcmp(arg, "64") == 0) {
        *word_bits = 64u;
        return true;
    }
    return false;
}

int main(int argc, char *argv[]) {
        if (argc < 5 || argc > 8) {
        fprintf(stderr,
            "Usage: %s <size> <number of bitsets> <number of reference bitsets> <gpu_iterations> [<gpu_id> [<word_bits>]]\n",
            argv[0]);
            fprintf(stderr, "Example: %s 1024 1000 1000000 10 0 64\n", argv[0]);
        fprintf(stderr,
            "This will create 1000 bitsets of size 1024 and run 10 GPU-only containerized intersection-count iterations on GPU 0.\n");
        return EXIT_FAILURE;
    }

    int size = atoi(argv[1]);
    int num_of_bits = atoi(argv[2]);
    int num_of_ref_bits = atoi(argv[3]);
    int gpu_iterations = atoi(argv[4]);
    int gpu_id = 0;
    RefLayout ref_layout = RefLayout::RowMajor;
    unsigned int word_bits = 64u;
    int arg_idx = 5;
    if (arg_idx < argc) {
        if (parse_word_bits(argv[arg_idx], &word_bits)) {
            arg_idx++;
        } else {
            gpu_id = atoi(argv[arg_idx++]);
        }
    }
    if (arg_idx < argc) {
        if (parse_word_bits(argv[arg_idx], &word_bits)) {
            arg_idx++;
        } else {
            fprintf(stderr, "Invalid argument '%s'. Expected gpu_id or 32/64.\n", argv[arg_idx]);
            return EXIT_FAILURE;
        }
    }
    if (arg_idx < argc) {
        if (parse_word_bits(argv[arg_idx], &word_bits)) {
            arg_idx++;
        } else {
            fprintf(stderr, "Invalid word size '%s'. Use 32 or 64.\n", argv[arg_idx]);
            return EXIT_FAILURE;
        }
    }
    if (arg_idx != argc) {
        fprintf(stderr, "Too many arguments.\n");
        return EXIT_FAILURE;
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

    GPU_CHECK(GPU_SET_DEVICE(gpu_id));

#ifndef NDEBUG
    printf("Debug mode is enabled.\n");
#else
    printf("Debug mode is disabled.\n");
#endif

    printf("Starting GPU-only benchmark\n");

    const size_t words_per_bitset = (size_t)(size + word_bits - 1) / word_bits;
    if (word_bits == 32u) {
        std::mt19937_64 rng(static_cast<unsigned long>(std::time(nullptr)));
        uint32_t *h_queries = (uint32_t *)calloc(words_per_bitset * (size_t)num_of_bits, sizeof(uint32_t));
        uint32_t *h_refs = (uint32_t *)calloc(words_per_bitset * (size_t)num_of_ref_bits, sizeof(uint32_t));
        uint32_t *cpu_results = (uint32_t *)calloc((size_t)num_of_bits * (size_t)num_of_ref_bits, sizeof(uint32_t));
        assert(h_queries && h_refs && cpu_results);

        fill_random_bitsets<uint32_t>(h_queries, (size_t)num_of_bits, words_per_bitset, rng);
        fill_random_bitsets<uint32_t>(h_refs, (size_t)num_of_ref_bits, words_per_bitset, rng);
        compute_cpu_popcount_reference_32bit(h_queries, h_refs, (size_t)size,
                                            (size_t)num_of_bits, (size_t)num_of_ref_bits,
                                            cpu_results);

        GPUBenchmarkResult result = benchmark_popcount_intersection_gpu(
            h_queries, h_refs, cpu_results, (size_t)size,
            (size_t)num_of_bits, (size_t)num_of_ref_bits, gpu_iterations, "WWG",
            ref_layout, word_bits);
        size_t agreements = 0;
        size_t disagreements = 0;
        compare_gpu_to_cpu_results((const int *)result.gpu_results,
                                   cpu_results,
                                   (size_t)num_of_bits,
                                   (size_t)num_of_ref_bits,
                       &agreements,
                       &disagreements,
                       NULL);
        free(cpu_results);

        double avg_algorithm_time = 0.0;
        double avg_total_operation_time = 0.0;
        for (int i = 0; i < gpu_iterations; ++i) {
            avg_algorithm_time += result.per_iter_kernel_ms[i] * 1.0e6;
            avg_total_operation_time += result.per_iter_total_ms[i] * 1.0e6;
        }
        avg_algorithm_time /= gpu_iterations;
        avg_total_operation_time /= gpu_iterations;

        const double db1_bytes_per_iter = (double)words_per_bitset * 4.0 * num_of_bits;
        const double results_bytes_per_iter = (double)num_of_bits * num_of_ref_bits * sizeof(int);
        const double db2_bytes_resident = (double)words_per_bitset * 4.0 * num_of_ref_bits;
        const double payload_per_iteration = (db1_bytes_per_iter + results_bytes_per_iter) / (1024.0 * 1024.0 * 1024.0);

        puts("GPU Algorithm Timing:");
        for (int i = 0; i < gpu_iterations; ++i) {
            int64_t ns = (int64_t)llround(result.per_iter_kernel_ms[i] * 1.0e6);
            summarize_results("Container - GPU - HIP", ns, i + 1,
                              (int)result.per_iter_results[i],
                              (float)(result.per_iter_kernel_ms[0] / result.per_iter_kernel_ms[i]));
        }

        puts("GPU Algorithm + PCIe Timings:");
        for (int i = 0; i < gpu_iterations; ++i) {
            int64_t ns = (int64_t)llround(result.per_iter_total_ms[i] * 1.0e6);
            summarize_results("Container - GPU - HIP", ns, i + 1,
                              (int)result.per_iter_results[i],
                              (float)(result.per_iter_total_ms[0] / result.per_iter_total_ms[i]));
        }

        puts("CPU Overhead Timings:");
        for (int i = 0; i < gpu_iterations; ++i) {
            int64_t ns = (int64_t)llround(result.per_iter_cpu_overhead_ms[i] * 1.0e6);
            summarize_results("Container - GPU - HIP", ns, i + 1,
                              (int)result.per_iter_results[i],
                              (float)(result.per_iter_cpu_overhead_ms[0] / result.per_iter_cpu_overhead_ms[i]));
        }

        puts("\nPer-Iteration Data Movement Breakdown:");
        printf("  db1 (queries) uploaded:   %.6lf GB\n", db1_bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
        printf("  results downloaded:       %.6lf GB\n", results_bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
        printf("  db2 (reference, resident): %.6lf GB (NOT transferred)\n", db2_bytes_resident / (1024.0 * 1024.0 * 1024.0));
        printf("  Total per-iteration:      %.6lf GB\n", payload_per_iteration);

        std::vector<double> kernel_ns(gpu_iterations);
        std::vector<double> total_ns(gpu_iterations);
        std::vector<double> compute_gbps(gpu_iterations);
        std::vector<double> total_gbps(gpu_iterations);
        for (int i = 0; i < gpu_iterations; ++i) {
            kernel_ns[i] = result.per_iter_kernel_ms[i] * 1.0e6;
            total_ns[i] = result.per_iter_total_ms[i] * 1.0e6;
            compute_gbps[i] = payload_per_iteration / (kernel_ns[i] / 1e9);
            total_gbps[i] = payload_per_iteration / (total_ns[i] / 1e9);
        }
        double avg_kernel_ns = 0.0;
        double stddev_kernel_ns = 0.0;
        compute_mean_stddev(kernel_ns.data(), gpu_iterations, &avg_kernel_ns, &stddev_kernel_ns);
        double avg_compute_gbps = 0.0;
        double stddev_compute_gbps = 0.0;
        compute_mean_stddev(compute_gbps.data(), gpu_iterations, &avg_compute_gbps, &stddev_compute_gbps);
        double avg_total_ns = 0.0;
        double stddev_total_ns = 0.0;
        compute_mean_stddev(total_ns.data(), gpu_iterations, &avg_total_ns, &stddev_total_ns);
        double avg_total_gbps = 0.0;
        double stddev_total_gbps = 0.0;
        compute_mean_stddev(total_gbps.data(), gpu_iterations, &avg_total_gbps, &stddev_total_gbps);
        puts("\nEstimated Throughput (iterations 1-N, steady-state):");
        printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n", avg_kernel_ns, stddev_kernel_ns);
        printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n", avg_compute_gbps, stddev_compute_gbps);
        printf("Total operation time: mean=%.3f ns, stddev=%.3f ns\n", avg_total_ns, stddev_total_ns);
        printf("Total operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n", avg_total_gbps, stddev_total_gbps);
        printf("HIP_SUMMARY,backend=HIP,method=WWG,bitset_bits=%d,nelem=%d,iterations=%d,avg_ns=%.3f,stddev_ns=%.3f,gbps=%.6f,gbps_stddev=%.6f,max=%u\n",
            size, num_of_bits, gpu_iterations,
            avg_kernel_ns, stddev_kernel_ns,
            avg_compute_gbps, stddev_compute_gbps,
            (unsigned int)result.per_iter_results.back());

         printf("GPU/CPU agreement: %zu, disagreement: %zu\n",
             agreements, disagreements);
         if (disagreements > 0) {
             printf("WARNING: GPU results disagree with CPU reference\n");
         }

        free(result.gpu_results);
        free(h_queries);
        free(h_refs);
        return EXIT_SUCCESS;
    }

    std::mt19937_64 rng(static_cast<unsigned long>(std::time(nullptr)));
    uint64_t *h_queries = (uint64_t *)calloc(words_per_bitset * (size_t)num_of_bits, sizeof(uint64_t));
    uint64_t *h_refs = (uint64_t *)calloc(words_per_bitset * (size_t)num_of_ref_bits, sizeof(uint64_t));
    uint32_t *cpu_results = (uint32_t *)calloc((size_t)num_of_bits * (size_t)num_of_ref_bits, sizeof(uint32_t));
    assert(h_queries && h_refs && cpu_results);

    fill_random_bitsets<uint64_t>(h_queries, (size_t)num_of_bits, words_per_bitset, rng);
    fill_random_bitsets<uint64_t>(h_refs, (size_t)num_of_ref_bits, words_per_bitset, rng);
    compute_cpu_popcount_reference(h_queries, h_refs, (size_t)size,
                                   (size_t)num_of_bits, (size_t)num_of_ref_bits,
                                   cpu_results);

    GPUBenchmarkResult result = benchmark_popcount_intersection_gpu(
        h_queries, h_refs, cpu_results, (size_t)size,
        (size_t)num_of_bits, (size_t)num_of_ref_bits, gpu_iterations, "WWG",
        ref_layout, word_bits);
    size_t agreements = 0;
    size_t disagreements = 0;
    compare_gpu_to_cpu_results((const int *)result.gpu_results,
                               cpu_results,
                               (size_t)num_of_bits,
                               (size_t)num_of_ref_bits,
                               &agreements,
                               &disagreements,
                               NULL);
    free(cpu_results);

    double avg_algorithm_time = 0.0;
    double avg_total_operation_time = 0.0;
    for (int i = 0; i < gpu_iterations; ++i) {
        avg_algorithm_time += result.per_iter_kernel_ms[i] * 1.0e6;
        avg_total_operation_time += result.per_iter_total_ms[i] * 1.0e6;
    }
    avg_algorithm_time /= gpu_iterations;
    avg_total_operation_time /= gpu_iterations;

    const double db1_bytes_per_iter = (double)words_per_bitset * 8.0 * num_of_bits;
    const double results_bytes_per_iter = (double)num_of_bits * num_of_ref_bits * sizeof(int);
    const double db2_bytes_resident = (double)words_per_bitset * 8.0 * num_of_ref_bits;
    const double payload_per_iteration = (db1_bytes_per_iter + results_bytes_per_iter) / (1024.0 * 1024.0 * 1024.0);

    puts("GPU Algorithm Timing:");
    for (int i = 0; i < gpu_iterations; ++i) {
        int64_t ns = (int64_t)llround(result.per_iter_kernel_ms[i] * 1.0e6);
        summarize_results("Container - GPU - HIP", ns, i + 1,
                          (int)result.per_iter_results[i],
                          (float)(result.per_iter_kernel_ms[0] / result.per_iter_kernel_ms[i]));
    }

    puts("GPU Algorithm + PCIe Timings:");
    for (int i = 0; i < gpu_iterations; ++i) {
        int64_t ns = (int64_t)llround(result.per_iter_total_ms[i] * 1.0e6);
        summarize_results("Container - GPU - HIP", ns, i + 1,
                          (int)result.per_iter_results[i],
                          (float)(result.per_iter_total_ms[0] / result.per_iter_total_ms[i]));
    }

    puts("CPU Overhead Timings:");
    for (int i = 0; i < gpu_iterations; ++i) {
        int64_t ns = (int64_t)llround(result.per_iter_cpu_overhead_ms[i] * 1.0e6);
        summarize_results("Container - GPU - HIP", ns, i + 1,
                          (int)result.per_iter_results[i],
                          (float)(result.per_iter_cpu_overhead_ms[0] / result.per_iter_cpu_overhead_ms[i]));
    }

    puts("\nPer-Iteration Data Movement Breakdown:");
    printf("  db1 (queries) uploaded:   %.6lf GB\n", db1_bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
    printf("  results downloaded:       %.6lf GB\n", results_bytes_per_iter / (1024.0 * 1024.0 * 1024.0));
    printf("  db2 (reference, resident): %.6lf GB (NOT transferred)\n", db2_bytes_resident / (1024.0 * 1024.0 * 1024.0));
    printf("  Total per-iteration:      %.6lf GB\n", payload_per_iteration);

    std::vector<double> kernel_ns(gpu_iterations);
    std::vector<double> total_ns(gpu_iterations);
    std::vector<double> compute_gbps(gpu_iterations);
    std::vector<double> total_gbps(gpu_iterations);
    for (int i = 0; i < gpu_iterations; ++i) {
        kernel_ns[i] = result.per_iter_kernel_ms[i] * 1.0e6;
        total_ns[i] = result.per_iter_total_ms[i] * 1.0e6;
        compute_gbps[i] = payload_per_iteration / (kernel_ns[i] / 1e9);
        total_gbps[i] = payload_per_iteration / (total_ns[i] / 1e9);
    }
    double avg_kernel_ns = 0.0;
    double stddev_kernel_ns = 0.0;
    compute_mean_stddev(kernel_ns.data(), gpu_iterations, &avg_kernel_ns, &stddev_kernel_ns);
    double avg_compute_gbps = 0.0;
    double stddev_compute_gbps = 0.0;
    compute_mean_stddev(compute_gbps.data(), gpu_iterations, &avg_compute_gbps, &stddev_compute_gbps);
    double avg_total_ns = 0.0;
    double stddev_total_ns = 0.0;
    compute_mean_stddev(total_ns.data(), gpu_iterations, &avg_total_ns, &stddev_total_ns);
    double avg_total_gbps = 0.0;
    double stddev_total_gbps = 0.0;
    compute_mean_stddev(total_gbps.data(), gpu_iterations, &avg_total_gbps, &stddev_total_gbps);
    puts("\nEstimated Throughput (iterations 1-N, steady-state):");
    printf("GPU compute time: mean=%.3f ns, stddev=%.3f ns\n", avg_kernel_ns, stddev_kernel_ns);
    printf("GPU compute throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n", avg_compute_gbps, stddev_compute_gbps);
    printf("Total operation time: mean=%.3f ns, stddev=%.3f ns\n", avg_total_ns, stddev_total_ns);
    printf("Total operation throughput: mean=%.3lf GB/s, stddev=%.3lf GB/s\n", avg_total_gbps, stddev_total_gbps);
    printf("HIP_SUMMARY,backend=HIP,method=WWG,bitset_bits=%d,nelem=%d,iterations=%d,avg_ns=%.3f,stddev_ns=%.3f,gbps=%.6f,gbps_stddev=%.6f,max=%u\n",
            size, num_of_bits, gpu_iterations,
            avg_kernel_ns, stddev_kernel_ns,
            avg_compute_gbps, stddev_compute_gbps,
            (unsigned int)result.per_iter_results.back());

    printf("GPU/CPU agreement: %zu, disagreement: %zu\n",
           agreements, disagreements);
    if (disagreements > 0) {
        printf("WARNING: GPU results disagree with CPU reference\n");
    }

    free(result.gpu_results);
    free(h_queries);
    free(h_refs);

    return EXIT_SUCCESS;
}
