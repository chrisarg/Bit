#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
typedef cudaEvent_t gpu_event_t;
typedef cudaDeviceProp gpu_device_prop_t;
#define GPU_CHECK(call) do { cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#define GPU_MALLOC cudaMalloc
#define GPU_MEMCPY cudaMemcpy
#define GPU_MEMCPY_H2D cudaMemcpyHostToDevice
#define GPU_MEMCPY_D2H cudaMemcpyDeviceToHost
#define GPU_FREE cudaFree
#define GPU_EVENT_CREATE(e) cudaEventCreate(e)
#define GPU_EVENT_DESTROY(e) cudaEventDestroy(e)
#define GPU_EVENT_RECORD(e, s) cudaEventRecord(e, s)
#define GPU_EVENT_SYNC(e) cudaEventSynchronize(e)
#define GPU_EVENT_ELAPSED_TIME(ms, start, end) cudaEventElapsedTime(ms, start, end)
#define GPU_DEVICE_SYNC cudaDeviceSynchronize
#define GPU_GET_LAST_ERROR cudaGetLastError
#define GPU_GET_DEVICE cudaGetDevice
#define GPU_GET_DEVICE_PROPERTIES cudaGetDeviceProperties
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
typedef hipEvent_t gpu_event_t;
typedef hipDeviceProp_t gpu_device_prop_t;
#define GPU_CHECK(call) do { hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#define GPU_MALLOC hipMalloc
#define GPU_MEMCPY hipMemcpy
#define GPU_MEMCPY_H2D hipMemcpyHostToDevice
#define GPU_MEMCPY_D2H hipMemcpyDeviceToHost
#define GPU_FREE hipFree
#define GPU_EVENT_CREATE(e) hipEventCreate(e)
#define GPU_EVENT_DESTROY(e) hipEventDestroy(e)
#define GPU_EVENT_RECORD(e, s) hipEventRecord(e, s)
#define GPU_EVENT_SYNC(e) hipEventSynchronize(e)
#define GPU_EVENT_ELAPSED_TIME(ms, start, end) hipEventElapsedTime(ms, start, end)
#define GPU_DEVICE_SYNC hipDeviceSynchronize
#define GPU_GET_LAST_ERROR hipGetLastError
#define GPU_GET_DEVICE hipGetDevice
#define GPU_GET_DEVICE_PROPERTIES hipGetDeviceProperties
#else
#error "gpu_harness.h requires CUDA or HIP"
#endif

#include "gpu_kernels.h"

// ===========================================================================
// GPU benchmark harness for all-pairs popcount intersection
// Shared CUDA/HIP benchmark behavior:
//   - refs (db2) uploaded once and kept resident
//   - queries (db1) uploaded each measured iteration
//   - results buffer allocated once and downloaded every measured iteration
//   - first iteration is burn-in and excluded from timing
// Timing semantics:
//   - kernel_time_ns: kernel launch window only (per measured iteration)
//   - total_time_ns: H2D(db1) + kernel + D2H(results) per measured iteration
//   - CPU overhead measures only the host-side result scan time
// ===========================================================================

struct GPUBenchmarkResult {
    uint32_t *gpu_results;
    size_t num_pairs;
    double kernel_time_ns;      // kernel-only
    double total_time_ns;       // per-iteration H2D(db1) + kernel + D2H(results)
    double cpu_overhead_ns;     // host-side result scan time only
    double compute_gbps;
    double total_gbps;          // normalized with OpenMP payload convention
    uint64_t checksum;
    size_t agreements;
    size_t disagreements;
    uint32_t max_result;
    std::vector<double> per_iter_kernel_ms;
    std::vector<double> per_iter_total_ms;
    std::vector<double> per_iter_cpu_overhead_ms;
    std::vector<double> per_iter_d2h_ms;
    std::vector<uint32_t> per_iter_results;
};

enum class RefLayout {
    RowMajor,
    Transposed
};

inline unsigned int choose_dynamic_threads_per_block(size_t words_per_bitset) {
    int device_id = 0;
    GPU_CHECK(GPU_GET_DEVICE(&device_id));

    gpu_device_prop_t props;
    GPU_CHECK(GPU_GET_DEVICE_PROPERTIES(&props, device_id));

    const unsigned int max_threads =
        (props.maxThreadsPerBlock > 0)
            ? static_cast<unsigned int>(props.maxThreadsPerBlock)
            : 256u;
    const unsigned int warp_size =
        (props.warpSize > 0)
            ? static_cast<unsigned int>(props.warpSize)
            : 32u;

    unsigned int desired_threads =
        (words_per_bitset >= static_cast<size_t>(max_threads))
            ? max_threads
            : static_cast<unsigned int>(words_per_bitset);

    if (desired_threads < warp_size) {
        desired_threads = warp_size;
    }

    unsigned int rounded_threads =
        ((desired_threads + warp_size - 1u) / warp_size) * warp_size;

    if (rounded_threads > max_threads) {
        rounded_threads = (max_threads / warp_size) * warp_size;
    }
    if (rounded_threads == 0u) {
        rounded_threads = std::max(1u, std::min(max_threads, warp_size));
    }

    return rounded_threads;
}

#ifdef GPU_OCCUPANCY_TUNING
inline unsigned int choose_occupancy_block_size(size_t dynamic_shared_bytes) {
    int minGridSize = 0;
    int blockSize = 0;
    #ifdef __CUDACC__
    (void)cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        popcount_intersection_matrix_wwg_kernel<uint64_t>,
        dynamic_shared_bytes, 0);
    #elif defined(__HIPCC__)
    (void)hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        popcount_intersection_matrix_wwg_kernel<uint64_t>,
        dynamic_shared_bytes, 0);
    #endif
    return static_cast<unsigned int>(blockSize);
}
#endif

inline unsigned int choose_gpu_tile_size(
    unsigned int num_queries,
    unsigned int num_refs,
    unsigned int words_per_bitset,
    const gpu_device_prop_t &props,
    unsigned int &query_tile,
    unsigned int &ref_tile,
    unsigned int &word_tile,
    unsigned int &threads_per_block)
{
    const unsigned int warp_size = (props.warpSize > 0)
        ? static_cast<unsigned int>(props.warpSize)
        : 32u;
    const unsigned int max_threads = (props.maxThreadsPerBlock > 0)
        ? static_cast<unsigned int>(props.maxThreadsPerBlock)
        : 256u;
    const unsigned int max_shared_words = static_cast<unsigned int>(
        props.sharedMemPerBlock / sizeof(uint64_t));

        auto align_up = [](unsigned int value, unsigned int align) {
            if (align == 0u) return value;
            return ((value + align - 1u) / align) * align;
        };

        const unsigned int kQueryTiles[] = {4u, 8u, 16u, 32u};
        const unsigned int kRefTiles[] = {4u, 8u, 16u, 32u, 64u};
        const unsigned int kWordTiles[] = {1u, 2u, 4u, 8u, 16u, 32u, 64u};

        double best_score = -1.0;
        unsigned int best_q = 4u;
        unsigned int best_r = 4u;
        unsigned int best_w = 1u;
        unsigned int best_threads = warp_size;

        const unsigned int effective_num_queries = std::max(1u, std::min(num_queries, 32u));
        const unsigned int effective_num_refs = std::max(1u, std::min(num_refs, 64u));

        for (unsigned int q : kQueryTiles) {
            const unsigned int q_tile = std::min(q, effective_num_queries);
            if (q_tile == 0u) continue;

            for (unsigned int r : kRefTiles) {
                const unsigned int r_tile = std::min(r, effective_num_refs);
                if (r_tile == 0u) continue;

                for (unsigned int w : kWordTiles) {
                    if (w > words_per_bitset) {
                        break;
                    }
                    if (max_shared_words > 0u && q_tile > 0u && q_tile * w > max_shared_words) {
                        break;
                    }

                    const unsigned int pair_count = q_tile * r_tile;
                    if (pair_count == 0u) {
                        continue;
                    }

                    unsigned int candidate_threads = align_up(std::min(pair_count, max_threads), warp_size);
                    if (candidate_threads == 0u) {
                        candidate_threads = warp_size;
                    }
                    if (candidate_threads > max_threads) {
                        candidate_threads = max_threads;
                    }

                    const double query_weight = static_cast<double>(q_tile);
                    const double ref_weight = static_cast<double>(r_tile);
                    const double reuse_factor = ref_weight;
                    const double occupancy_factor = static_cast<double>(pair_count) / static_cast<double>(candidate_threads);
                    const double shared_penalty = (max_shared_words > 0u)
                        ? 1.0 + (static_cast<double>(q_tile) * static_cast<double>(w)) / static_cast<double>(max_shared_words)
                        : 1.0;

                    const double candidate_score = occupancy_factor * reuse_factor / shared_penalty;

                    if (candidate_score > best_score) {
                        best_score = candidate_score;
                        best_q = q_tile;
                        best_r = r_tile;
                        best_w = w;
                        best_threads = candidate_threads;
                    }
                }
            }
        }

        query_tile = std::min(best_q, num_queries);
        ref_tile = std::min(best_r, num_refs);
        word_tile = std::min(best_w, words_per_bitset);
        if (word_tile == 0u) {
            word_tile = 1u;
        }

        const size_t shared_bytes = static_cast<size_t>(query_tile) * word_tile * sizeof(uint64_t);
        unsigned int candidate_threads = best_threads;
        #ifdef GPU_OCCUPANCY_TUNING
        candidate_threads = choose_occupancy_block_size(shared_bytes);
        if (candidate_threads == 0u) {
            candidate_threads = best_threads;
        }
        #endif

        threads_per_block = std::min(candidate_threads, max_threads);
        threads_per_block = align_up(threads_per_block, warp_size);
        if (threads_per_block > max_threads) {
            threads_per_block = (max_threads / warp_size) * warp_size;
        }
        if (threads_per_block == 0u) {
            threads_per_block = warp_size;
        }

        const unsigned int pair_count = query_tile * ref_tile;
        if (threads_per_block > pair_count) {
            threads_per_block = std::min(std::max(warp_size, align_up(pair_count, warp_size)), max_threads);
            if (threads_per_block > pair_count) {
                threads_per_block = std::min(pair_count, warp_size);
            }
        }

        if (threads_per_block == 0u) {
            threads_per_block = std::min(warp_size, max_threads);
        }

        return threads_per_block;

}

inline GPUBenchmarkResult benchmark_popcount_intersection_gpu(
    const uint64_t *h_queries,
    const uint64_t *h_refs,
    const uint32_t *cpu_results,    // optional verification data
    size_t bitset_bits,
    size_t num_queries,
    size_t num_refs,
    int iterations,
    const char *method_label,
    RefLayout ref_layout = RefLayout::RowMajor)
{
    GPUBenchmarkResult result = {};

    const size_t words_per_bitset = (bitset_bits + 63) / 64;
    const size_t queries_bytes = words_per_bitset * num_queries * sizeof(uint64_t);
    const size_t refs_bytes = words_per_bitset * num_refs * sizeof(uint64_t);
    const size_t results_bytes = num_queries * num_refs * sizeof(uint32_t);

    // Payload (OpenMP definition): queries + results, refs resident
    const double payload_per_iter_gb = 
        (queries_bytes + results_bytes) / (1024.0 * 1024.0 * 1024.0);

    // Allocate GPU memory
    uint64_t *d_queries = nullptr;
    uint64_t *d_refs = nullptr;
    uint64_t *d_refs_transposed = nullptr;
    uint32_t *d_results = nullptr;

    GPU_CHECK(GPU_MALLOC(&d_queries, queries_bytes));
    GPU_CHECK(GPU_MALLOC(&d_refs, refs_bytes));
    if (ref_layout == RefLayout::Transposed) {
        GPU_CHECK(GPU_MALLOC(&d_refs_transposed, refs_bytes));
    }
    GPU_CHECK(GPU_MALLOC(&d_results, results_bytes));

    // Upload refs (resident, upload once before benchmark)
    GPU_CHECK(GPU_MEMCPY(d_refs, h_refs, refs_bytes, GPU_MEMCPY_H2D));

    // Set up timing
    gpu_event_t start_event = nullptr;
    gpu_event_t stop_event = nullptr;
    GPU_CHECK(GPU_EVENT_CREATE(&start_event));
    GPU_CHECK(GPU_EVENT_CREATE(&stop_event));

    if (num_refs > static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
        num_queries > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        fprintf(stderr, "Error: num_refs/num_queries exceed GPU grid dimension limits.
");
        GPU_CHECK(GPU_FREE(d_results));
        GPU_CHECK(GPU_FREE(d_refs));
        GPU_CHECK(GPU_FREE(d_queries));
        return result;
    }
    if (words_per_bitset > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
        fprintf(stderr, "Error: words_per_bitset exceeds kernel parameter range.
");
        GPU_CHECK(GPU_FREE(d_results));
        GPU_CHECK(GPU_FREE(d_refs));
        GPU_CHECK(GPU_FREE(d_queries));
        return result;
    }
    const unsigned int words_per_bitset_u = static_cast<unsigned int>(words_per_bitset);
    const unsigned int num_queries_u = static_cast<unsigned int>(num_queries);
    const unsigned int num_refs_u = static_cast<unsigned int>(num_refs);
    const unsigned int max_gpu_blocks = 65535u;

    if (ref_layout == RefLayout::Transposed) {
        const unsigned int transpose_threads = 256u;
        const unsigned int transpose_total = num_refs_u * words_per_bitset_u;
        const unsigned int transpose_blocks =
            (transpose_total + transpose_threads - 1u) / transpose_threads;
        transpose_refs_kernel<uint64_t><<<transpose_blocks, transpose_threads>>>(
            d_refs, d_refs_transposed,
            num_refs_u, words_per_bitset_u);
        GPU_CHECK(GPU_GET_LAST_ERROR());
    }

    gpu_device_prop_t props;
    GPU_CHECK(GPU_GET_DEVICE_PROPERTIES(&props, 0));

    unsigned int query_tile = 0u;
    unsigned int ref_tile = 0u;
    unsigned int word_tile = 0u;
    unsigned int threads_per_block = 0u;
    choose_gpu_tile_size(num_queries_u, num_refs_u, words_per_bitset_u,
                         props, query_tile, ref_tile, word_tile,
                         threads_per_block);

    unsigned int blocks_x = (num_queries_u + query_tile - 1u) / query_tile;
    unsigned int blocks_y = (num_refs_u + ref_tile - 1u) / ref_tile;
    if (blocks_x > max_gpu_blocks) {
        query_tile = (num_queries_u + max_gpu_blocks - 1u) / max_gpu_blocks;
        blocks_x = (num_queries_u + query_tile - 1u) / query_tile;
    }
    if (blocks_y > max_gpu_blocks) {
        ref_tile = (num_refs_u + max_gpu_blocks - 1u) / max_gpu_blocks;
        blocks_y = (num_refs_u + ref_tile - 1u) / ref_tile;
    }

    const size_t shared_bytes = static_cast<size_t>(query_tile) * word_tile * sizeof(uint64_t);

    const uint64_t *d_refs_used =
        (ref_layout == RefLayout::Transposed) ? d_refs_transposed : d_refs;
    auto launch_kernel = [&]() {
        if (strcmp(method_label, "WWG") == 0) {
            if (ref_layout == RefLayout::Transposed) {
                popcount_intersection_matrix_wwg_kernel<uint64_t, true><<<dim3(blocks_x, blocks_y),
                                                              threads_per_block,
                                                              shared_bytes>>>(
                    d_queries, d_refs_used,
                    words_per_bitset_u, words_per_bitset_u,
                    num_queries_u, num_refs_u,
                    query_tile, ref_tile, word_tile,
                    d_results);
            } else {
                popcount_intersection_matrix_wwg_kernel<uint64_t, false><<<dim3(blocks_x, blocks_y),
                                                              threads_per_block,
                                                              shared_bytes>>>(
                    d_queries, d_refs_used,
                    words_per_bitset_u, words_per_bitset_u,
                    num_queries_u, num_refs_u,
                    query_tile, ref_tile, word_tile,
                    d_results);
            }
        } else {
            if (ref_layout == RefLayout::Transposed) {
                popcount_intersection_builtin_kernel<uint64_t, true><<<dim3(blocks_x, blocks_y),
                                                              threads_per_block,
                                                              shared_bytes>>>(
                    d_queries, d_refs_used,
                    words_per_bitset_u, words_per_bitset_u,
                    num_queries_u, num_refs_u,
                    query_tile, ref_tile, word_tile,
                    d_results);
            } else {
                popcount_intersection_builtin_kernel<uint64_t, false><<<dim3(blocks_x, blocks_y),
                                                              threads_per_block,
                                                              shared_bytes>>>(
                    d_queries, d_refs_used,
                    words_per_bitset_u, words_per_bitset_u,
                    num_queries_u, num_refs_u,
                    query_tile, ref_tile, word_tile,
                    d_results);
            }
        }
        GPU_CHECK(GPU_GET_LAST_ERROR());
    };

    GPU_CHECK(GPU_MEMCPY(d_queries, h_queries, queries_bytes, GPU_MEMCPY_H2D));
    launch_kernel();
    GPU_CHECK(GPU_DEVICE_SYNC());
    result.gpu_results = (uint32_t *)malloc(results_bytes);
    GPU_CHECK(GPU_MEMCPY(result.gpu_results, d_results, results_bytes, GPU_MEMCPY_D2H));
    puts("Completed burn-in iteration to warm up GPU and PCIe paths");

    std::vector<double> kernel_times;
    std::vector<double> total_times;
    std::vector<double> cpu_overhead_times;
    std::vector<uint32_t> iteration_results;
    kernel_times.reserve(iterations);
    total_times.reserve(iterations);
    cpu_overhead_times.reserve(iterations);
    iteration_results.reserve(iterations);

    for (int repeat = 0; repeat < iterations; ++repeat) {
        struct timespec total_start_ts;
        clock_gettime(CLOCK_MONOTONIC, &total_start_ts);

        GPU_CHECK(GPU_MEMCPY(d_queries, h_queries, queries_bytes, GPU_MEMCPY_H2D));
        GPU_CHECK(GPU_EVENT_RECORD(start_event, 0));
        launch_kernel();
        GPU_CHECK(GPU_EVENT_RECORD(stop_event, 0));
        GPU_CHECK(GPU_EVENT_SYNC(stop_event));
        struct timespec d2h_start_ts;
        clock_gettime(CLOCK_MONOTONIC, &d2h_start_ts);
        GPU_CHECK(GPU_MEMCPY(result.gpu_results, d_results, results_bytes, GPU_MEMCPY_D2H));
        struct timespec d2h_end_ts;
        clock_gettime(CLOCK_MONOTONIC, &d2h_end_ts);

        struct timespec cpu_scan_start_ts;
        clock_gettime(CLOCK_MONOTONIC, &cpu_scan_start_ts);
        int max_val = 0;
        int current = 0;
        const size_t nelem = num_queries * num_refs;
        for (size_t idx = 0; idx < nelem; ++idx) {
            current = (int)result.gpu_results[idx];
            if (current > max_val) {
                max_val = current;
            }
        }
        struct timespec cpu_scan_end_ts;
        clock_gettime(CLOCK_MONOTONIC, &cpu_scan_end_ts);
        iteration_results.push_back((uint32_t)max_val);
        struct timespec total_end_ts;
        clock_gettime(CLOCK_MONOTONIC, &total_end_ts);

        float kernel_ms = 0.0f;
        GPU_CHECK(GPU_EVENT_ELAPSED_TIME(&kernel_ms, start_event, stop_event));
        kernel_times.push_back(kernel_ms);

        const double total_ns = (double)(total_end_ts.tv_sec - total_start_ts.tv_sec) * 1.0e9 +
            (double)(total_end_ts.tv_nsec - total_start_ts.tv_nsec);
        const double cpu_ns = (double)(cpu_scan_end_ts.tv_sec - cpu_scan_start_ts.tv_sec) * 1.0e9 +
            (double)(cpu_scan_end_ts.tv_nsec - cpu_scan_start_ts.tv_nsec);
        const double d2h_ns = (double)(d2h_end_ts.tv_sec - d2h_start_ts.tv_sec) * 1.0e9 +
            (double)(d2h_end_ts.tv_nsec - d2h_start_ts.tv_nsec);
        const double total_ms = total_ns / 1.0e6;
        const double cpu_ms = cpu_ns / 1.0e6;
        const double d2h_ms = d2h_ns / 1.0e6;
        total_times.push_back(total_ms);
        cpu_overhead_times.push_back(cpu_ms);
        result.per_iter_d2h_ms.push_back(d2h_ms);
    }

    double kernel_sum_ms = 0.0;
    double total_sum_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        kernel_sum_ms += kernel_times[i];
        total_sum_ms += total_times[i];
    }
    const double kernel_avg_ms = kernel_sum_ms / iterations;
    const double total_avg_ms = total_sum_ms / iterations;
    
    result.per_iter_kernel_ms = std::move(kernel_times);
    result.per_iter_total_ms = std::move(total_times);
    result.per_iter_cpu_overhead_ms = std::move(cpu_overhead_times);
    result.per_iter_results = std::move(iteration_results);

    // Compute average metrics
    const double cpu_overhead_ms = total_avg_ms - kernel_avg_ms;
    const double kernel_ns = kernel_avg_ms * 1.0e6;
    const double total_ns = total_avg_ms * 1.0e6;
    const double cpu_ns = cpu_overhead_ms * 1.0e6;
    const double pairs = (double)num_queries * (double)num_refs;
    
    // Compute metrics
    
    
    result.num_pairs = num_queries * num_refs;
    result.kernel_time_ns = kernel_ns;
    result.total_time_ns = total_ns;
    result.cpu_overhead_ns = cpu_ns;
    result.compute_gbps = payload_per_iter_gb / (kernel_ns / 1e9);
    result.total_gbps = payload_per_iter_gb / (total_ns / 1e9);
    
    // Verify results
    result.checksum = 0;
    result.agreements = 0;
    result.disagreements = 0;
    result.max_result = 0;
    
    for (size_t idx = 0; idx < num_queries * num_refs; ++idx) {
        result.checksum += result.gpu_results[idx];
        if (result.gpu_results[idx] > result.max_result) {
            result.max_result = result.gpu_results[idx];
        }
        if (cpu_results) {
            if (result.gpu_results[idx] == cpu_results[idx]) {
                result.agreements++;
            } else {
                result.disagreements++;
            }
        }
    }
    
    // Cleanup
    GPU_CHECK(GPU_EVENT_DESTROY(start_event));
    GPU_CHECK(GPU_EVENT_DESTROY(stop_event));
    GPU_CHECK(GPU_FREE(d_results));
    GPU_CHECK(GPU_FREE(d_refs));
    if (d_refs_transposed) {
        GPU_CHECK(GPU_FREE(d_refs_transposed));
    }
    GPU_CHECK(GPU_FREE(d_queries));
    
    return result;
}
