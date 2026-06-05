#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define GPU_KERNEL __global__
typedef cudaStream_t gpu_stream_t;
typedef cudaEvent_t gpu_event_t;
#ifndef GPU_EVENT_CREATE
#define GPU_EVENT_CREATE(event) cudaEventCreate(event)
#endif
#ifndef GPU_EVENT_DESTROY
#define GPU_EVENT_DESTROY(event) cudaEventDestroy(event)
#endif
#ifndef GPU_EVENT_RECORD
#define GPU_EVENT_RECORD(event, stream) cudaEventRecord(event, stream)
#endif
#ifndef GPU_EVENT_SYNC
#define GPU_EVENT_SYNC(event) cudaEventSynchronize(event)
#endif
#ifndef GPU_EVENT_ELAPSED_TIME
#define GPU_EVENT_ELAPSED_TIME(ms, start, stop) cudaEventElapsedTime(ms, start, stop)
#endif
#ifndef GPU_GET_LAST_ERROR
#define GPU_GET_LAST_ERROR cudaGetLastError()
#endif
#ifndef GPU_DEVICE_SYNC
#define GPU_DEVICE_SYNC cudaDeviceSynchronize()
#endif
#ifndef GPU_SET_DEVICE
#define GPU_SET_DEVICE cudaSetDevice
#endif
#ifndef GPU_LAUNCH_KERNEL
#define GPU_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif
#elif defined(__HIP__)
#include <hip/hip_runtime.h>
#define GPU_KERNEL __global__
typedef hipStream_t gpu_stream_t;
typedef hipEvent_t gpu_event_t;
#ifndef GPU_EVENT_CREATE
#define GPU_EVENT_CREATE(event) hipEventCreate(event)
#endif
#ifndef GPU_EVENT_DESTROY
#define GPU_EVENT_DESTROY(event) hipEventDestroy(event)
#endif
#ifndef GPU_EVENT_RECORD
#define GPU_EVENT_RECORD(event, stream) hipEventRecord(event, stream)
#endif
#ifndef GPU_EVENT_SYNC
#define GPU_EVENT_SYNC(event) hipEventSynchronize(event)
#endif
#ifndef GPU_EVENT_ELAPSED_TIME
#define GPU_EVENT_ELAPSED_TIME(ms, start, stop) hipEventElapsedTime(ms, start, stop)
#endif
#ifndef GPU_GET_LAST_ERROR
#define GPU_GET_LAST_ERROR hipGetLastError()
#endif
#ifndef GPU_DEVICE_SYNC
#define GPU_DEVICE_SYNC hipDeviceSynchronize()
#endif
#ifndef GPU_SET_DEVICE
#define GPU_SET_DEVICE hipSetDevice
#endif
#ifndef GPU_LAUNCH_KERNEL
#define GPU_LAUNCH_KERNEL(kernel, grid, block, shared, stream, ...) \
    hipLaunchKernelGGL(kernel, grid, block, shared, stream, __VA_ARGS__)
#endif
#else
#error "GPU_KERNELS_H requires CUDA or HIP"
#endif

#include <stdint.h>
#include <stddef.h>

#ifndef TILE_J
#define TILE_J 2048
#endif

#ifndef ILP
#define ILP 16
#endif

// ===========================================================================
// All-pairs (matrix) popcount intersection kernels
// Each thread block processes a tile of query/ref pairs.
// The tile is broken into word chunks and query words are reused in shared memory.
// ===========================================================================

// Match OpenMP benchmark WWG popcount logic exactly (no builtin popcount).
#define C1_WWG_GPU 0x5555555555555555ULL
#define C2_WWG_GPU 0x3333333333333333ULL
#define C3_WWG_GPU 0x0F0F0F0F0F0F0F0FULL
#define C4_WWG_GPU 0x0101010101010101ULL

static __device__ __forceinline__ unsigned int count_WWG_gpu_impl(unsigned long long x) {
    x -= (x >> 1) & C1_WWG_GPU;
    x = ((x >> 2) & C2_WWG_GPU) + (x & C2_WWG_GPU);
    x = (x + (x >> 4)) & C3_WWG_GPU;
    x *= C4_WWG_GPU;
    return static_cast<unsigned int>(x >> 56);
}
static __device__ __forceinline__ unsigned int popcount_uint32(uint32_t x) {
#ifdef USE_BUILTIN_POPCOUNT
    return static_cast<unsigned int>(__builtin_popcount(x));
#else
    return count_WWG_gpu_impl(static_cast<unsigned long long>(x));
#endif
}
static __device__ __forceinline__ unsigned int popcount_uint64(uint64_t x) {
#ifdef USE_BUILTIN_POPCOUNT
    return static_cast<unsigned int>(__builtin_popcountll(static_cast<unsigned long long>(x)));
#else
    return count_WWG_gpu_impl(static_cast<unsigned long long>(x));
#endif
}

static __device__ __forceinline__ unsigned int popcount_pair_word(
    uint32_t lhs, uint32_t rhs)
{
    return popcount_uint32(lhs & rhs);
}

static __device__ __forceinline__ unsigned int popcount_pair_word(
    uint64_t lhs, uint64_t rhs)
{
    return popcount_uint64(lhs & rhs);
}

template <typename T>
GPU_KERNEL void compute_setop_popcount_coarsened_kernel(
    const T *bit_qwords,
    const T *bits_qwords_T,
    int *counts,
    int K,
    int N,
    int J)
{
    extern __shared__ unsigned char s_query_raw[];
    T *s_query = reinterpret_cast<T *>(s_query_raw);

    const int cols_per_block = blockDim.x * ILP;
    const int blocks_per_row = (N + cols_per_block - 1) / cols_per_block;
    const int total_jobs = K * blocks_per_row;
    const int job = blockIdx.x;

    for (int job_idx = job; job_idx < total_jobs; job_idx += gridDim.x) {
        const int k = job_idx / blocks_per_row;
        const int i_base = (job_idx % blocks_per_row) * cols_per_block;
        int i[ILP];
        int sum[ILP] = {0};

        #pragma unroll
        for (int u = 0; u < ILP; ++u) {
            i[u] = i_base + threadIdx.x + u * blockDim.x;
        }

        for (int j_tile = 0; j_tile < J; j_tile += TILE_J) {
            const int current_tile = ((j_tile + TILE_J) > J) ? (J - j_tile) : TILE_J;
            for (int t = threadIdx.x; t < current_tile; t += blockDim.x) {
                s_query[t] = bit_qwords[k * J + j_tile + t];
            }
            __syncthreads();

            const bool fast_path = (i_base + cols_per_block <= N);
            if (fast_path) {
                for (int j = 0; j < current_tile; ++j) {
                    const T sk = s_query[j];
                    #pragma unroll
                    for (int u = 0; u < ILP; ++u) {
                        sum[u] += popcount_pair_word(sk,
                                                     bits_qwords_T[(j_tile + j) * N + i[u]]);
                    }
                }
            } else {
                for (int j = 0; j < current_tile; ++j) {
                    const T sk = s_query[j];
                    #pragma unroll
                    for (int u = 0; u < ILP; ++u) {
                        if (i[u] < N) {
                            sum[u] += popcount_pair_word(sk,
                                                         bits_qwords_T[(j_tile + j) * N + i[u]]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int u = 0; u < ILP; ++u) {
            if (i[u] < N) {
                counts[k * N + i[u]] = sum[u];
            }
        }
    }
}

template <typename T>
static inline void launch_setop_coarsened(
    const T *d_bit_qwords,
    const T *d_bits_qwords_T,
    int *d_counts,
    int K,
    int N,
    int J)
{
    const dim3 blockDim(256);
    const int cols_per_block = blockDim.x * ILP;
    const int blocks_per_row = (N + cols_per_block - 1) / cols_per_block;
    const int total_jobs = K * blocks_per_row;
    const int max_physical_blocks = 65536;
    const dim3 gridDim(total_jobs < max_physical_blocks ? total_jobs : max_physical_blocks);
    const size_t shared_mem_bytes = static_cast<size_t>(TILE_J) * sizeof(T);

    gpu_event_t start, stop;
    GPU_CHECK(GPU_EVENT_CREATE(&start));
    GPU_CHECK(GPU_EVENT_CREATE(&stop));
    GPU_CHECK(GPU_EVENT_RECORD(start, 0));

    GPU_LAUNCH_KERNEL(compute_setop_popcount_coarsened_kernel<T>,
                      gridDim,
                      blockDim,
                      shared_mem_bytes,
                      0,
                      d_bit_qwords,
                      d_bits_qwords_T,
                      d_counts,
                      K,
                      N,
                      J);
    GPU_CHECK(GPU_GET_LAST_ERROR);

    GPU_CHECK(GPU_EVENT_RECORD(stop, 0));
    GPU_EVENT_SYNC(stop);
    float milliseconds = 0.0f;
    GPU_EVENT_ELAPSED_TIME(&milliseconds, start, stop);

    GPU_EVENT_DESTROY(start);
    GPU_EVENT_DESTROY(stop);
}

template <typename T>
GPU_KERNEL void popcount_intersection_matrix_wwg_kernel(
    const T *queries,
    const T *refs,
    unsigned int words_per_query,
    unsigned int words_per_ref,
    unsigned int num_queries,
    unsigned int num_refs,
    unsigned int query_tile,
    unsigned int ref_tile,
    unsigned int word_tile,
    uint32_t *results)
{
    extern __shared__ unsigned char q_shared_raw[];
    T *q_shared = reinterpret_cast<T *>(q_shared_raw);
    const unsigned int tid = threadIdx.x;
    const unsigned int block_q = blockIdx.x * query_tile;
    const unsigned int block_r = blockIdx.y * ref_tile;
    if (block_q >= num_queries || block_r >= num_refs) return;

    const unsigned int q_limit = ((block_q + query_tile) <= num_queries)
        ? query_tile
        : (num_queries - block_q);
    const unsigned int r_limit = ((block_r + ref_tile) <= num_refs)
        ? ref_tile
        : (num_refs - block_r);
    const unsigned int pair_count = q_limit * r_limit;
    const unsigned int pair_stride = blockDim.x;

    for (unsigned int pair = tid; pair < pair_count; pair += pair_stride) {
        const unsigned int local_r = pair / q_limit;
        const unsigned int local_q = pair % q_limit;
        const unsigned int query_idx = block_q + local_q;
        const unsigned int ref_idx = block_r + local_r;
        uint32_t count = 0;

        for (unsigned int word_start = 0; word_start < words_per_query;
             word_start += word_tile) {
            const unsigned int current_tile =
                ((words_per_query - word_start) < word_tile)
                    ? (words_per_query - word_start)
                    : word_tile;
            const unsigned int load_count = q_limit * current_tile;
            for (unsigned int idx = tid; idx < load_count; idx += pair_stride) {
                const unsigned int q_local = idx / current_tile;
                const unsigned int w_local = idx % current_tile;
                const T *q_base = queries + (block_q + q_local) * words_per_query;
                q_shared[q_local * current_tile + w_local] = q_base[word_start + w_local];
            }
            __syncthreads();

            const T *q_tile = q_shared + local_q * current_tile;
            const T *r_data = refs + ref_idx * words_per_ref + word_start;
            const T *r_tile = r_data;
            for (unsigned int w = 0; w < current_tile; ++w) {
                count += popcount_pair_word(q_tile[w], r_tile[w]);
            }
            __syncthreads();
        }
        results[query_idx * num_refs + ref_idx] = count;
    }
}

template <typename T>
GPU_KERNEL void popcount_intersection_builtin_kernel(
    const T *queries,
    const T *refs,
    unsigned int words_per_query,
    unsigned int words_per_ref,
    unsigned int num_queries,
    unsigned int num_refs,
    unsigned int query_tile,
    unsigned int ref_tile,
    unsigned int word_tile,
    uint32_t *results)
{
    extern __shared__ unsigned char q_shared_raw[];
    T *q_shared = reinterpret_cast<T *>(q_shared_raw);
    const unsigned int tid = threadIdx.x;
    const unsigned int block_q = blockIdx.x * query_tile;
    const unsigned int block_r = blockIdx.y * ref_tile;
    if (block_q >= num_queries || block_r >= num_refs) return;

    const unsigned int q_limit = ((block_q + query_tile) <= num_queries)
        ? query_tile
        : (num_queries - block_q);
    const unsigned int r_limit = ((block_r + ref_tile) <= num_refs)
        ? ref_tile
        : (num_refs - block_r);
    const unsigned int pair_count = q_limit * r_limit;
    const unsigned int pair_stride = blockDim.x;

    for (unsigned int pair = tid; pair < pair_count; pair += pair_stride) {
        const unsigned int local_r = pair / q_limit;
        const unsigned int local_q = pair % q_limit;
        const unsigned int query_idx = block_q + local_q;
        const unsigned int ref_idx = block_r + local_r;
        uint32_t count = 0;

        for (unsigned int word_start = 0; word_start < words_per_query;
             word_start += word_tile) {
            const unsigned int current_tile =
                ((words_per_query - word_start) < word_tile)
                    ? (words_per_query - word_start)
                    : word_tile;
            const unsigned int load_count = q_limit * current_tile;
            for (unsigned int idx = tid; idx < load_count; idx += pair_stride) {
                const unsigned int q_local = idx / current_tile;
                const unsigned int w_local = idx % current_tile;
                const T *q_base = queries + (block_q + q_local) * words_per_query;
                q_shared[q_local * current_tile + w_local] = q_base[word_start + w_local];
            }
            __syncthreads();

            const T *q_tile = q_shared + local_q * current_tile;
            const T *r_data = refs + ref_idx * words_per_ref + word_start;
            const T *r_tile = r_data;
            for (unsigned int w = 0; w < current_tile; ++w) {
                count += popcount_pair_word(q_tile[w], r_tile[w]);
            }
            __syncthreads();
        }
        results[query_idx * num_refs + ref_idx] = count;
    }
}
