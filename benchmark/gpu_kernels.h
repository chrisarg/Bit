#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define GPU_KERNEL __global__
typedef cudaStream_t gpu_stream_t;
#elif defined(__HIP__)
#include <hip/hip_runtime.h>
#define GPU_KERNEL __global__
typedef hipStream_t gpu_stream_t;
#else
#error "GPU_KERNELS_H requires CUDA or HIP"
#endif

#include <stdint.h>
#include <stddef.h>

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

static __device__ __forceinline__ unsigned int count_WWG_gpu(unsigned long long x) {
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
    return count_WWG_gpu(static_cast<unsigned long long>(x));
#endif
}

static __device__ __forceinline__ unsigned int popcount_uint64(uint64_t x) {
#ifdef USE_BUILTIN_POPCOUNT
    return static_cast<unsigned int>(__builtin_popcountll(static_cast<unsigned long long>(x)));
#else
    return count_WWG_gpu(static_cast<unsigned long long>(x));
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
