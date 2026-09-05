#include "topk.h"
#include "topk_internal.h"
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NOGPU

#if defined(__clang__) || defined(__INTEL_LLVM_COMPILER) || defined(__llvm__)
void topk_int_omp_gpu(const int *dist, int64_t N, int64_t M, int K,
                      int *out_dist, int *out_idx, int dev_id) {
  if (N <= 0 || M <= 0 || K <= 0)
    return;
  if (K > M)
    K = M;

  // Map the output buffers directly to the device to use as in-place heaps.
#pragma omp target data map(from : out_dist[0 : N * K], out_idx[0 : N * K]) \
    device(dev_id)
  {
    // UNIFIED OPTIMIZATION:
    // Force exactly 1 team per row, and 1 thread per team.
    // This entirely eliminates intra-warp divergence and allows us to 
    // work directly in the output buffers without malloc/free.
#pragma omp target teams distribute num_teams(N) thread_limit(1) \
    is_device_ptr(dist) device(dev_id)
    for (int64_t row = 0; row < N; ++row) {
      
      // Because thread_limit is 1, this executes sequentially by the master 
      // thread of the team. No inner parallel region is needed.
      int *my_heap_d = out_dist + row * K;
      int *my_heap_i = out_idx + row * K;
      const int *row_dist = dist + row * M;

      // --- PHASE 1: Initialize the max-heap ---
      for (int k = 0; k < K; ++k) {
        my_heap_d[k] = row_dist[k];
        my_heap_i[k] = k;
      }

      // Local, linear heapify (tid = 0, stride = 1)
      heapify(my_heap_d, my_heap_i, K, 0, 1);

      // --- PHASE 2: Stream remaining M elements ---
      for (int64_t j = K; j < M; ++j) {
        int d = row_dist[j];
        if (d < my_heap_d[0]) {
          my_heap_d[0] = d;
          my_heap_i[0] = (int)j;
          sift_down(my_heap_d, my_heap_i, 0, K, 0, 1);
        }
      }

      // --- PHASE 3: Convert max-heap to sorted ascending in-place ---
      for (int i = K - 1; i > 0; --i) {
        int td = my_heap_d[0];
        my_heap_d[0] = my_heap_d[i];
        my_heap_d[i] = td;

        int ti = my_heap_i[0];
        my_heap_i[0] = my_heap_i[i];
        my_heap_i[i] = ti;

        sift_down(my_heap_d, my_heap_i, 0, i, 0, 1);
      }
    }
  }
}

#else
void topk_int_omp_gpu(const int *dist, int64_t N, int64_t M, int K,
                      int *out_dist, int *out_idx, int dev_id) {
  if (N <= 0 || M <= 0 || K <= 0)
    return;
  if (K > M)
    K = M;

  // We map out_dist and out_idx to the device. 'from' allocates uninitialized 
  // memory on the GPU, which we use directly as our in-place working heap.
#pragma omp target data map(from : out_dist[0 : N * K], out_idx[0 : N * K])    \
    device(dev_id)
  {
    // Flatten the parallel execution: 1 Thread = 1 Row.
    // The runtime handles grouping these threads into optimal warps/teams.
#pragma omp target teams distribute parallel for is_device_ptr(dist) device(dev_id)
    for (int64_t row = 0; row < N; ++row) {
      
      // Each thread works directly in the output buffer
      int *my_heap_d = out_dist + row * K;
      int *my_heap_i = out_idx + row * K;
      const int *row_dist = dist + row * M;

      // --- PHASE 1: Initialize the max-heap ---
      for (int k = 0; k < K; ++k) {
        my_heap_d[k] = row_dist[k];
        my_heap_i[k] = k;
      }

      // Local, linear heapify (tid = 0, stride = 1)
      heapify(my_heap_d, my_heap_i, K, 0, 1);

      // --- PHASE 2: Stream remaining M elements ---
      for (int64_t j = K; j < M; ++j) {
        int d = row_dist[j];
        if (d < my_heap_d[0]) {
          my_heap_d[0] = d;
          my_heap_i[0] = (int)j;
          sift_down(my_heap_d, my_heap_i, 0, K, 0, 1);
        }
      }

      // --- PHASE 3: Convert max-heap to sorted ascending in-place ---
      for (int i = K - 1; i > 0; --i) {
        int td = my_heap_d[0];
        my_heap_d[0] = my_heap_d[i];
        my_heap_d[i] = td;

        int ti = my_heap_i[0];
        my_heap_i[0] = my_heap_i[i];
        my_heap_i[i] = ti;

        sift_down(my_heap_d, my_heap_i, 0, i, 0, 1);
      }
    }
  }
}
#endif

#endif /* NOGPU */