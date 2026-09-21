#include "topk.h"
#include "topk_internal.h"
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void topk_int_omp_cpu(const uint64_t *dist, size_t N, size_t M, size_t K,
                      uint64_t *out_dist, size_t *out_idx, int dev_id)
{
  (void)dev_id; // host implementation: device id is ignored
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (K > M) K = M;

    // guided schedule balances load if some rows trigger more heap-swaps than others
    #pragma omp parallel for schedule(guided)
    for (size_t row = 0; row < N; ++row) {
        
        // OPTIMIZATION: Write directly to the output buffers. 
        // Each thread owns its 'row', eliminating race conditions and the need for malloc/free.
            uint64_t *my_heap_d = out_dist + row * K;
            size_t *my_heap_i = out_idx + row * K;
            const uint64_t *row_ptr = dist + row * M;

        // 1. Initialize the max-heap with the first K elements
        for (size_t k = 0; k < K; ++k) {
            my_heap_d[k] = row_ptr[k];
            my_heap_i[k] = k;
        }
        
        // Use tid = 0 and stride = 1 for cache-line friendly contiguous memory access
        heapify(my_heap_d, my_heap_i, K, 0, 1);

        // 2. Stream the rest of the M dimension (CPU prefetcher handles this perfectly)
        for (size_t j = K; j < M; ++j) {
            uint64_t d = row_ptr[j];
            // Branch prediction favors skipping the swap for large M
            if (d < my_heap_d[0]) {
                my_heap_d[0] = d;
                my_heap_i[0] = j;
                sift_down(my_heap_d, my_heap_i, 0, K, 0, 1);
            }
        }

        // 3. Convert max-heap to sorted ascending in-place
        for (size_t i = K - 1; i > 0; --i) {
            uint64_t td = my_heap_d[0];
            my_heap_d[0] = my_heap_d[i]; 
            my_heap_d[i] = td;
            
            size_t ti = my_heap_i[0];
            my_heap_i[0] = my_heap_i[i]; 
            my_heap_i[i] = ti;
            
            sift_down(my_heap_d, my_heap_i, 0, i, 0, 1);
        }
    }
}