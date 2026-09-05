#include "topk.h"
#include "topk_internal.h"
#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void topk_int_omp_cpu(const int *dist, int64_t N, int64_t M, int K,
                      int *out_dist, int *out_idx, int dev_id) 
{
  (void)dev_id; // host implementation: device id is ignored
    if (N <= 0 || M <= 0 || K <= 0) return;
    if (K > M) K = M;

    // guided schedule balances load if some rows trigger more heap-swaps than others
    #pragma omp parallel for schedule(guided)
    for (int64_t row = 0; row < N; ++row) {
        
        // OPTIMIZATION: Write directly to the output buffers. 
        // Each thread owns its 'row', eliminating race conditions and the need for malloc/free.
        int *my_heap_d = out_dist + (row * K);
        int *my_heap_i = out_idx  + (row * K);
        const int *row_ptr = dist + (row * M);

        // 1. Initialize the max-heap with the first K elements
        for (int k = 0; k < K; ++k) {
            my_heap_d[k] = row_ptr[k];
            my_heap_i[k] = k;
        }
        
        // Use tid = 0 and stride = 1 for cache-line friendly contiguous memory access
        heapify(my_heap_d, my_heap_i, K, 0, 1);

        // 2. Stream the rest of the M dimension (CPU prefetcher handles this perfectly)
        for (int64_t j = K; j < M; ++j) {
            int d = row_ptr[j];
            // Branch prediction favors skipping the swap for large M
            if (d < my_heap_d[0]) {
                my_heap_d[0] = d;
                my_heap_i[0] = (int)j; 
                sift_down(my_heap_d, my_heap_i, 0, K, 0, 1);
            }
        }

        // 3. Convert max-heap to sorted ascending in-place
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