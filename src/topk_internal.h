#pragma once

#include "omp.h"
#include <stdint.h>

#if GPU_COMPILE_TOPK
#pragma omp declare target
#endif

// Max Heap : finds the top smallest candidates (min values)
static inline void sift_down(int *dist_base, int *idx_base,
                                       int64_t start, int64_t n, int tid,
                                       int stride) {
  int64_t root = start;
  while (2 * root + 1 < n) {
    int64_t child = 2 * root + 1;
    int64_t c1_off = child * stride + tid;
    int64_t c2_off = (child + 1) * stride + tid;

    if (child + 1 < n && dist_base[c1_off] < dist_base[c2_off]) {
      child++;
      c1_off = c2_off;
    }

    int64_t root_off = root * stride + tid;
    if (dist_base[root_off] >= dist_base[c1_off])
      break;

    int td = dist_base[root_off];
    dist_base[root_off] = dist_base[c1_off];
    dist_base[c1_off] = td;

    int ti = idx_base[root_off];
    idx_base[root_off] = idx_base[c1_off];
    idx_base[c1_off] = ti;

    root = child;
  }
}

// Min Heap : finds the top largest candidates (max values)
static inline void sift_down_min(int *dist_base, int *idx_base,
                                           int64_t start, int64_t n, int tid,
                                           int stride) {
  int64_t root = start;
  while (2 * root + 1 < n) {
    int64_t child = 2 * root + 1;
    int64_t c1_off = child * stride + tid;
    int64_t c2_off = (child + 1) * stride + tid;

    // For a min-heap, we want to find the SMALLER of the two children.
    // Notice the '>' operator here compared to the max-heap's '<'
    if (child + 1 < n && dist_base[c1_off] > dist_base[c2_off]) {
      child++;
      c1_off = c2_off;
    }

    int64_t root_off = root * stride + tid;
    
    // If the root is already smaller than or equal to the smallest child, the heap property is satisfied.
    if (dist_base[root_off] <= dist_base[c1_off])
      break;

    // Swap distances
    int td = dist_base[root_off];
    dist_base[root_off] = dist_base[c1_off];
    dist_base[c1_off] = td;

    // Swap indices
    int ti = idx_base[root_off];
    idx_base[root_off] = idx_base[c1_off];
    idx_base[c1_off] = ti;

    root = child;
  }
}

static inline void heapify(int *dist_base, int *idx_base, int64_t n,
                                     int tid, int stride) {
  for (int64_t i = (n - 2) / 2; i >= 0; --i) {
    sift_down(dist_base, idx_base, i, n, tid, stride);
  }
}

#if GPU_COMPILE_TOPK
#pragma omp end declare target
#endif
