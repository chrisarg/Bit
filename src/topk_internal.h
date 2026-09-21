#pragma once

#include "omp.h"
#include <stdint.h>

/* The heap helpers must be visible to the device compilation pass whenever
 * offloading is enabled (i.e. whenever NOGPU is not defined), because
 * topk_gpu.c invokes them from `omp target` regions. */
#ifndef NOGPU
#pragma omp declare target
#endif

// Max Heap : finds the top smallest candidates (min values)
static inline void sift_down(uint64_t *dist_base, size_t *idx_base,
                             size_t start, size_t n, size_t tid,
                             size_t stride) {
  size_t root = start;
  while (2 * root + 1 < n) {
    size_t child = 2 * root + 1;
    size_t c1_off = child * stride + tid;
    size_t c2_off = (child + 1) * stride + tid;

    if (child + 1 < n && dist_base[c1_off] < dist_base[c2_off]) {
      child++;
      c1_off = c2_off;
    }

    size_t root_off = root * stride + tid;
    if (dist_base[root_off] >= dist_base[c1_off])
      break;

    uint64_t td = dist_base[root_off];
    dist_base[root_off] = dist_base[c1_off];
    dist_base[c1_off] = td;

    size_t ti = idx_base[root_off];
    idx_base[root_off] = idx_base[c1_off];
    idx_base[c1_off] = ti;

    root = child;
  }
}

// Min Heap : finds the top largest candidates (max values)
static inline void sift_down_min(uint64_t *dist_base, size_t *idx_base,
                                 size_t start, size_t n, size_t tid,
                                 size_t stride) {
  size_t root = start;
  while (2 * root + 1 < n) {
    size_t child = 2 * root + 1;
    size_t c1_off = child * stride + tid;
    size_t c2_off = (child + 1) * stride + tid;

    // For a min-heap, we want to find the SMALLER of the two children.
    // Notice the '>' operator here compared to the max-heap's '<'
    if (child + 1 < n && dist_base[c1_off] > dist_base[c2_off]) {
      child++;
      c1_off = c2_off;
    }

    size_t root_off = root * stride + tid;
    
    // If the root is already smaller than or equal to the smallest child, the heap property is satisfied.
    if (dist_base[root_off] <= dist_base[c1_off])
      break;

    // Swap distances
    uint64_t td = dist_base[root_off];
    dist_base[root_off] = dist_base[c1_off];
    dist_base[c1_off] = td;

    // Swap indices
    size_t ti = idx_base[root_off];
    idx_base[root_off] = idx_base[c1_off];
    idx_base[c1_off] = ti;

    root = child;
  }
}

static inline void heapify(uint64_t *dist_base, size_t *idx_base, size_t n,
                           size_t tid, size_t stride) {
  for (size_t i = n / 2; i-- > 0;) {
    sift_down(dist_base, idx_base, i, n, tid, stride);
  }
}

#ifndef NOGPU
#pragma omp end declare target
#endif
