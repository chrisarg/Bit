#pragma once

#include <stdint.h>
#include <stddef.h>

/* Unified API. The Makefile determines if this links to the CPU or GPU object */
void topk_int_omp(const int *dist, int64_t N, int64_t M, int K,
                     int *out_dist, int *out_idx, int dev_id);

