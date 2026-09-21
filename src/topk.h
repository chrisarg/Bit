#pragma once

#include <stdint.h>
#include <stddef.h>

/* Private top-k selection API. Not part of the public bit.h interface; it is
 * compiled into libbit and exists to serve the FAISS-comparison benchmarks
 * and a future filtered-results library function.
 *
 * Host/device dispatch is resolved at BUILD TIME by the caller choosing the
 * explicitly suffixed entry point (mirroring the gpu_layout_kernels
 * convention): CPU executables call topk_int_omp_cpu, GPU executables call
 * topk_int_omp_gpu. */
void topk_int_omp_cpu(const uint64_t *dist, size_t N, size_t M, size_t K,
                      uint64_t *out_dist, size_t *out_idx, int dev_id);
void topk_int_omp_gpu(const uint64_t *dist, size_t N, size_t M, size_t K,
                      uint64_t *out_dist, size_t *out_idx, int dev_id);

/* Unified name reserved for the future internal filtered-results wrapper. */
void topk_int_omp(const uint64_t *dist, size_t N, size_t M, size_t K,
                  uint64_t *out_dist, size_t *out_idx, int dev_id);

