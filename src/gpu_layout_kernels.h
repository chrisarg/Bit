#pragma once

#include "gpu_layout.h"
#include <stddef.h>

void GPU_region_transpose(GPUDataState from, GPUDataState to, uint64_t *bits,
                          size_t rows, size_t columns, int device_id,
                          void *params, size_t params_size);

void cpu_universal_transpose(GPUDataState from, GPUDataState to, uint64_t *bits,
                             size_t rows, size_t columns, int device_id,
                             void *params, size_t params_size);
