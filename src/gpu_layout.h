#pragma once

#include <stddef.h>
#include <stdint.h>

#define MASK_BASE_LAYOUT 0x000000FFu
#define MASK_FLAGS 0xFFFFFF00u
#define MAKE_STATE(layout, flags) ((uint32_t)(layout) | (uint32_t)(flags))

typedef enum {
    LAYOUT_ROW_MAJOR = 0,
    LAYOUT_COL_MAJOR = 1,
    LAYOUT_Z_CURVE = 2,
} GPUDataState;

typedef enum {
    FLAG_NONE = 0u,
    FLAG_TRANSFORM = 1u,
} GPUDataFlags;

typedef void (*TransitionKernel)(
    GPUDataState from,
    GPUDataState to,
    uint64_t *bits,
    size_t rows,
    size_t columns,
    int device_id,
    void *transform_params,
    size_t params_size
);
