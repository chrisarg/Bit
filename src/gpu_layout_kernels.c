/* ==========================================================================
   SECTION 1: INCLUDES
   Standard library headers first, then project headers.
   ========================================================================== */

#include <omp.h>
#include <stdlib.h>
#include <stddef.h>
#include "gpu_layout_kernels.h"

/* --- End Section 1: INCLUDES --- */

/* ==========================================================================
   SECTION 2: COMPILE-TIME CONFIGURATION
   File-local compile-time constants, build-time defaults, and feature flags.
   ========================================================================== */

/* No compile-time configuration required here. */

/* --- End Section 2: COMPILE-TIME CONFIGURATION --- */

/* ==========================================================================
   SECTION 3: INFRASTRUCTURAL MACROS
   Private macros used by helper functions and low-level operations.
   ========================================================================== */

/* No additional macros required at this time. */

/* --- End Section 3: INFRASTRUCTURAL MACROS --- */

/* ==========================================================================
   SECTION 4: PRIVATE IMPLEMENTATION MACROS
   Reserved for file-local operational macros.
   ========================================================================== */
#define GPU_KERNEL_ALLOC(dev, count) omp_target_alloc((count) * sizeof(uint64_t), (dev))
/* --- End Section 4: PRIVATE IMPLEMENTATION MACROS --- */

/* ==========================================================================
   SECTION 5: INTERNAL DATA STRUCTURES AND ENUMS
   Types used only inside this translation unit.
   ========================================================================== */

/* No internal data structures are required for this module. */

/* --- End Section 5: INTERNAL DATA STRUCTURES AND ENUMS --- */

/* ==========================================================================
   SECTION 6: STATIC DATA
   File-scope constants and persistent state.
   ========================================================================== */

/* No static data is required for this module. */

/* --- End Section 6: STATIC DATA --- */

/* ==========================================================================
   SECTION 7: INTERNAL FUNCTION FORWARD DECLARATIONS
   Static helper prototypes for this translation unit.
   ========================================================================== */

static void _GPU_transpose_kernel(uint64_t *bits, size_t rows,
                                  size_t columns, int device_id);

/* --- End Section 7: INTERNAL FUNCTION FORWARD DECLARATIONS --- */

/* ==========================================================================
   SECTION 8: INTERNAL HELPER FUNCTION DEFINITIONS
   Low-level helpers and device kernels.
   ========================================================================== */
static void _GPU_transpose_kernel(uint64_t *bits, size_t rows,
                                  size_t columns, int device_id) {
    size_t t_elements = rows * columns;
    uint64_t *bits_T = (uint64_t *)GPU_KERNEL_ALLOC(device_id, t_elements);
    if (!bits_T) {
        return;
    }

#pragma omp target teams distribute parallel for collapse(2) device(device_id)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < columns; j++) {
            bits_T[j * rows + i] = bits[i * columns + j];
        }
    }

#pragma omp target teams distribute parallel for device(device_id) \
    is_device_ptr(bits_T)
    for (size_t idx = 0; idx < t_elements; idx++) {
        bits[idx] = bits_T[idx];
    }

    omp_target_free(bits_T, device_id);
}

/* --- End Section 8: INTERNAL HELPER FUNCTION DEFINITIONS --- */

/* ==========================================================================
   SECTION 9: PUBLIC API
   GPU and CPU region transformation entry points.
   ========================================================================== */

void GPU_region_transpose(GPUDataState from, GPUDataState to, uint64_t *bits,
                          size_t rows, size_t columns, int device_id,
                          void *params, size_t params_size) {
    (void)from;
    (void)to;
    (void)params;
    (void)params_size;

    _GPU_transpose_kernel(bits, rows, columns, device_id);
}

void cpu_universal_transpose(GPUDataState from, GPUDataState to,
                             uint64_t *bits, size_t rows, size_t columns,
                             int device_id, void *params, size_t params_size) {
    (void)from;
    (void)to;
    (void)params;
    (void)params_size;

    _GPU_transpose_kernel(bits, rows, columns, device_id);
}

/* --- End Section 9: PUBLIC API --- */
