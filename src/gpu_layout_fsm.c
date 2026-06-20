/* ==========================================================================
   SECTION 1: INCLUDES
   Standard library headers first, then project headers.
   ========================================================================== */

#include <stdlib.h>
#include <stdio.h>
#include "gpu_layout_fsm.h"
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

/* No private implementation macros required here. */

/* --- End Section 4: PRIVATE IMPLEMENTATION MACROS --- */

/* ==========================================================================
   SECTION 5: INTERNAL DATA STRUCTURES AND ENUMS
   Types used only inside this translation unit.
   ========================================================================== */

typedef struct {
    GPUDataState from;
    GPUDataState to;
    TransitionKernel func;
} TransitionEntry;

/* --- End Section 5: INTERNAL DATA STRUCTURES AND ENUMS --- */

/* ==========================================================================
   SECTION 6: STATIC DATA
   File-scope constants and persistent state.
   ========================================================================== */

static const TransitionEntry transition_table[] = {
    {LAYOUT_ROW_MAJOR, LAYOUT_COL_MAJOR, GPU_region_transpose},
    {LAYOUT_COL_MAJOR, LAYOUT_ROW_MAJOR, GPU_region_transpose},
};

#define TRANSITION_COUNT (sizeof(transition_table) / sizeof(transition_table[0]))

/* --- End Section 6: STATIC DATA --- */

/* ==========================================================================
   SECTION 7: INTERNAL FUNCTION FORWARD DECLARATIONS
   Static helper prototypes for this translation unit.
   ========================================================================== */

static TransitionKernel lookup_transition(GPUDataState from, GPUDataState to);

/* --- End Section 7: INTERNAL FUNCTION FORWARD DECLARATIONS --- */

/* ==========================================================================
   SECTION 8: INTERNAL HELPER FUNCTION DEFINITIONS
   Low-level helpers for transition routing.
   ========================================================================== */

static TransitionKernel lookup_transition(GPUDataState from, GPUDataState to) {
    for (size_t i = 0; i < TRANSITION_COUNT; i++) {
        if (transition_table[i].from == from && transition_table[i].to == to) {
            return transition_table[i].func;
        }
    }
    return NULL;
}

/* --- End Section 8: INTERNAL HELPER FUNCTION DEFINITIONS --- */

/* ==========================================================================
   SECTION 9: PUBLIC API
   Transition router entry points.
   ========================================================================== */

TransitionKernel get_transition_kernel(GPUDataState from, GPUDataState to) {
    TransitionKernel func = lookup_transition(from, to);
    if (!func) {
        fprintf(stderr, "Fatal: No transition kernel defined for %d -> %d\n", from, to);
        exit(EXIT_FAILURE);
    }
    return func;
}

/* --- End Section 9: PUBLIC API --- */
