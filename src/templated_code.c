/*
    Template C source file for new translation units.
    Copy this file when creating new implementation files.

    Use this structure for consistent layout and easier code review.
    Obviously, you should replace the template code with your 
    actual implementation !!
*/

/* ==========================================================================
   SECTION 1: INCLUDES
   Standard library headers first, then project headers.
   ========================================================================== */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


/* --- End Section 1: INCLUDES --- */

/* ==========================================================================
   SECTION 2: COMPILE-TIME CONFIGURATION
   File-local compile-time constants, build-time defaults, and feature flags.
   ========================================================================== */

#ifndef TEMPLATE_ALIGNMENT
#define TEMPLATE_ALIGNMENT 64
#endif

/* --- End Section 2: COMPILE-TIME CONFIGURATION --- */

/* ==========================================================================
   SECTION 3: INFRASTRUCTURAL MACROS
   Private macros used by helper functions and internal operations.
   ========================================================================== */

#define TEMPLATE_STRINGIFY(x) #x
#define TEMPLATE_PRAGMA(x) _Pragma(TEMPLATE_STRINGIFY(x))

/* --- End Section 3: INFRASTRUCTURAL MACROS --- */

/* ==========================================================================
   SECTION 4: PRIVATE IMPLEMENTATION MACROS
   File-local operational macros and helper wrappers.
   ========================================================================== */

#define TEMPLATE_MAX(a, b) ((a) > (b) ? (a) : (b))

/* --- End Section 4: PRIVATE IMPLEMENTATION MACROS --- */

/* ==========================================================================
   SECTION 5: INTERNAL DATA STRUCTURES AND ENUMS
   Local types and opaque data structures.
   ========================================================================== */

typedef struct TemplateState {
    int value;
} TemplateState;

/* --- End Section 5: INTERNAL DATA STRUCTURES AND ENUMS --- */

/* ==========================================================================
   SECTION 6: STATIC DATA
   File-scope constants and persistent state.
   ========================================================================== */

static TemplateState g_template_state = {0};

/* --- End Section 6: STATIC DATA --- */

/* ==========================================================================
   SECTION 7: INTERNAL FUNCTION FORWARD DECLARATIONS
   Static helper prototypes for this translation unit.
   ========================================================================== */

static void template_initialize_state(void);
static int template_compute(int a, int b);

/* --- End Section 7: INTERNAL FUNCTION FORWARD DECLARATIONS --- */

/* ==========================================================================
   SECTION 8: INTERNAL HELPER FUNCTION DEFINITIONS
   Low-level helpers and internal utility functions.
   ========================================================================== */

static void template_initialize_state(void) {
    g_template_state.value = 0;
}

static int template_compute(int a, int b) {
    return TEMPLATE_MAX(a, b) + g_template_state.value;
}

/* --- End Section 8: INTERNAL HELPER FUNCTION DEFINITIONS --- */

/* ==========================================================================
   SECTION 9: PUBLIC API
   Exported functions for this module.
   ========================================================================== */

void Template_init(void) {
    template_initialize_state();
}

int Template_run(int a, int b) {
    return template_compute(a, b);
}

/* --- End Section 9: PUBLIC API --- */
