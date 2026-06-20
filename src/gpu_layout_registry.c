/* ==========================================================================
   SECTION 1: INCLUDES
   Standard library headers first, then project headers.
   ========================================================================== */

#include <stdlib.h>
#include <stdio.h>
#include "gpu_layout_registry.h"

/* --- End Section 1: INCLUDES --- */

/* ==========================================================================
   SECTION 2: COMPILE-TIME CONFIGURATION
   File-local compile-time constants, build-time defaults, and feature flags.
   ========================================================================== */

/* No compile-time configuration required for this module. */

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
   Types local to the registry implementation.
   ========================================================================== */

/* No extra internal types are needed beyond the public registry node type. */

/* --- End Section 5: INTERNAL DATA STRUCTURES AND ENUMS --- */

/* ==========================================================================
   SECTION 6: STATIC DATA
   File-scope constants and persistent state.
   ========================================================================== */

static atomic_flag g_registry_lock = ATOMIC_FLAG_INIT;
static GPUAllocationState *g_registry_head = NULL;

/* --- End Section 6: STATIC DATA --- */

/* ==========================================================================
   SECTION 7: INTERNAL FUNCTION FORWARD DECLARATIONS
   Static helper prototypes for this translation unit.
   ========================================================================== */

static GPUAllocationState *get_or_create_node(const void *host_ptr, int device_id);

/* --- End Section 7: INTERNAL FUNCTION FORWARD DECLARATIONS --- */

/* ==========================================================================
   SECTION 8: INTERNAL HELPER FUNCTION DEFINITIONS
   Low-level helpers and registry internals.
   ========================================================================== */

static GPUAllocationState *get_or_create_node(const void *host_ptr, int device_id) {
    while (atomic_flag_test_and_set_explicit(&g_registry_lock, memory_order_acquire))
        ;

    GPUAllocationState *entry = NULL;
    GPUAllocationState *curr = g_registry_head;
    GPUAllocationState *first_empty = NULL;

    while (curr != NULL) {
        if (curr->host_ptr == host_ptr && curr->device_id == device_id) {
            entry = curr;
            break;
        }
        if (curr->host_ptr == NULL && first_empty == NULL) {
            first_empty = curr;
        }
        curr = curr->next;
    }

    if (!entry) {
        GPUAllocationState *target = first_empty
                                          ? first_empty
                                          : (GPUAllocationState *)malloc(
                                                sizeof(GPUAllocationState));
        if (!first_empty && target) {
            target->next = g_registry_head;
            g_registry_head = target;
        }
        if (target) {
            target->host_ptr = host_ptr;
            target->device_id = device_id;
            atomic_init(&target->state_word,
                        MAKE_STATE(LAYOUT_ROW_MAJOR, FLAG_NONE));
            atomic_init(&target->transition_lock, 0);
            atomic_init(&target->active_users, 0);
            entry = target;
        }
    }

    atomic_flag_clear_explicit(&g_registry_lock, memory_order_release);
    return entry;
}

/* --- End Section 8: INTERNAL HELPER FUNCTION DEFINITIONS --- */

/* ==========================================================================
   SECTION 9: PUBLIC API
   Registry entry points.
   ========================================================================== */

int registry_checkout_fast_path(uint64_t *bits, int device_id, uint32_t target_state) {
    GPUAllocationState *node = get_or_create_node(bits, device_id);
    if (atomic_load_explicit(&node->state_word, memory_order_acquire) == target_state) {
        atomic_fetch_add_explicit(&node->active_users, 1, memory_order_release);
        return 1;
    }
    return 0;
}

GPUAllocationState *registry_claim_transition(uint64_t *bits, int device_id) {
    GPUAllocationState *node = get_or_create_node(bits, device_id);
    while (1) {
        if (atomic_load_explicit(&node->transition_lock, memory_order_acquire) == 1)
            continue;
        if (atomic_load_explicit(&node->active_users, memory_order_acquire) > 0)
            continue;

        int expected = 0;
        if (atomic_compare_exchange_weak_explicit(&node->transition_lock, &expected,
                                                 1, memory_order_acq_rel,
                                                 memory_order_acquire)) {
            return node;
        }
    }
}

void registry_commit_transition(GPUAllocationState *node, uint32_t final_target) {
    atomic_store_explicit(&node->state_word, final_target, memory_order_release);
    atomic_fetch_add_explicit(&node->active_users, 1, memory_order_release);
    atomic_store_explicit(&node->transition_lock, 0, memory_order_release);
}

void release_gpu_layout(uint64_t *bits, int device_id) {
    while (atomic_flag_test_and_set_explicit(&g_registry_lock, memory_order_acquire))
        ;

    GPUAllocationState *curr = g_registry_head;
    GPUAllocationState *target = NULL;
    while (curr != NULL) {
        if (curr->host_ptr == bits && curr->device_id == device_id) {
            target = curr;
            break;
        }
        curr = curr->next;
    }

    atomic_flag_clear_explicit(&g_registry_lock, memory_order_release);
    if (target) {
        atomic_fetch_sub_explicit(&target->active_users, 1, memory_order_release);
    }
}

/* --- End Section 9: PUBLIC API --- */
