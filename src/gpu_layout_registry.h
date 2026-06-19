#pragma once

#include <stdatomic.h>
#include "gpu_layout.h"
#include "gpu_layout_fsm.h"

/* The allocation node type is private to the registry module. */
typedef struct GPUAllocationState {
    const void *host_ptr;
    int device_id;
    _Atomic uint32_t state_word;
    _Atomic int transition_lock;
    _Atomic int active_users;
    struct GPUAllocationState *next;
} GPUAllocationState;

int registry_checkout_fast_path(uint64_t *bits, int device_id, uint32_t target_state);
GPUAllocationState *registry_claim_transition(uint64_t *bits, int device_id);
void registry_commit_transition(GPUAllocationState *node, uint32_t final_target);
void release_gpu_layout(uint64_t *bits, int device_id);

#define ENSURE_GPU_LAYOUT(bits, rows, cols, target_state, device_id, params, params_size) \
    do { \
        if (!registry_checkout_fast_path((bits), (device_id), (target_state))) { \
            GPUAllocationState *_node = registry_claim_transition((bits), (device_id)); \
            uint32_t _current = atomic_load_explicit(&_node->state_word, memory_order_acquire); \
            if (_current != (target_state)) { \
                GPUDataState _c_base = (GPUDataState)(_current & MASK_BASE_LAYOUT); \
                GPUDataState _t_base = (GPUDataState)((target_state) & MASK_BASE_LAYOUT); \
                if (_c_base != _t_base) { \
                    if (_c_base != LAYOUT_ROW_MAJOR) { \
                        TransitionKernel _kfunc = get_transition_kernel(_c_base, LAYOUT_ROW_MAJOR); \
                        _kfunc(_c_base, LAYOUT_ROW_MAJOR, (bits), (rows), (cols), (device_id), (params), (params_size)); \
                    } \
                    if (_t_base != LAYOUT_ROW_MAJOR) { \
                        TransitionKernel _kfunc = get_transition_kernel(LAYOUT_ROW_MAJOR, _t_base); \
                        _kfunc(LAYOUT_ROW_MAJOR, _t_base, (bits), (rows), (cols), (device_id), (params), (params_size)); \
                    } \
                } \
            } \
            registry_commit_transition(_node, (target_state)); \
        } \
    } while (0)
