# ----------------------------------------------------------------------------
# BENCHMARK EXTENSION MAKEFILE - MULTI-GPU RUNNER
# ----------------------------------------------------------------------------
include Makefile

# NOTE: this file appends to CFLAGS/HOST_ONLY_CFLAGS below (it never
# reassigns them), so every target here (including openmp_bit_nocpu)
# automatically inherits Makefile's clang toolchain-consistency fixes:
#   - version-matched lld/llvm-ar for LTO bitcode (APPLY_LTO=1, default)
#   - version-matched OpenMP runtime rpath, preventing a mismatched-version
#     libomp.so.5/libomptarget.so from being loaded at run time via
#     LD_LIBRARY_PATH (CLANG_RUNTIME_RPATH=1, default) - this was the fix
#     for a confirmed deadlock hang in openmp_bit/openmp_bit_nogpu on
#     systems with multiple LLVM versions installed. See Makefile's
#     "clang toolchain-consistency fixes" section for full details.
# The NVCC/HIPCC-linked cuda_gpu_bench/hip_gpu_bench targets below link
# directly via $(NVCC)/$(HIPCC) rather than through $(CC)/CFLAGS, so they do
# NOT automatically get the rpath fix; they have not been observed to hit
# the same hang, so this has been left alone rather than fixed speculatively.

# used to configure gpu_bench_csv target parameters (can be overridden by \
# environment variables or command-line arguments)
GPU_TILE_DIM ?= 32
GPU_BLOCK_ROWS ?= 8
GPU_DEVICE_ID ?= 0
GPU_NUM_BITS ?= 16384
GPU_NUM_QUERIES ?= 1000
GPU_NUM_REFS ?= 1024
GPU_ITERATIONS ?= 100
GPU_CSV_OUTPUT ?= gpu_bench_summary.csv
GPU_LOG_OUTPUT ?= gpu_bench_raw.log

GPU_CSV_NUMERICS := GPU_TILE_DIM GPU_BLOCK_ROWS GPU_DEVICE_ID GPU_NUM_BITS GPU_NUM_QUERIES GPU_NUM_REFS GPU_ITERATIONS
GPU_CSV_STRINGS := GPU_CSV_OUTPUT GPU_LOG_OUTPUT
$(foreach var,$(GPU_CSV_NUMERICS), \
    $(if $(shell echo "$($(var))" | grep -Eq '^[0-9]+$$' && echo ok),, \
        $(eval $(call APPEND_ERROR, $(var) must be a positive integer. Got: '$($(var))')) \
    ) \
)

$(foreach var,$(GPU_CSV_STRINGS), \
    $(if $(strip $($(var))),, \
        $(eval $(call APPEND_ERROR, $(var) must be a non-empty string. Got: '$($(var))')) \
    ) \
)

# this is used to generate GPU-compatible topk code in topk_internal.h
# stops if one is trying to use a GPU-incompatible configuration for topk
ifeq ($(origin GPU_COMPILE_TOPK), undefined)
    GPU_COMPILE_TOPK := $(if $(filter NONE,$(GPU)),0,1)
endif
ifeq ($(GPU_COMPILE_TOPK),1)
    ifeq ($(GPU),NONE)
        $(eval $(call APPEND_ERROR, GPU_COMPILE_TOPK cannot be 1 when GPU is NONE))
    endif
endif
CFLAGS += -DGPU_COMPILE_TOPK=$(GPU_COMPILE_TOPK)



CUDA_BENCH_SRC := benchmark/native_device_code.cpp
CUDA_BENCH_EXEC := $(BUILD_DIR)/cuda_gpu_benchmark
CUDA_BENCH_OBJ := $(BUILD_DIR)/cuda_gpu_benchmark.o
NVCC ?= nvcc
NVCC_ARCH ?= sm_70
NVCC_FLAGS ?= -O3 -std=c++14 -x cu -I./include -I./src

HIP_BENCH_SRC := benchmark/native_device_code.cpp
HIP_BENCH_EXEC := $(BUILD_DIR)/hip_gpu_benchmark
HIP_BENCH_OBJ := $(BUILD_DIR)/hip_gpu_benchmark.o
HIPCC ?= hipcc
HIPCC_FLAGS ?= -O3 -std=c++17 -x hip -I./include -I./src


GPU_BENCH_EXECS :=
ifeq ($(filter NVIDIA,$(GPU_LIST)),NVIDIA)
  GPU_BENCH_EXECS += $(CUDA_BENCH_EXEC)
  NVCC_FLAGS += -DGPU_TILE_J=$(GPU_TILE_J) -DGPU_ILP=$(GPU_ILP) -DGPU_TILE_DIM=$(GPU_TILE_DIM) -DGPU_BLOCK_ROWS=$(GPU_BLOCK_ROWS)
endif
ifeq ($(filter AMD,$(GPU_LIST)),AMD)
  GPU_BENCH_EXECS += $(HIP_BENCH_EXEC)
  HIPCC_FLAGS += -DGPU_TILE_J=$(GPU_TILE_J) -DGPU_ILP=$(GPU_ILP) -DGPU_TILE_DIM=$(GPU_TILE_DIM) -DGPU_BLOCK_ROWS=$(GPU_BLOCK_ROWS)
endif

# Cleanly extract just the numeric version for NVCC & inject JIT fallback code payload
NVCC_ARCH_LIST := $(strip $(NVIDIA_ARCH_LIST))
NVCC_ARCH_FLAGS :=
ifneq ($(NVCC_ARCH_LIST),)
  NVCC_ARCH_NUMS := $(subst sm_,,$(subst compute_,,$(NVCC_ARCH_LIST)))
  NVCC_ARCH_FLAGS := $(foreach arch,$(NVCC_ARCH_NUMS),-gencode=arch=compute_$(arch),code=sm_$(arch) -gencode=arch=compute_$(arch),code=compute_$(arch))
else
  NVCC_ARCH_NUM := $(subst sm_,,$(subst compute_,,$(NVCC_ARCH)))
  NVCC_ARCH_FLAGS := -gencode=arch=compute_$(NVCC_ARCH_NUM),code=sm_$(NVCC_ARCH_NUM) -gencode=arch=compute_$(NVCC_ARCH_NUM),code=compute_$(NVCC_ARCH_NUM)
endif

HIPCC_ARCH_FLAGS := $(foreach arch,$(AMD_ARCH_LIST),--offload-arch=$(arch))



.PHONY: gpu_bench_csv cuda_gpu_bench hip_gpu_bench openmp_bit_nocpu openmp_bit_nocpu_FAISS_COMP clean-bench gpu_param_sweep distclean-bench


gpu_bench_csv: $(GPU_BENCH_EXECS)
	@mkdir -p $(dir $(GPU_CSV_OUTPUT)) 2>/dev/null || true
	@mkdir -p $(dir $(GPU_LOG_OUTPUT)) 2>/dev/null || true
	@echo "Running GPU bench..."
	# Loop through the list of executables and run each one
	@for exec in $(GPU_BENCH_EXECS); do \
		echo "Executing $$exec"; \
		GPU_CSV_OUTPUT="$(GPU_CSV_OUTPUT)" ./$$exec $(GPU_NUM_BITS) $(GPU_NUM_QUERIES) $(GPU_NUM_REFS) $(GPU_ITERATIONS) $(GPU_DEVICE_ID) >> "$(GPU_LOG_OUTPUT)" 2>&1; \
	done


ifeq ($(filter NVIDIA,$(GPU_LIST)),)
ifneq ($(filter cuda_gpu_bench,$(MAKECMDGOALS) $(.DEFAULT_GOAL)),)
$(eval $(call APPEND_ERROR,Execution Halted: Cannot build 'cuda_gpu_bench' \
    because GPU=NVIDIA was not requested in your GPU list. \
    Current GPU_LIST='$(GPU_LIST)'))
endif
else
cuda_gpu_bench: $(CUDA_BENCH_EXEC)
endif

ifeq ($(filter AMD,$(GPU_LIST)),)
ifneq ($(filter hip_gpu_bench,$(MAKECMDGOALS) $(.DEFAULT_GOAL)),)
$(eval $(call APPEND_ERROR,Execution Halted: Cannot build 'hip_gpu_bench' \
    because GPU=AMD was not requested in your GPU list. \
    Current GPU_LIST='$(GPU_LIST)'))
endif
else
hip_gpu_bench: $(HIP_BENCH_EXEC)
endif

BENCH_OMP_NO_CPU_SRC := benchmark/openmp_bit_nocpu.c
BENCH_OMP_NO_CPU_OBJ := $(BUILD_DIR)/openmp_bit_nocpu.o
BENCH_OMP_NO_CPU_FAISS_COMP_SRC := benchmark/openmp_bit_nocpu_FAISS_comp.c
BENCH_OMP_NO_CPU_FAISS_COMP_OBJ := $(BUILD_DIR)/openmp_bit_nocpu_FAISS_comp.o
BENCH_OMP_NO_CPU_BIT_OBJ := $(BUILD_DIR)/bit_nocpu_host.o
BENCH_OMP_NO_CPU_EXEC := $(BUILD_DIR)/openmp_bit_nocpu
BENCH_OMP_NO_CPU_FAISS_COMP_EXEC := $(BUILD_DIR)/openmp_bit_nocpu_FAISS_comp
BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ := $(BUILD_DIR)/gpu_layout_registry.o
BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ := $(BUILD_DIR)/gpu_layout_fsm.o
BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ := $(BUILD_DIR)/gpu_layout_kernels.o

# When NONE is requested we must *not* allow the offloading-only targets
ifeq ($(filter NONE,$(GPU_LIST)),NONE)

  # Only record the error if the user actually asked for one of these targets
  ifneq ($(filter openmp_bit_nocpu,$(MAKECMDGOALS) $(.DEFAULT_GOAL)),)
  $(eval $(call APPEND_ERROR,Execution Halted: \
  openmp_bit_nocpu requires functional offloading; specify NVIDIA or AMD in your target array))
  endif

  ifneq ($(filter openmp_bit_nocpu_FAISS_comp,$(MAKECMDGOALS) $(.DEFAULT_GOAL)),)
  $(eval $(call APPEND_ERROR,Execution Halted: \
  openmp_bit_nocpu_FAISS_comp requires functional offloading; specify NVIDIA or AMD in your target array))
  endif

  ifneq ($(filter gpu_param_sweep,$(MAKECMDGOALS) $(.DEFAULT_GOAL)),)
  $(eval $(call APPEND_ERROR,gpu_param_sweep requires functional offloading; specify NVIDIA or AMD in your target array))
  endif
else
  # Normal case – GPU offloading is available
  openmp_bit_nocpu:        $(BENCH_OMP_NO_CPU_EXEC)
  openmp_bit_nocpu_FAISS_comp: $(BENCH_OMP_NO_CPU_FAISS_COMP_EXEC)
  gpu_param_sweep:         $(BENCH_GPU_PARAM_SWEEP_EXEC)
endif

ifeq ($(GPU_COMPILE_TOPK),1)
TOPK_OBJ := $(BUILD_DIR)/topk_gpu.o
$(BUILD_DIR)/topk_gpu.o: src/topk_gpu.c src/topk.h src/topk_internal.h   
	$(CC) $(CFLAGS) -c $< -o $@
else
TOPK_OBJ := $(BUILD_DIR)/topk_cpu.o
$(BUILD_DIR)/topk_cpu.o: src/topk_cpu.c src/topk.h src/topk_internal.h
	$(CC) $(CFLAGS) -c $< -o $@
endif

# Warn about configuration errors and exit early if any are found
ifneq ($(strip $(ERRORS)),)
  FORMATTED_ERRORS := $(subst |,$(newline) -> ,$(ERRORS))
  $(info )
  $(info =============================================)
  $(info ===       Build Configuration Errors      ===)
  $(info =============================================)
  $(info $(FORMATTED_ERRORS))
  $(info )
  $(error Build configuration is invalid - see details above)
endif

$(BENCH_OMP_NO_CPU_OBJ): $(BENCH_OMP_NO_CPU_SRC) $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BENCH_OMP_NO_CPU_FAISS_COMP_OBJ): $(BENCH_OMP_NO_CPU_FAISS_COMP_SRC) $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BENCH_OMP_NO_CPU_BIT_OBJ): src/bit.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -c $< -o $@

$(BENCH_OMP_NO_CPU_EXEC): $(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $@ \
	$(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ) \
	$(OMPTARGET_RPATH_FLAG) -lrt -lm

$(BENCH_OMP_NO_CPU_FAISS_COMP_EXEC): $(BENCH_OMP_NO_CPU_FAISS_COMP_OBJ) $(TOPK_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $@ \
	$(BENCH_OMP_NO_CPU_FAISS_COMP_OBJ) $(TOPK_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ) \
	$(OMPTARGET_RPATH_FLAG) -lrt -lm


# Wrapped OpenMP linker dependencies for NVCC
$(CUDA_BENCH_EXEC): $(CUDA_BENCH_OBJ) $(OPENMP_BIT_HELPERS_OBJ)
	$(NVCC) -o $@ $^ -lm -Xlinker --no-as-needed $(HOST_OPENMP_LIBS) -Xlinker --as-needed

# Added CONFIG_STAMP dependency here
$(CUDA_BENCH_OBJ): $(CUDA_BENCH_SRC) $(CONFIG_STAMP)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH_FLAGS) -c $< -o $@

# Wrapped OpenMP linker dependencies for HIPCC
$(HIP_BENCH_EXEC): $(HIP_BENCH_OBJ) $(OPENMP_BIT_HELPERS_OBJ)
	$(HIPCC) --hip-link -lstdc++ -o $@ $^ -lm -Wl,--no-as-needed $(HOST_OPENMP_LIBS) -Wl,--as-needed

# Added CONFIG_STAMP dependency here
$(HIP_BENCH_OBJ): $(HIP_BENCH_SRC) $(CONFIG_STAMP)
	$(HIPCC) $(HIPCC_FLAGS) $(HIPCC_ARCH_FLAGS) -c $< -o $@

# ----------------------------------------------------------------------------
# GPU PARAMETER SWEEP EXECUTABLE
# ----------------------------------------------------------------------------
GPU_SWEEP_SRC := benchmark/gpu_param_sweep.c
GPU_SWEEP_OBJ := $(BUILD_DIR)/gpu_param_sweep.o
GPU_SWEEP_EXEC := $(BUILD_DIR)/gpu_param_sweep


$(GPU_SWEEP_OBJ): $(GPU_SWEEP_SRC) $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(GPU_SWEEP_EXEC): $(GPU_SWEEP_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $@ \
	$(GPU_SWEEP_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ) \
	$(OMPTARGET_RPATH_FLAG) -lrt -lm


clean-bench:
	rm -f $(BUILD_DIR)/cuda_gpu_benchmark $(BUILD_DIR)/cuda_gpu_benchmark.o $(BUILD_DIR)/hip_gpu_benchmark $(BUILD_DIR)/hip_gpu_benchmark.o $(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_FAISS_COMP_OBJ)
	rm -f $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ)
	rm -f $(BENCH_OMP_NO_CPU_EXEC) $(BENCH_OMP_NO_CPU_FAISS_COMP_EXEC)
	rm -f $(OPENMP_BIT_HELPERS_OBJ)
	rm -f $(BUILD_DIR)/gpu_param_sweep $(BUILD_DIR)/gpu_param_sweep.o

distclean-bench: clean-bench