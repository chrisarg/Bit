# ----------------------------------------------------------------------------
# BENCHMARK EXTENSION MAKEFILE - MULTI-GPU RUNNER
# ----------------------------------------------------------------------------
include Makefile

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

# Validate runtime algorithms configuration flags
OPENMP_GPU_IMPL ?= TEAM_PARALLEL_SIMD
override OPENMP_GPU_IMPL := $(shell printf '%s' '$(OPENMP_GPU_IMPL)' | tr 'a-z' 'A-Z' | tr -d '[:space:]')
OPENMP_GPU_IMPL_OPTIONS := TEAM_PARALLEL_SIMD TRANSPOSED_TEAM_PARALLEL_SIMD SHARED_TILE_ILP
OPENMP_GPU_IMPL_OK := $(filter $(OPENMP_GPU_IMPL),$(OPENMP_GPU_IMPL_OPTIONS))
ifeq ($(strip $(OPENMP_GPU_IMPL_OK)),)
  $(error OPENMP_GPU_IMPL=$(OPENMP_GPU_IMPL) is not one of $(OPENMP_GPU_IMPL_OPTIONS))
endif

$(info Utilizing OpenMP GPU Strategy: $(OPENMP_GPU_IMPL))
OPENMP_GPU_IMPL_MACRO := -DOPENMP_GPU_IMPL_$(OPENMP_GPU_IMPL)

# Append ONLY new macros to prevent duplicating all optimization & architecture rules
CFLAGS += -DUSE_LIBPOPCNT=$(LIBPOPCNT_VAL) $(OPENMP_GPU_IMPL_MACRO) -I./src

.PHONY: gpu_bench_csv cuda_gpu_bench hip_gpu_bench openmp_bit_nocpu clean-bench distclean-bench


gpu_bench_csv: $(GPU_BENCH_EXECS)
	@mkdir -p $(dir $(GPU_CSV_OUTPUT)) 2>/dev/null || true
	@mkdir -p $(dir $(GPU_LOG_OUTPUT)) 2>/dev/null || true
	@echo "Running GPU bench..."
	# Loop through the list of executables and run each one
	@for exec in $(GPU_BENCH_EXECS); do \
		echo "Executing $$exec"; \
		GPU_CSV_OUTPUT="$(GPU_CSV_OUTPUT)" ./$$exec $(GPU_NUM_BITS) $(GPU_NUM_QUERIES) $(GPU_NUM_REFS) $(GPU_ITERATIONS) $(GPU_DEVICE_ID) >> "$(GPU_LOG_OUTPUT)" 2>&1; \
	done


ifeq ($(filter NVIDIA,$(GPU_LIST)),NVIDIA)
cuda_gpu_bench: $(CUDA_BENCH_EXEC)
else
cuda_gpu_bench:
	$(error Execution Halted: Cannot build 'cuda_gpu_bench' because GPU=NVIDIA was not requested in your GPU list. Current GPU_LIST='$(GPU_LIST)')
endif

ifeq ($(filter AMD,$(GPU_LIST)),AMD)
hip_gpu_bench: $(HIP_BENCH_EXEC)
else
hip_gpu_bench:
	$(error Execution Halted: Cannot build 'hip_gpu_bench' because GPU=AMD was not requested in your GPU list. Current GPU_LIST='$(GPU_LIST)')
endif

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

BENCH_OMP_NO_CPU_SRC := benchmark/openmp_bit_nocpu.c
BENCH_OMP_NO_CPU_OBJ := $(BUILD_DIR)/openmp_bit_nocpu.o
BENCH_OMP_NO_CPU_BIT_OBJ := $(BUILD_DIR)/bit_nocpu_host.o
BENCH_OMP_NO_CPU_EXEC := $(BUILD_DIR)/openmp_bit_nocpu
BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ := $(BUILD_DIR)/gpu_layout_registry.o
BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ := $(BUILD_DIR)/gpu_layout_fsm.o
BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ := $(BUILD_DIR)/gpu_layout_kernels.o

ifeq ($(filter NONE,$(GPU_LIST)),NONE)
openmp_bit_nocpu:
	$(error openmp_bit_nocpu requires functional offloading; specify NVIDIA or AMD in your target array)
else
openmp_bit_nocpu: $(BENCH_OMP_NO_CPU_EXEC)

$(BENCH_OMP_NO_CPU_OBJ): $(BENCH_OMP_NO_CPU_SRC) $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BENCH_OMP_NO_CPU_BIT_OBJ): src/bit.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

$(BENCH_OMP_NO_CPU_EXEC): $(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $@ \
	$(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ) $(OPENMP_BIT_HELPERS_OBJ) \
	$(OMPTARGET_RPATH_FLAG) -lrt -lm
endif

clean-bench:
	rm -f $(BUILD_DIR)/cuda_gpu_benchmark $(BUILD_DIR)/cuda_gpu_benchmark.o $(BUILD_DIR)/hip_gpu_benchmark $(BUILD_DIR)/hip_gpu_benchmark.o $(BUILD_DIR)/openmp_bit_nocpu.o
	rm -f $(BENCH_OMP_NO_CPU_GPUTL_REGISTRY_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_FSM_OBJ) $(BENCH_OMP_NO_CPU_GPUTL_KERNELS_OBJ)
	rm -f $(BUILD_DIR)/openmp_bit_nocpu
	rm -f $(OPENMP_BIT_HELPERS_OBJ)

distclean-bench: clean-bench