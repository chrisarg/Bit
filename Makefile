IS_CLEAN_GOAL := $(filter clean distclean,$(MAKECMDGOALS))

# Ensure plain `make` builds the libraries (not the first internal target).
.DEFAULT_GOAL := all

# Keep libomptarget runtime diagnostics quiet by default.
# Users can override when invoking make, e.g.:
#   make LIBOMPTARGET_INFO=16 LIBOMPTARGET_DEBUG=1 ...
LIBOMPTARGET_INFO ?= 0
LIBOMPTARGET_DEBUG ?= 0
export LIBOMPTARGET_INFO
export LIBOMPTARGET_DEBUG

# Check for available compilers if user hasn't specified one
ifeq ($(origin CC),default)
ifeq ($(IS_CLEAN_GOAL),)
$(info Default compiler not set, checking for available compilers...)
# Check for gcc first
ifneq ($(shell which gcc 2>/dev/null),)
$(info Using gcc to compile)
CC=gcc
# Then check for clang
else ifneq ($(shell which clang 2>/dev/null),)
$(info Using clang to compile)
CC=clang
endif
else
# For clean-only invocations we don't need compiler discovery.
CC=gcc
endif
endif

ifeq ($(origin GPU),undefined)
ifeq ($(IS_CLEAN_GOAL),)
$(info Default GPU offload not set, will set to NONE)
GPU=NONE
else
# For clean-only invocations, pick a quiet, valid default.
GPU=NONE
endif
endif

# Convert GPU to uppercase for case-insensitive comparison
override GPU := $(shell printf '%s' '$(GPU)' | tr 'a-z' 'A-Z' \
	| tr -d '[:space:]')

# check to see if the CC is one of clang or gcc
# Skip validation/messages for clean-only invocations.
ifeq ($(IS_CLEAN_GOAL),)
ifneq ($(filter $(notdir $(CC)),gcc clang),)
$(info CC is set to $(CC), which is a supported compiler)
else
$(error CC=$(CC) is not one of the supported compilers (gcc, clang))
endif
endif

## additional flags
DEFINES ?=
ROCM_PATH ?= /opt/rocm
ROCM_DEVICE_LIB_PATH ?= $(ROCM_PATH)/amdgcn/bitcode
ROCM_LLVM_BIN ?= $(ROCM_PATH)/lib/llvm/bin
CC_ENV :=
NVIDIA_ARCH ?=
NVIDIA_ARCH_POLICY ?= A
AMD_ARCH ?=
CUDA_PATH ?= /usr/lib/cuda
BUG_REPORT ?= 0
BUG_REPORT_DIR ?= bug_reports
BUG_REPORT_OUT ?= $(BUG_REPORT_DIR)
BUG_TARGET ?= bench_omp

empty :=
space := $(empty) $(empty)
comma := ,

override NVIDIA_ARCH := $(shell printf '%s' '$(NVIDIA_ARCH)' | tr 'A-Z' 'a-z' \
	| tr -d '[:space:]')
override NVIDIA_ARCH_POLICY := $(shell printf '%s' '$(NVIDIA_ARCH_POLICY)' \
	| tr 'a-z' 'A-Z' | tr -d '[:space:]')
override AMD_ARCH := $(shell printf '%s' '$(AMD_ARCH)' | tr 'A-Z' 'a-z' \
	| tr -d '[:space:]')

# Preserve order while removing duplicate architectures.
dedup_words = $(strip $(shell \
	printf '%s\n' "$(strip $(1))" | tr ' ' '\n' | sed '/^$$/d' | awk '!seen[$$0]++'))

# Probe a list of NVIDIA SM archs and return the supported subset.
probe_nvidia_archs = $(strip $(shell for arch in $(1); do \
	if [ "$(notdir $(CC))" = "gcc" ]; then \
		printf 'int main(void){return 0;}\n' \
		| $(CC) -x c - -fPIC -shared -fopenmp -foffload=nvptx-none \
		-foffload-options=nvptx-none=-march=$$arch -c -o /dev/null \
		>/dev/null 2>&1 && printf '%s ' $$arch; \
	elif [ "$(notdir $(CC))" = "clang" ]; then \
		printf 'int main(void){return 0;}\n' \
		| $(CC) -x c - -fPIC -shared -fopenmp \
		-fopenmp-targets=nvptx64-nvidia-cuda \
		--cuda-path=$(CUDA_PATH) --offload-arch=$$arch -c -o /dev/null \
		>/dev/null 2>&1 && printf '%s ' $$arch; \
	fi; \
done))

$(if $(filter-out A B C,$(NVIDIA_ARCH_POLICY)), \
	$(error NVIDIA_ARCH_POLICY must be one of A, B, or C; \
	got '$(NVIDIA_ARCH_POLICY)'))

NVIDIA_SYSTEM_ARCHES := $(shell \
	nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
	| awk -F'.' '{gsub(/ /, "", $$1); gsub(/ /, "", $$2); \
	print "sm_" $$1 $$2}' \
	| awk '!seen[$$0]++' \
	| paste -sd, -)
NVIDIA_SYSTEM_ARCH_LIST := $(strip $(subst $(comma), ,$(NVIDIA_SYSTEM_ARCHES)))
NVIDIA_KNOWN_ARCH_LIST := sm_35 sm_37 sm_50 sm_52 sm_53 sm_60 sm_61 sm_62 \
	sm_70 sm_72 sm_75 sm_80 sm_86 sm_87 sm_89 sm_90 sm_100 sm_101 sm_103 \
	sm_110 sm_120
NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST :=
NVIDIA_COMPILER_SUPPORTED_ARCH_LIST :=
NVIDIA_SELECTION_MODE := explicit-override
ifneq ($(strip $(NVIDIA_ARCH)),)
NVIDIA_EFFECTIVE_ARCHES := $(NVIDIA_ARCH)
NVIDIA_ARCH_LIST := $(strip $(subst $(comma), ,$(NVIDIA_EFFECTIVE_ARCHES)))
NVIDIA_ARCH_LIST := $(call dedup_words,$(NVIDIA_ARCH_LIST))
NVIDIA_EFFECTIVE_ARCHES := \
	$(strip $(subst $(space),$(comma),$(NVIDIA_ARCH_LIST)))
else
NVIDIA_SELECTION_MODE := policy-implicit
NVIDIA_ARCH_LIST :=
ifeq ($(GPU),NVIDIA)
NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST := \
	$(call probe_nvidia_archs,$(NVIDIA_SYSTEM_ARCH_LIST))
NVIDIA_COMPILER_SUPPORTED_ARCH_LIST := \
	$(call probe_nvidia_archs,$(NVIDIA_KNOWN_ARCH_LIST))
NVIDIA_EARLIEST_VISIBLE_SUPPORTED_ARCH := $(shell \
	if [ -n "$(strip $(NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST))" ]; then \
		printf '%s\n' '$(NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST)' \
		| tr ' ' '\n' | sed '/^$$/d' | sed 's/^sm_//' | sort -n \
		| head -n 1 | awk '{print "sm_" $$1}'; \
	fi)
NVIDIA_EARLIEST_COMPILER_SUPPORTED_ARCH := $(shell \
	if [ -n "$(strip $(NVIDIA_COMPILER_SUPPORTED_ARCH_LIST))" ]; then \
		printf '%s\n' '$(NVIDIA_COMPILER_SUPPORTED_ARCH_LIST)' \
		| tr ' ' '\n' | sed '/^$$/d' | sed 's/^sm_//' | sort -n \
		| head -n 1 | awk '{print "sm_" $$1}'; \
	fi)
ifneq ($(strip $(NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST)),)
ifeq ($(notdir $(CC)),gcc)
NVIDIA_ARCH_LIST := $(NVIDIA_EARLIEST_COMPILER_SUPPORTED_ARCH)
else
NVIDIA_ARCH_LIST := $(NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST)
endif
else
ifeq ($(NVIDIA_ARCH_POLICY),A)
NVIDIA_ARCH_LIST := $(NVIDIA_EARLIEST_COMPILER_SUPPORTED_ARCH)
else ifeq ($(NVIDIA_ARCH_POLICY),B)
$(error NVIDIA_ARCH_POLICY=B requires explicit NVIDIA_ARCH=sm_xy \
when no visible NVIDIA GPU architecture is compiler-supported)
else ifeq ($(NVIDIA_ARCH_POLICY),C)
ifeq ($(notdir $(CC)),gcc)
$(error No visible NVIDIA SM is compiler-supported with CC=gcc; \
rerun with CC=clang or set explicit NVIDIA_ARCH=sm_xy)
else
$(error No visible NVIDIA SM is compiler-supported with CC=clang; \
set explicit NVIDIA_ARCH=sm_xy)
endif
endif
endif
endif
NVIDIA_EFFECTIVE_ARCHES := \
	$(strip $(subst $(space),$(comma),$(call dedup_words,$(NVIDIA_ARCH_LIST))))
endif

NVIDIA_MISSING_VISIBLE_ARCHES := \
	$(filter-out $(NVIDIA_ARCH_LIST),$(NVIDIA_SYSTEM_ARCH_LIST))
NVIDIA_INVALID_ARCH := $(shell if [ -n "$(NVIDIA_EFFECTIVE_ARCHES)" ]; then \
	printf '%s\n' '$(NVIDIA_EFFECTIVE_ARCHES)' | tr ',' '\n' \
	| grep -Ev '^sm_[0-9]+$$' | head -n 1; fi)
AMD_ARCH_INVALID := $(shell if [ -n "$(AMD_ARCH)" ] \
	&& ! printf '%s\n' '$(AMD_ARCH)' | grep -Eq '^gfx[0-9a-f]+$$'; then \
	printf '%s' '$(AMD_ARCH)'; fi)

NVIDIA_MISSING_ARCH_MESSAGE := NVIDIA arch detection failed; rerun with \
	make CC=clang GPU=NVIDIA NVIDIA_ARCH=sm_xy

$(if $(strip $(NVIDIA_INVALID_ARCH)), \
	$(error NVIDIA_ARCH must be a comma-separated list of \
	sm_xy targets; invalid entry '$(NVIDIA_INVALID_ARCH)'))

$(if $(strip $(AMD_ARCH_INVALID)), \
	$(error AMD_ARCH must match gfx<target>, e.g. gfx90a; \
	got '$(AMD_ARCH_INVALID)'))

ifeq ($(GPU),NONE)
ifneq ($(strip $(NVIDIA_ARCH)$(AMD_ARCH)),)
$(error GPU=NONE does not accept NVIDIA_ARCH or AMD_ARCH overrides)
endif
else ifeq ($(GPU),NVIDIA)
ifneq ($(strip $(AMD_ARCH)),)
$(error GPU=NVIDIA does not accept AMD_ARCH; use NVIDIA_ARCH=sm_xy[,sm_ab,...])
endif
else ifeq ($(GPU),AMD)
ifneq ($(strip $(NVIDIA_ARCH)),)
$(error GPU=AMD does not accept NVIDIA_ARCH; use AMD_ARCH=gfx_target)
endif
endif

ifeq ($(filter $(GPU),NVIDIA AMD),$(GPU))
ifeq ($(strip $(NVIDIA_ARCH)$(AMD_ARCH)),)
$(info No GPU arch specified; set NVIDIA_ARCH=... or AMD_ARCH=... to override.)
endif
endif

ifeq ($(GPU),NVIDIA)
ifneq ($(strip $(NVIDIA_EFFECTIVE_ARCHES)),)
$(info NVIDIA device images ($(NVIDIA_SELECTION_MODE)): \
	$(NVIDIA_EFFECTIVE_ARCHES))
ifneq ($(strip $(NVIDIA_ARCH)),)
ifneq ($(strip $(NVIDIA_MISSING_VISIBLE_ARCHES)),)
$(warning Explicit NVIDIA_ARCH=$(NVIDIA_ARCH) omits visible GPU \
architectures $(subst $(space),$(comma),$(NVIDIA_MISSING_VISIBLE_ARCHES)); \
running on those GPUs can fail with 'No images found compatible'. \
Include all required SMs or set CUDA_VISIBLE_DEVICES to a compatible GPU)
endif
endif
else
$(error $(NVIDIA_MISSING_ARCH_MESSAGE))
endif
endif

ifeq ($(GPU),AMD)
ifneq ($(strip $(AMD_ARCH)),)
BUG_GPU_ARCH_TAG := $(AMD_ARCH)
else
BUG_GPU_ARCH_TAG := default
endif
else ifeq ($(GPU),NVIDIA)
ifneq ($(strip $(NVIDIA_EFFECTIVE_ARCHES)),)
BUG_GPU_ARCH_TAG := $(subst $(comma),-,$(NVIDIA_EFFECTIVE_ARCHES))
else
BUG_GPU_ARCH_TAG := default
endif
else
BUG_GPU_ARCH_TAG := cpu
endif
BUG_REPORT_TAG ?= $(notdir $(CC))-$(GPU)-$(BUG_GPU_ARCH_TAG)

NVIDIA_CLANG_ARCH_FLAGS :=
ifneq ($(strip $(NVIDIA_ARCH_LIST)),)
NVIDIA_CLANG_ARCH_FLAGS := \
	$(foreach arch,$(NVIDIA_ARCH_LIST),--offload-arch=$(arch))
endif
NVIDIA_GCC_ARCH_FLAG :=
ifneq ($(strip $(NVIDIA_ARCH_LIST)),)
ifeq ($(notdir $(CC)),gcc)
ifneq ($(word 2,$(NVIDIA_ARCH_LIST)),)
$(error CC=gcc supports one NVIDIA device image in this Makefile; \
pass a single NVIDIA_ARCH=sm_xy)
endif
NVIDIA_GCC_ARCH_FLAG := \
	-foffload-options=nvptx-none=-march=$(firstword $(NVIDIA_ARCH_LIST))
endif
endif

# Set the appropriate OpenMP flag based on compiler
ifeq ($(notdir $(CC)),gcc)
OPENMP_FLAG = -fopenmp
OPENMP_LINK_LIBS = -lgomp
OPENMP_LINK_PRE = -Wl,--no-as-needed
OPENMP_LINK_POST = -Wl,--as-needed
else ifeq ($(notdir $(CC)),clang)
OPENMP_FLAG = -fopenmp
else
$(error Unsupported compiler $(CC); use gcc or clang)
endif

ifeq ($(IS_CLEAN_GOAL),)
$(info GPU is set to $(GPU))
endif

# Set the appropriate offload flag based on GPU type
ifeq ($(GPU),NONE)
ifeq ($(notdir $(CC)),gcc)
OFFLOAD_FL = -foffload=disable
else
OFFLOAD_FL =
endif
DEFINES += -DNOGPU
else ifeq ($(GPU),NVIDIA)
ifeq ($(notdir $(CC)),gcc)
$(warning GPU=NVIDIA with CC=gcc is experimental; reliable fat-binary \
or multi-arch offload is not guaranteed)
OFFLOAD_FL = -foffload=nvptx-none $(NVIDIA_GCC_ARCH_FLAG)
HOST_PROTECTION_FLAGS = -fno-stack-protector -fcf-protection=none
else ifeq ($(notdir $(CC)),clang)
ifeq ($(strip $(NVIDIA_CLANG_ARCH_FLAGS)),)
$(error $(NVIDIA_MISSING_ARCH_MESSAGE))
endif
OFFLOAD_FL = -fopenmp-targets=nvptx64-nvidia-cuda \
	--cuda-path=$(CUDA_PATH) $(NVIDIA_CLANG_ARCH_FLAGS) \
	-Xopenmp-target=nvptx64-nvidia-cuda --no-cuda-version-check \
	-foffload-lto
else
$(error GPU=NVIDIA is supported with gcc or clang only)
endif
else ifeq ($(GPU),AMD)
AMD_OPENMP_RUNTIME_REQUIRED := 1
ifneq ($(filter-out hip_gpu_bench build/hip_gpu_benchmark,$(MAKECMDGOALS)),)
AMD_OPENMP_RUNTIME_REQUIRED := 1
else ifneq ($(filter hip_gpu_bench build/hip_gpu_benchmark,$(MAKECMDGOALS)),)
AMD_OPENMP_RUNTIME_REQUIRED :=
endif
ifeq ($(notdir $(CC)),gcc)
$(warning GPU=AMD with CC=gcc can trigger GCC amdgcn internal \
compiler errors on this setup; prefer CC=clang)
OFFLOAD_FL = -foffload=amdgcn-amdhsa
HOST_PROTECTION_FLAGS = -fno-stack-protector -fcf-protection=none
ifneq ($(strip $(AMD_ARCH)),)
OFFLOAD_FL += -foffload-options=amdgcn-amdhsa="-march=$(AMD_ARCH)"
endif
else ifeq ($(notdir $(CC)),clang)
OFFLOAD_FL = -fopenmp-targets=amdgcn-amd-amdhsa \
	--rocm-path=$(ROCM_PATH) \
	--rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH)
ifneq ($(strip $(AMD_ARCH)),)
OFFLOAD_FL += -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AMD_ARCH)
AMD_DEVICE_RTL_FILE := $(firstword \
	$(wildcard /usr/lib/llvm-*/lib/libomptarget-amdgpu-$(AMD_ARCH).bc) \
	$(wildcard $(ROCM_PATH)/lib/llvm/lib/libomptarget-amdgpu-$(AMD_ARCH).bc) \
	$(wildcard $(ROCM_PATH)/lib/llvm/lib/libomptarget-500-amdgpu-$(AMD_ARCH).bc) \
)
ifneq ($(strip $(AMD_OPENMP_RUNTIME_REQUIRED)),)
ifeq ($(strip $(AMD_DEVICE_RTL_FILE)),)
$(error No OpenMP AMD device runtime was found for \
AMD_ARCH=$(AMD_ARCH). Install LLVM/ROCm support for this gfx target \
or override AMD_ARCH to a supported value)
endif
endif
endif
else
$(error GPU=AMD is not supported with CC=$(CC); use gcc or clang)
endif
else
$(error GPU=$(GPU) is not one of the supported GPUs (NONE, NVIDIA, AMD))
endif
# Allow custom build directory, default to 'build' folder
BUILD_DIR ?= build

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

## compiler flags
CFLAGS0 = -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=199309L -std=c11 \
	-fPIC -O3 -march=native \
-Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable

REPORT_CFLAGS :=
ifeq ($(BUG_REPORT),1)
ifeq ($(notdir $(CC)),gcc)
REPORT_CFLAGS += -freport-bug
else ifeq ($(notdir $(CC)),clang)
REPORT_CFLAGS += -gen-reproducer \
	-fcrash-diagnostics-dir=$(abspath $(BUG_REPORT_OUT))
endif
endif

HOST_PROTECTION_FLAGS ?=

CFLAGS += $(DEFINES) $(OPENMP_FLAG) $(OFFLOAD_FL) $(HOST_PROTECTION_FLAGS) $(CFLAGS0) $(REPORT_CFLAGS)
HOST_ONLY_CFLAGS := $(filter-out $(OFFLOAD_FL),$(CFLAGS))

COMPILE_CMD = $(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@
HOST_COMPILE_CMD = $(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -c $< -o $@
BIT_LINK_DIRS = -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG)
HOST_OPENMP_LIBS = $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST)
BIT_LINK_LIBS = $(BIT_LINK_DIRS) -lbit $(HOST_OPENMP_LIBS)
BIT_OPENMP_LINK_LIBS = $(BIT_LINK_DIRS) -lbit $(OPENMP_FLAG) $(HOST_OPENMP_LIBS)
BENCH_OMP_NO_CPU_EXTRA_LINK_LIBS =
ifeq ($(notdir $(CC)),clang)
BENCH_OMP_NO_CPU_EXTRA_LINK_LIBS += $(OPENMP_LINK_PRE) -lomptarget $(OPENMP_LINK_POST)
endif
SHARED_LIB_LINK_CMD = $(CC_ENV) $(CC) $(CFLAGS) -shared -o $@ $^ \
	$(LDFLAGS) $(OMPTARGET_RPATH_FLAG) $(HOST_OPENMP_LIBS)
BUG_REPORT_SCRIPT = scripts/generate_bug_report.sh

# Runtime search paths - occasionally you have to do this
BUILD_RPATH_FLAG := -Wl,-rpath,$(shell pwd)/$(BUILD_DIR)
OMPTARGET_LIBDIR :=
ifeq ($(notdir $(CC)),clang)
CLANG_LIBRARY_DIRS := $(subst :, ,$(patsubst libraries: =%,%, \
	$(shell $(CC) -print-search-dirs 2>/dev/null | grep '^libraries: =')))
OMPTARGET_LIBDIR := $(firstword $(foreach d,$(CLANG_LIBRARY_DIRS), \
	$(if $(wildcard $(d)/libomptarget.so*),$(d),)))
endif
OMPTARGET_RPATH_FLAG :=
ifneq ($(strip $(OMPTARGET_LIBDIR)),)
OMPTARGET_RPATH_FLAG := -Wl,-rpath,$(OMPTARGET_LIBDIR)
endif

# Rebuild objects when toolchain/config changes.
# Make does not automatically notice changes to variables like GPU/CC/CFLAGS,
# so without this you can get a stale $(BUILD_DIR)/bit.o built with offload
# flags from a previous invocation.
.PHONY: FORCE
CONFIG_STAMP := $(BUILD_DIR)/.config.stamp

# Treat the config stamp as a temporary build artifact.
# This keeps `build/` clean; note it will be regenerated on the next `make`.
.INTERMEDIATE: $(CONFIG_STAMP)

$(CONFIG_STAMP): FORCE
	@mkdir -p $(BUILD_DIR)
	@{ \
		echo "CC=$(CC)"; \
		echo "GPU=$(GPU)"; \
		echo "NVIDIA_ARCH=$(NVIDIA_ARCH)"; \
		echo "NVIDIA_ARCH_POLICY=$(NVIDIA_ARCH_POLICY)"; \
		echo "NVIDIA_SELECTION_MODE=$(NVIDIA_SELECTION_MODE)"; \
		echo "NVIDIA_SYSTEM_ARCHES=$(NVIDIA_SYSTEM_ARCHES)"; \
		echo "NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST=" \
		"$(NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST)"; \
		echo "NVIDIA_COMPILER_SUPPORTED_ARCH_LIST=" \
		"$(NVIDIA_COMPILER_SUPPORTED_ARCH_LIST)"; \
		echo "NVIDIA_EFFECTIVE_ARCHES=$(NVIDIA_EFFECTIVE_ARCHES)"; \
		echo "DEFINES=$(DEFINES)"; \
		echo "ROCM_PATH=$(ROCM_PATH)"; \
		echo "ROCM_DEVICE_LIB_PATH=$(ROCM_DEVICE_LIB_PATH)"; \
		echo "OPENMP_FLAG=$(OPENMP_FLAG)"; \
		echo "OFFLOAD_FL=$(OFFLOAD_FL)"; \
		echo "OMPTARGET_LIBDIR=$(OMPTARGET_LIBDIR)"; \
		echo "CFLAGS=$(CFLAGS)"; \
	} > $(CONFIG_STAMP).tmp
	@cmp -s $(CONFIG_STAMP).tmp $(CONFIG_STAMP) 2>/dev/null \
		|| mv $(CONFIG_STAMP).tmp $(CONFIG_STAMP)
	@rm -f $(CONFIG_STAMP).tmp


# Default to enabled
LIBPOPCNT ?= 1

# Convert to lowercase for case-insensitive comparison
LIBPOPCNT_LC := $(shell echo $(LIBPOPCNT) | tr A-Z a-z)

# Check if it's one of the "false" values
ifneq ($(filter $(LIBPOPCNT_LC),0 no n false f off),)
    # LIBPOPCNT is disabled - don't add the flag
    $(info libpopcnt integration disabled)
else
    # LIBPOPCNT is enabled
    CFLAGS += -DUSE_LIBPOPCNT
    ifeq ($(filter clean test bench distclean,$(MAKECMDGOALS)),)
        $(info Using libpopcnt for population count)
    endif
endif

SRC = src/bit.c
OBJ = $(BUILD_DIR)/bit.o
# Change from static to shared library
TARGET = $(BUILD_DIR)/libbit.so
TARGET_STATIC = $(BUILD_DIR)/libbit.a
TEST_SRC = tests/test_bit.c
TEST_OBJ = $(BUILD_DIR)/test_bit.o
TEST_EXEC = $(BUILD_DIR)/test_bit
TEST_OFFLOAD_SRC = tests/test_offload.c
TEST_OFFLOAD_OBJ = $(BUILD_DIR)/test_offload.o
TEST_OFFLOAD_EXEC = $(BUILD_DIR)/test_offload

# Add benchmark source and executable
BENCH_SRC = benchmark/benchmark.c
BENCH_OBJ = $(BUILD_DIR)/benchmark.o
BENCH_EXEC = $(BUILD_DIR)/benchmark

# Add OpenMP benchmark source and executable
BENCH_OMP_SRC = benchmark/openmp_bit.c
BENCH_OMP_OBJ = $(BUILD_DIR)/openmp_bit.o
BENCH_OMP_EXEC = $(BUILD_DIR)/openmp_bit

BENCH_OMP_NO_GPU_SRC = benchmark/openmp_bit_nogpu.c
BENCH_OMP_GPU_OBJ = $(BUILD_DIR)/openmp_bit_nogpu.o
BENCH_OMP_GPU_EXEC = $(BUILD_DIR)/openmp_bit_nogpu

BENCH_OMP_NO_CPU_SRC = benchmark/openmp_bit_nocpu.c
BENCH_OMP_NO_CPU_OBJ = $(BUILD_DIR)/openmp_bit_nocpu.o
BENCH_OMP_NO_CPU_BIT_OBJ = $(BUILD_DIR)/bit_nocpu_host.o
BENCH_OMP_NO_CPU_EXEC = $(BUILD_DIR)/openmp_bit_nocpu

UNIFIED_GPU_BENCH_SRC = benchmark/unified_gpu_benchmark.c
UNIFIED_GPU_BENCH_OBJ = $(BUILD_DIR)/unified_gpu_benchmark.o
UNIFIED_GPU_BENCH_EXEC = $(BUILD_DIR)/unified_gpu_benchmark

COMPARE_SIZE ?= 1024
CUDA_BENCH_SRC = benchmark/cuda_gpu_benchmark.cu
CUDA_BENCH_EXEC = $(BUILD_DIR)/cuda_gpu_benchmark
NVCC ?= nvcc
NVCC_ARCH ?= sm_70
NVCC_FLAGS ?= -O3 -std=c++14 -I./include -I./src

HIP_BENCH_SRC = benchmark/hip_gpu_benchmark.cpp
HIP_BENCH_EXEC = $(BUILD_DIR)/hip_gpu_benchmark
HIPCC ?= hipcc
HIPCC_FLAGS ?= -O3 -std=c++17 -I./include -I./src

USE_BUILTIN_POPCOUNT ?= 0
GPU_OCCUPANCY_TUNING ?= 1
## OpenMP Offload GPU implementation selection for openmp_bit_nocpu
## Mode options:
##   TEAM_PARALLEL_SIMD - current hierarchical teams -> parallel -> simd path
##   FLAT_COLLAPSE      - current flat collapse(2) path using OMP_GPU_FLAT
OPENMP_GPU_IMPL ?= TEAM_PARALLEL_SIMD

ifeq ($(strip $(USE_BUILTIN_POPCOUNT)),1)
NVCC_FLAGS += -DUSE_BUILTIN_POPCOUNT
HIPCC_FLAGS += -DUSE_BUILTIN_POPCOUNT
endif

ifeq ($(strip $(GPU_OCCUPANCY_TUNING)),1)
NVCC_FLAGS += -DGPU_OCCUPANCY_TUNING
HIPCC_FLAGS += -DGPU_OCCUPANCY_TUNING
endif

ifeq ($(strip $(OPENMP_GPU_IMPL)),TEAM_PARALLEL_SIMD)
CFLAGS += -DOPENMP_GPU_IMPL_TEAM_PARALLEL_SIMD
else ifeq ($(strip $(OPENMP_GPU_IMPL)),FLAT_COLLAPSE)
CFLAGS += -DOPENMP_GPU_IMPL_FLAT_COLLAPSE
else
$(error OPENMP_GPU_IMPL must be TEAM_PARALLEL_SIMD or FLAT_COLLAPSE)
endif

# Keep compare_nocpu architecture behavior aligned with bench_omp logic.
NVCC_ARCH_EFFECTIVE := $(if $(strip $(NVIDIA_ARCH_LIST)),$(firstword $(NVIDIA_ARCH_LIST)),$(NVCC_ARCH))

NVCC_ARCH_LIST :=
ifneq ($(strip $(NVIDIA_ARCH_LIST)),)
NVCC_ARCH_LIST := $(NVIDIA_ARCH_LIST)
ifneq ($(strip $(NVIDIA_SYSTEM_ARCH_LIST)),)
NVCC_ARCH_LIST := $(call dedup_words,$(NVIDIA_ARCH_LIST) $(NVIDIA_SYSTEM_ARCH_LIST))
endif
else
NVCC_ARCH_LIST := $(NVIDIA_SYSTEM_ARCH_LIST)
endif

NVCC_ARCH_FLAGS :=
ifneq ($(strip $(NVCC_ARCH_LIST)),)
ifneq ($(words $(NVCC_ARCH_LIST)),1)
NVCC_ARCH_FLAGS := $(foreach arch,$(NVCC_ARCH_LIST),-gencode=arch=compute_$(subst sm_,,$(arch)),code=$(arch))
else
NVCC_ARCH_FLAGS := -arch=$(firstword $(NVCC_ARCH_LIST))
endif
else
NVCC_ARCH_FLAGS := -arch=$(NVCC_ARCH_EFFECTIVE)
endif
HIPCC_ARCH_FLAGS :=
ifneq ($(strip $(AMD_ARCH)),)
HIPCC_ARCH_FLAGS := $(foreach arch,$(subst $(comma), ,$(AMD_ARCH)),--offload-arch=$(arch))
endif
COMPARE_NUM_BITS ?= 64
COMPARE_NUM_REF_BITS ?= 64
COMPARE_GPU_ITERATIONS ?= 2
COMPARE_GPU_ID ?= 0
COMPARE_BACKEND ?= AUTO
COMPARE_BACKENDS_ENV ?= BIT_COMPARE_BACKENDS=1
COMPARE_RUN_ENV ?= OMP_TARGET_OFFLOAD=MANDATORY
COMPARE_VISIBLE_DEVICE_ENV :=
ifeq ($(GPU),NVIDIA)
COMPARE_VISIBLE_DEVICE_ENV := CUDA_VISIBLE_DEVICES=$(COMPARE_GPU_ID)
endif

ifeq ($(COMPARE_BACKEND),AUTO)
ifeq ($(GPU),NVIDIA)
COMPARE_BACKEND_EFFECTIVE := CUDA
else ifeq ($(GPU),AMD)
COMPARE_BACKEND_EFFECTIVE := HIP
else
COMPARE_BACKEND_EFFECTIVE := AUTO
endif
else
COMPARE_BACKEND_EFFECTIVE := $(COMPARE_BACKEND)
endif

COMPARE_BACKEND_TARGETS :=
ifeq ($(COMPARE_BACKEND_EFFECTIVE),CUDA)
COMPARE_BACKEND_TARGETS += $(CUDA_BENCH_EXEC)
else ifeq ($(COMPARE_BACKEND_EFFECTIVE),HIP)
COMPARE_BACKEND_TARGETS += $(HIP_BENCH_EXEC)
else
COMPARE_BACKEND_TARGETS += $(CUDA_BENCH_EXEC) $(HIP_BENCH_EXEC)
endif

.PHONY: all clean test test_offload bench compare_nocpu unified_gpu_bench LIBPOPCNT bug_report
.PHONY: cuda_gpu_bench hip_gpu_bench

# Default targets are to build the shared and static libraries
all: $(TARGET) $(TARGET_STATIC)

# Rule to build object files in the build directory
$(BUILD_DIR)/%.o: src/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/%.o: tests/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

# Add pattern rule for benchmark source files
$(BUILD_DIR)/%.o: benchmark/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

# Build OpenMP harness benchmark objects without OFFLOAD_FL so only libbit
# contributes target images (avoids duplicate clang offload bundles).
$(BENCH_OBJ): $(BENCH_SRC) $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(BENCH_OMP_OBJ): $(BENCH_OMP_SRC) $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(BENCH_OMP_GPU_OBJ): $(BENCH_OMP_NO_GPU_SRC) $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(BENCH_OMP_NO_CPU_OBJ): $(BENCH_OMP_NO_CPU_SRC) $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BENCH_OMP_NO_CPU_BIT_OBJ): $(SRC) $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

$(UNIFIED_GPU_BENCH_OBJ): $(UNIFIED_GPU_BENCH_SRC) $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BENCH_OMP_EXEC): $(TARGET) $(BENCH_OMP_OBJ)
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $@ $(BENCH_OMP_OBJ) \
		$(BIT_OPENMP_LINK_LIBS) -lrt

$(BENCH_OMP_GPU_EXEC): $(TARGET) $(BENCH_OMP_GPU_OBJ)
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $@ $(BENCH_OMP_GPU_OBJ) \
		$(BIT_OPENMP_LINK_LIBS) -lrt

$(BENCH_OMP_NO_CPU_EXEC): $(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $@ \
		$(BENCH_OMP_NO_CPU_OBJ) $(BENCH_OMP_NO_CPU_BIT_OBJ) \
		$(OMPTARGET_RPATH_FLAG) $(OPENMP_FLAG) $(HOST_OPENMP_LIBS) \
		$(BENCH_OMP_NO_CPU_EXTRA_LINK_LIBS) -lrt

$(UNIFIED_GPU_BENCH_EXEC): $(OBJ) $(UNIFIED_GPU_BENCH_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $@ $(UNIFIED_GPU_BENCH_OBJ) $(OBJ) \
		$(OMPTARGET_RPATH_FLAG) $(OPENMP_FLAG) $(HOST_OPENMP_LIBS) -lrt

$(CUDA_BENCH_EXEC): $(CUDA_BENCH_SRC)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH_FLAGS) -o $@ $< -lm

$(HIP_BENCH_EXEC): $(HIP_BENCH_SRC)
	$(HIPCC) $(HIPCC_FLAGS) $(HIPCC_ARCH_FLAGS) -o $@ $< -lm

# Change from static to shared library compilation
$(TARGET): $(OBJ)
	$(SHARED_LIB_LINK_CMD)

# Build the static library as well
$(TARGET_STATIC): $(OBJ)
	$(AR) rcs $@ $^
	
# Update test to use shared library
test: $(TARGET) $(TEST_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJ) $(BIT_LINK_LIBS)

# test_offload uses the same offload flags as the main library
# You can pass NVIDIA_ARCH or AMD_ARCH when invoking test_offload
test_offload: $(TEST_OFFLOAD_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_OFFLOAD_EXEC) \
		$(TEST_OFFLOAD_OBJ) $(OMPTARGET_RPATH_FLAG) \
		$(OPENMP_FLAG) $(HOST_OPENMP_LIBS) -lm


# Add target to build the benchmark executable and the OpenMP benchmark
bench: $(TARGET) $(BENCH_OBJ) bench_omp
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $(BENCH_EXEC) $(BENCH_OBJ) $(BIT_LINK_LIBS) -lrt

# Conditional bench_omp target based on GPU setting
ifeq ($(GPU),NONE)
bench_omp: $(BENCH_OMP_GPU_EXEC)
else
bench_omp: $(BENCH_OMP_EXEC) $(BENCH_OMP_GPU_EXEC) $(BENCH_OMP_NO_CPU_EXEC)
endif

compare_nocpu: $(BENCH_OMP_NO_CPU_EXEC) $(COMPARE_BACKEND_TARGETS)
	$(COMPARE_BACKENDS_ENV) BIT_COMPARE_BACKEND=$(COMPARE_BACKEND_EFFECTIVE) \
		BIT_GPU_TARGET=$(GPU) \
		$(COMPARE_VISIBLE_DEVICE_ENV) $(COMPARE_RUN_ENV) ./$(BENCH_OMP_NO_CPU_EXEC) \
		$(COMPARE_SIZE) $(COMPARE_NUM_BITS) $(COMPARE_NUM_REF_BITS) \
		$(COMPARE_GPU_ITERATIONS) $(COMPARE_GPU_ID)

unified_gpu_bench: $(UNIFIED_GPU_BENCH_EXEC)
	$(COMPARE_VISIBLE_DEVICE_ENV) $(COMPARE_RUN_ENV) ./$(UNIFIED_GPU_BENCH_EXEC) \
		1024 1024 1024 5 $(COMPARE_GPU_ID)
cuda_gpu_bench: $(CUDA_BENCH_EXEC)
	./$(CUDA_BENCH_EXEC) 1024 1024 1024 5 0

hip_gpu_bench: $(HIP_BENCH_EXEC)
	./$(HIP_BENCH_EXEC) 1024 1024 1024 5 0


bug_report:
	@REPORT_PATH="$(BUG_REPORT_DIR)/$$(date +%Y%m%d-%H%M%S)-$(BUG_REPORT_TAG)" \
	CC="$(CC)" \
	GPU="$(GPU)" \
	NVIDIA_ARCH="$(NVIDIA_ARCH)" \
	NVIDIA_ARCH_POLICY="$(NVIDIA_ARCH_POLICY)" \
	NVIDIA_SELECTION_MODE="$(NVIDIA_SELECTION_MODE)" \
	NVIDIA_SYSTEM_ARCHES="$(NVIDIA_SYSTEM_ARCHES)" \
	NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST="$(NVIDIA_VISIBLE_SUPPORTED_ARCH_LIST)" \
	NVIDIA_COMPILER_SUPPORTED_ARCH_LIST="$(NVIDIA_COMPILER_SUPPORTED_ARCH_LIST)" \
	NVIDIA_EFFECTIVE_ARCHES="$(NVIDIA_EFFECTIVE_ARCHES)" \
	AMD_ARCH="$(AMD_ARCH)" \
	CUDA_PATH="$(CUDA_PATH)" \
	OPENMP_FLAG="$(OPENMP_FLAG)" \
	OFFLOAD_FL='$(OFFLOAD_FL)' \
	BUG_TARGET="$(BUG_TARGET)" \
	CFLAGS='$(CFLAGS)' \
	CFLAGS0='$(CFLAGS0)' \
	DEFINES='$(DEFINES)' \
	CC_ENV='$(CC_ENV)' \
	BUILD_DIR="$(BUILD_DIR)" \
	MAKE="$(MAKE)" \
	$(BUG_REPORT_SCRIPT)


clean:
	@rm -rf $(BUILD_DIR)
    
# Additional target to clean everything including dependencies
distclean: clean
