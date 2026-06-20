# ----------------------------------------------------------------------------
# CORE BUILD MAKEFILE - MULTI-GPU & FAT BINARY EXTENSION
# ----------------------------------------------------------------------------

# =============================================================
# USER CONFIGURATION: PATHS AND DIRECTORIES
# =============================================================

ROCM_PATH ?= /opt/rocm
ROCM_DEVICE_LIB_PATH ?= $(ROCM_PATH)/amdgcn/bitcode
ROCM_LLVM_BIN ?= $(ROCM_PATH)/lib/llvm/bin
ROCM_LLVM_LIB_PATH   ?= $(ROCM_PATH)/lib/llvm/lib
CUDA_PATH ?= /usr/lib/cuda

# =============================================================
# INTERNAL UTILITY VARIABLES
# =============================================================

empty :=
space := $(empty) $(empty)
comma := ,
ERRORS :=

define newline


endef

define APPEND_ERROR
  ERRORS += |$(strip $(1)$(if $(2),$(comma)$(2))$(if $(3),$(comma)$(3))$(if $(4),$(comma)$(4)))
endef

# =============================================================
# Early Setup
# =============================================================

IS_CLEAN_GOAL := $(filter clean distclean,$(MAKECMDGOALS))
.DEFAULT_GOAL := all

LIBOMPTARGET_INFO  ?= 0
LIBOMPTARGET_DEBUG ?= 0
export LIBOMPTARGET_INFO LIBOMPTARGET_DEBUG

PREFERRED_CC  := amdclang clang gcc icx
PREFERRED_CXX := amdclang++ clang++ g++ icpx

# =============================================================
# Compiler Auto-Detection & Validation
# =============================================================

ifeq ($(IS_CLEAN_GOAL),)
  ifeq ($(origin CC),default)
    DETECTED_CC := $(firstword $(foreach c,$(PREFERRED_CC),$(shell which $(c) 2>/dev/null)))
    ifeq ($(strip $(DETECTED_CC)),)
     $(eval $(call APPEND_ERROR,No suitable compiler found. Install one of: $(PREFERRED_CC) or set CC= explicitly))
    else
      CC := $(DETECTED_CC)
    endif
  endif

  override CC_BASENAME := $(notdir $(shell printf '%s' '$(CC)' | tr 'A-Z' 'a-z' | tr -d '[:space:]'))
  override CC          := $(CC_BASENAME)

  GPU ?= NONE
  override GPU_LIST := $(subst $(comma),$(space),$(shell printf '%s' '$(GPU)' | tr 'a-z' 'A-Z' | tr -d '[:space:]'))

  ifneq ($(filter NONE,$(GPU_LIST)),)
    ifneq ($(words $(GPU_LIST)),1)
      $(eval $(call APPEND_ERROR,GPU=none cannot be combined with active offloading targets. Got: GPU=$(GPU)))
    endif
  endif

  ifeq ($(filter $(addsuffix %,$(PREFERRED_CC)),$(CC_BASENAME)),)
    $(eval $(call APPEND_ERROR,CC=$(CC) is not supported. Allowed compilers: $(PREFERRED_CC)))
  endif
endif

DEFINES ?=
CC_ENV :=
BUG_REPORT ?= 0
BUG_REPORT_DIR ?= bug_reports
BUG_REPORT_OUT ?= $(BUG_REPORT_DIR)
BUG_TARGET ?= bench_omp
BUG_REPORT_SCRIPT := scripts/generate_bug_report.sh

GPU_ARCH ?=
override GPU_ARCH := $(shell printf '%s' '$(GPU_ARCH)' | tr 'A-Z' 'a-z' | tr -d '[:space:]')
override NVIDIA_ARCH := $(shell printf '%s' '$(NVIDIA_ARCH)' | tr 'A-Z' 'a-z' | tr -d '[:space:]')
override AMD_ARCH := $(shell printf '%s' '$(AMD_ARCH)' | tr 'A-Z' 'a-z' | tr -d '[:space:]')
ALL_ARCHS_PARSED := $(subst $(comma),$(space),$(strip $(GPU_ARCH) $(NVIDIA_ARCH) $(AMD_ARCH)))

NVIDIA_ARCH_LIST := $(filter sm_% compute_%,$(ALL_ARCHS_PARSED))
AMD_ARCH_LIST    := $(filter gfx%,$(ALL_ARCHS_PARSED))

UNSUPPORTED_ARCHS := $(filter-out sm_% compute_% gfx%,$(ALL_ARCHS_PARSED))
ifneq ($(strip $(UNSUPPORTED_ARCHS)),)
  $(eval $(call APPEND_ERROR,Invalid or unsupported architecture(s) detected: $(UNSUPPORTED_ARCHS). Supported prefixes are sm_, compute_, and gfx))
endif

NVIDIA_ARCH_INVALID := $(foreach arch,$(NVIDIA_ARCH_LIST),$(shell if ! printf '%s\n' "$(arch)" | grep -Eq '^(sm|compute)_[0-9]+$$'; then printf '%s' "$(arch)"; fi))
ifneq ($(strip $(NVIDIA_ARCH_INVALID)),)
  $(eval $(call APPEND_ERROR,NVIDIA architecture targets must match sm_<target> or compute_<target> with numeric suffixes, e.g., sm_70; got '$(strip $(NVIDIA_ARCH_INVALID))'))
endif

AMD_ARCH_INVALID := $(foreach arch,$(AMD_ARCH_LIST),$(shell if ! printf '%s\n' "$(arch)" | grep -Eq '^gfx[0-9a-f]+$$'; then printf '%s' "$(arch)"; fi))
ifneq ($(strip $(AMD_ARCH_INVALID)),)
  $(eval $(call APPEND_ERROR,AMD architecture targets must match gfx<target>, e.g. gfx90a; got '$(strip $(AMD_ARCH_INVALID))'))
endif

# NVIDIA Auto detection using nvidia-smi
NVIDIA_SYSTEM_ARCHES := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk -F '.' '{gsub(/ /,""); print "sm_"$$1$$2}' | sort -u 2>/dev/null || true)
ifeq ($(filter NVIDIA,$(GPU_LIST)),NVIDIA)
  ifeq ($(strip $(NVIDIA_ARCH_LIST)),)
    NVIDIA_ARCH_LIST := $(strip $(NVIDIA_SYSTEM_ARCHES))
  endif
  ifeq ($(strip $(NVIDIA_ARCH_LIST)),)
    $(eval $(call APPEND_ERROR,NVIDIA was requested but no targets were provided via GPU_ARCH, and system-level auto-detection failed))
  endif
endif

# AMD Auto-detection using roc-smi
AMD_SYSTEM_ARCHES := $(shell rocm-smi --showproductname 2>/dev/null | grep 'GFX Version' | grep -o 'gfx[0-9a-f]*' | sort -u 2>/dev/null || true)
ifeq ($(filter AMD,$(GPU_LIST)),AMD)
  ifeq ($(strip $(AMD_ARCH_LIST)),)
    AMD_ARCH_LIST := $(strip $(AMD_SYSTEM_ARCHES))
  endif
  ifeq ($(strip $(AMD_ARCH_LIST)),)
    $(eval $(call APPEND_ERROR,AMD was requested but no targets were provided via GPU_ARCH, and system-level auto-detection failed))
  endif
endif

ifeq ($(IS_CLEAN_GOAL),)
  $(info Active GPU execution targets: $(GPU_LIST))
  ifneq ($(strip $(NVIDIA_ARCH_LIST)),)
    $(info Resolved NVIDIA Architectures: $(NVIDIA_ARCH_LIST))
  endif
  ifneq ($(strip $(AMD_ARCH_LIST)),)
    $(info Resolved AMD Architectures: $(AMD_ARCH_LIST))
  endif
endif

CC_BASE := $(notdir $(CC))
COMPILER_ID := $(strip $(firstword \
  $(if $(filter amdclang%,$(CC_BASE)),amdclang) \
  $(if $(filter clang%,$(CC_BASE)),clang) \
  $(if $(filter gcc%,$(CC_BASE)),gcc) \
  $(if $(filter icx%,$(CC_BASE)),icx) \
  unknown))

ifeq ($(filter INTEL,$(GPU_LIST)),INTEL)
  ifneq ($(COMPILER_ID),icx)
    $(eval $(call APPEND_ERROR,GPU=Intel requires the 'icx' compiler, but CC=$(CC) was used))
  endif
endif

ifeq ($(COMPILER_ID),amdclang)
  ifneq ($(strip $(GPU_LIST)),AMD)
    $(eval $(call APPEND_ERROR,CC=amdclang is active, but GPU_LIST is '$(GPU_LIST)'. amdclang exclusively requires GPU=AMD))
  endif
endif

# =====================================================================
# CONSTRUCT UNIFIED OFFLOADING PIPELINE FLAGS
# =====================================================================

OFFLOAD_FL :=
OPENMP_FLAG := -fopenmp

# Host OpenMP Libs (Only used for NVCC/HIPCC to link OpenMP helper objects)
ifeq ($(findstring clang,$(COMPILER_ID)),clang)
  HOST_OPENMP_LIBS := -L$(ROCM_LLVM_LIB_PATH) -lomp
else ifeq ($(COMPILER_ID),icx)
  OPENMP_FLAG := -fiopenmp
  HOST_OPENMP_LIBS := -liomp5
else
  # GCC fallback
  HOST_OPENMP_LIBS := -lgomp
endif

ifeq ($(filter NONE,$(GPU_LIST)),NONE)
  ifeq ($(COMPILER_ID),gcc)
    OFFLOAD_FL += -foffload=disable
  endif
  DEFINES += -DNOGPU
endif

ifeq ($(COMPILER_ID),gcc)
  OFFLOAD_FL += -fcf-protection=none -fno-stack-protector
  ifeq ($(filter NVIDIA,$(GPU_LIST)),NVIDIA)
    NVIDIA_GCC_ARCH_FLAG := $(if $(strip $(NVIDIA_ARCH_LIST)),-foffload-options=nvptx-none=-march=$(firstword $(sort $(NVIDIA_ARCH_LIST))),)
    OFFLOAD_FL += -foffload=nvptx-none $(NVIDIA_GCC_ARCH_FLAG)
  endif
  ifeq ($(filter AMD,$(GPU_LIST)),AMD)
    OFFLOAD_FL += -foffload=amdgcn-amdhsa $(foreach arch,$(AMD_ARCH_LIST),-foffload-options=amdgcn-amdhsa="-march=$(arch)")
  endif
endif

ifeq ($(COMPILER_ID),clang)
  ifeq ($(filter NVIDIA,$(GPU_LIST)),NVIDIA)
    NVIDIA_CLANG_ARCH_FLAGS := $(foreach arch,$(NVIDIA_ARCH_LIST),--offload-arch=$(arch))
    OFFLOAD_FL += --cuda-path=$(CUDA_PATH) $(NVIDIA_CLANG_ARCH_FLAGS)
  endif
  ifeq ($(filter AMD,$(GPU_LIST)),AMD)
    AMD_CLANG_ARCH_FLAGS := $(foreach arch,$(AMD_ARCH_LIST),--offload-arch=$(arch))
    OFFLOAD_FL += --rocm-path=$(ROCM_PATH) --rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH) --libomptarget-amdgpu-bc-path=$(ROCM_DEVICE_LIB_PATH) $(AMD_CLANG_ARCH_FLAGS)
  endif
endif

ifeq ($(COMPILER_ID),amdclang)
  AMD_CLANG_ARCH_FLAGS := $(foreach arch,$(AMD_ARCH_LIST),--offload-arch=$(arch))
  OFFLOAD_FL += --rocm-path=$(ROCM_PATH) --rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH) --libomptarget-amdgpu-bc-path=$(ROCM_DEVICE_LIB_PATH) $(AMD_CLANG_ARCH_FLAGS)
endif

# =====================================================================
# PUT TILES FOR OPENMP (CPU AND GPU)
# =====================================================================
GPU_TILE_J      ?= 2048
GPU_ILP         ?= 16
CPU_TILE        ?= 32

TILE_VARS := GPU_TILE_J GPU_ILP CPU_TILE
$(foreach var,$(TILE_VARS), \
    $(if $(shell echo "$($(var))" | grep -Eq '^[0-9]+$$' && echo ok),, \
        $(eval $(call APPEND_ERROR, $(var) must be a positive integer. Got: '$($(var))')) \
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

BUILD_DIR ?= build
$(shell mkdir -p $(BUILD_DIR))

CFLAGS0 := -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=199309L -std=c11 -fPIC -O3 -march=native -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable
CFLAGS0 += -DGPU_TILE_J=$(GPU_TILE_J) -DGPU_ILP=$(GPU_ILP) -DCPU_TILE=$(CPU_TILE)

REPORT_CFLAGS :=
ifeq ($(BUG_REPORT),1)
  ifneq ($(filter gcc%,$(COMPILER_ID)),)
    REPORT_CFLAGS += -freport-bug
  else ifneq ($(filter clang%,$(COMPILER_ID)),)
    REPORT_CFLAGS += -gen-reproducer -fcrash-diagnostics-dir=$(abspath $(BUG_REPORT_OUT))
  endif
endif

CFLAGS := $(DEFINES) $(OPENMP_FLAG) $(OFFLOAD_FL) $(CFLAGS0) $(REPORT_CFLAGS)
HOST_ONLY_CFLAGS := $(DEFINES) $(OPENMP_FLAG) $(CFLAGS0) $(REPORT_CFLAGS)

COMPILE_CMD = $(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@
HOST_COMPILE_CMD = $(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -c $< -o $@

BUILD_RPATH_FLAG := -Wl,-rpath,$(CURDIR)/$(BUILD_DIR)
OMPTARGET_RPATH_FLAG :=

.PHONY: FORCE all clean distclean test test_offload bench bench_omp bug_report
CONFIG_STAMP := $(BUILD_DIR)/.config.stamp
.INTERMEDIATE: $(CONFIG_STAMP)
$(CONFIG_STAMP): FORCE
	@mkdir -p $(BUILD_DIR)
	@echo "CC=$(CC)" > $(CONFIG_STAMP).tmp
	@echo "GPU_LIST=$(GPU_LIST)" >> $(CONFIG_STAMP).tmp
	@echo "NVIDIA_ARCH_LIST=$(NVIDIA_ARCH_LIST)" >> $(CONFIG_STAMP).tmp
	@echo "AMD_ARCH_LIST=$(AMD_ARCH_LIST)" >> $(CONFIG_STAMP).tmp
	@echo "OFFLOAD_FL=$(OFFLOAD_FL)" >> $(CONFIG_STAMP).tmp
	@echo "CFLAGS=$(CFLAGS)" >> $(CONFIG_STAMP).tmp
	@if cmp -s $(CONFIG_STAMP).tmp $(CONFIG_STAMP) 2>/dev/null; \
	then rm -f $(CONFIG_STAMP).tmp; else mv $(CONFIG_STAMP).tmp $(CONFIG_STAMP); fi

SRC := src/bit.c
OBJ := $(BUILD_DIR)/bit.o $(BUILD_DIR)/gpu_layout_registry.o $(BUILD_DIR)/gpu_layout_fsm.o $(BUILD_DIR)/gpu_layout_kernels.o
TARGET := $(BUILD_DIR)/libbit.so
TARGET_STATIC := $(BUILD_DIR)/libbit.a
TEST_SRC := tests/test_bit.c
TEST_OBJ := $(BUILD_DIR)/test_bit.o
TEST_EXEC := $(BUILD_DIR)/test_bit
TEST_OFFLOAD_SRC := tests/test_offload.c
TEST_OFFLOAD_OBJ := $(BUILD_DIR)/test_offload.o
TEST_OFFLOAD_EXEC := $(BUILD_DIR)/test_offload
BENCH_SRC := benchmark/benchmark.c
BENCH_OBJ := $(BUILD_DIR)/benchmark.o
BENCH_EXEC := $(BUILD_DIR)/benchmark
BENCH_OMP_SRC := benchmark/openmp_bit.c
BENCH_OMP_OBJ := $(BUILD_DIR)/openmp_bit.o
BENCH_OMP_EXEC := $(BUILD_DIR)/openmp_bit
BENCH_OMP_GPU_SRC := benchmark/openmp_bit_nogpu.c
BENCH_OMP_GPU_OBJ := $(BUILD_DIR)/openmp_bit_nogpu.o
BENCH_OMP_GPU_EXEC := $(BUILD_DIR)/openmp_bit_nogpu
OPENMP_BIT_HELPERS_OBJ := $(BUILD_DIR)/openmp_bit_helpers.o

LIBPOPCNT ?= 1
LIBPOPCNT_LC := $(shell echo $(LIBPOPCNT) | tr A-Z a-z)
ifneq ($(filter $(LIBPOPCNT_LC),0 no n false f off),)
  $(info libpopcnt integration disabled)
else
  CFLAGS += -DUSE_LIBPOPCNT
endif

ifeq ($(strip $(USE_BUILTIN_POPCOUNT)),1)
  DEFINES += -DUSE_BUILTIN_POPCOUNT
endif

all: $(TARGET) $(TARGET_STATIC)

$(BUILD_DIR)/%.o: src/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/bit.o: src/bit.c src/bit_internal.h $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/gpu_layout_registry.o: src/gpu_layout_registry.c src/gpu_layout_registry.h src/gpu_layout.h src/gpu_layout_fsm.h $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/gpu_layout_fsm.o: src/gpu_layout_fsm.c src/gpu_layout_fsm.h src/gpu_layout.h src/gpu_layout_kernels.h $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/gpu_layout_kernels.o: src/gpu_layout_kernels.c src/gpu_layout_kernels.h src/gpu_layout.h $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/%.o: tests/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/%.o: benchmark/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(OPENMP_BIT_HELPERS_OBJ): benchmark/openmp_bit_helpers.c $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(TARGET): $(OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -shared -o $@ $^ $(BUILD_RPATH_FLAG)

$(TARGET_STATIC): $(OBJ)
	ar rcs $@ $^

test: $(TARGET) $(TEST_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) 

test_offload: $(TEST_OFFLOAD_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_OFFLOAD_EXEC) $(TEST_OFFLOAD_OBJ) $(BUILD_RPATH_FLAG) -lm

bench: $(TARGET) $(BENCH_OBJ) bench_omp
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $(BENCH_EXEC) $(BENCH_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) -lrt

ifeq ($(filter NONE,$(GPU_LIST)),NONE)
bench_omp: $(BENCH_OMP_GPU_EXEC)
else
bench_omp: $(BENCH_OMP_EXEC) $(BENCH_OMP_GPU_EXEC)
endif

$(BENCH_OMP_OBJ): $(BENCH_OMP_SRC) $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(BENCH_OMP_GPU_OBJ): $(BENCH_OMP_GPU_SRC) $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(BENCH_OBJ): $(BENCH_SRC) $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(BENCH_OMP_EXEC): $(BENCH_OMP_OBJ) $(OPENMP_BIT_HELPERS_OBJ) $(TARGET)
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $@ $(BENCH_OMP_OBJ) $(OPENMP_BIT_HELPERS_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) -lm -lrt

$(BENCH_OMP_GPU_EXEC): $(BENCH_OMP_GPU_OBJ) $(OPENMP_BIT_HELPERS_OBJ) $(TARGET)
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $@ $(BENCH_OMP_GPU_OBJ) $(OPENMP_BIT_HELPERS_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) -lm -lrt

bug_report:
	@BUG_GPU_ARCH_TAG := $(subst $(space),-,$(strip $(NVIDIA_ARCH_LIST) $(AMD_ARCH_LIST))) \
	$(if $(strip $$BUG_GPU_ARCH_TAG),,$$BUG_GPU_ARCH_TAG := default); \
	ifeq ($(filter NONE,$(GPU_LIST)),NONE) \
	  BUG_GPU_ARCH_TAG := cpu; \
	endif \
	BUG_REPORT_TAG="$(notdir $(CC))-$(subst $(space),-,$(strip $(GPU_LIST)))-$$BUG_GPU_ARCH_TAG" \
	@REPORT_PATH="$(BUG_REPORT_DIR)/$$(date +%Y%m%d-%H%M%S)-$$BUG_REPORT_TAG" \
	CC="$(CC)" GPU="$(GPU)" BUG_TARGET="$(BUG_TARGET)" BUILD_DIR="$(BUILD_DIR)" \
	MAKE="$(MAKE)" OPENMP_FLAG="$(OPENMP_FLAG)" OFFLOAD_FL='$(OFFLOAD_FL)' \
	CFLAGS0='$(CFLAGS0)' CFLAGS='$(CFLAGS)' DEFINES='$(DEFINES)' CC_ENV='$(CC_ENV)' \
	CUDA_PATH="$(CUDA_PATH)" AMD_ARCH="$(AMD_ARCH)" NVIDIA_ARCH="$(NVIDIA_ARCH)" \
	NVIDIA_SYSTEM_ARCHES="$(NVIDIA_SYSTEM_ARCHES)" NVIDIA_EFFECTIVE_ARCHES="$(NVIDIA_ARCH_LIST)" \
	bash $(BUG_REPORT_SCRIPT)

clean:
	rm -rf $(BUILD_DIR)

distclean: clean