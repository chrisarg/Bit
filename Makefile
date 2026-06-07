# ----------------------------------------------------------------------------
# CORE BUILD MAKEFILE
# ----------------------------------------------------------------------------

IS_CLEAN_GOAL := $(filter clean distclean,$(MAKECMDGOALS))
.DEFAULT_GOAL := all

LIBOMPTARGET_INFO ?= 0
LIBOMPTARGET_DEBUG ?= 0
export LIBOMPTARGET_INFO LIBOMPTARGET_DEBUG

ifeq ($(origin CC),default)
  ifeq ($(IS_CLEAN_GOAL),)
    ifneq ($(shell which gcc 2>/dev/null),)
      CC := gcc
    else ifneq ($(shell which clang 2>/dev/null),)
      CC := clang
    endif
  endif
endif

GPU ?= NONE
override GPU := $(shell printf '%s' '$(GPU)' | tr 'a-z' 'A-Z' | tr -d '[:space:]')

ifeq ($(IS_CLEAN_GOAL),)
  ifneq ($(filter $(notdir $(CC)),gcc clang),)
    CC_OK := yes
  else
    $(error CC=$(CC) is not one of the supported compilers (gcc, clang))
  endif
endif

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
BUG_REPORT_SCRIPT := scripts/generate_bug_report.sh

empty :=
space := $(empty) $(empty)
comma := ,

override NVIDIA_ARCH := $(shell printf '%s' '$(NVIDIA_ARCH)' | tr 'A-Z' 'a-z' | tr -d '[:space:]')
override NVIDIA_ARCH_POLICY := $(shell printf '%s' '$(NVIDIA_ARCH_POLICY)' | tr 'a-z' 'A-Z' | tr -d '[:space:]')
override AMD_ARCH := $(shell printf '%s' '$(AMD_ARCH)' | tr 'A-Z' 'a-z' | tr -d '[:space:]')

nvidia_archs := $(subst $(comma),$(space),$(strip $(NVIDIA_ARCH)))
amd_archs := $(subst $(comma),$(space),$(strip $(AMD_ARCH)))

AMD_ARCH_INVALID := $(shell if [ -n "$(strip $(AMD_ARCH))" ] && ! printf '%s\n' "$(AMD_ARCH)" | grep -Eq '^gfx[0-9a-f]+$$'; then printf '%s' "$(AMD_ARCH)"; fi)

NVIDIA_SYSTEM_ARCHES := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | awk -F '.' '{gsub(/ /,""); print "sm_"$$1$$2}' | sort -u 2>/dev/null || true)

ifeq ($(GPU),NVIDIA)
  ifneq ($(strip $(NVIDIA_ARCH)),)
    NVIDIA_ARCH_LIST := $(strip $(subst $(comma),$(space),$(NVIDIA_ARCH)))
  else
    NVIDIA_ARCH_LIST := $(strip $(NVIDIA_SYSTEM_ARCHES))
  endif
else
  NVIDIA_ARCH_LIST :=
endif

ifeq ($(strip $(NVIDIA_ARCH_LIST)),)
  ifeq ($(GPU),NVIDIA)
    $(error NVIDIA_ARCH was not specified and no visible supported NVIDIA GPU archs were detected)
  endif
endif

NVIDIA_EFFECTIVE_ARCHES := $(strip $(NVIDIA_ARCH_LIST))

ifeq ($(GPU),AMD)
  BUG_GPU_ARCH_TAG := $(if $(strip $(amd_archs)),$(subst $(space),-,$(amd_archs)),default)
else ifeq ($(GPU),NVIDIA)
  BUG_GPU_ARCH_TAG := $(if $(strip $(NVIDIA_ARCH_LIST)),$(subst $(space),-,$(NVIDIA_ARCH_LIST)),default)
else
  BUG_GPU_ARCH_TAG := cpu
endif
BUG_REPORT_TAG ?= $(notdir $(CC))-$(GPU)-$(BUG_GPU_ARCH_TAG)

NVIDIA_CLANG_ARCH_FLAGS := $(foreach arch,$(NVIDIA_ARCH_LIST),--offload-arch=$(arch))
ifeq ($(notdir $(CC)),gcc)
  NVIDIA_GCC_ARCH_FLAG := $(foreach arch,$(NVIDIA_ARCH_LIST),-foffload-options=nvptx-none=-march=$(arch))
endif

ifeq ($(IS_CLEAN_GOAL),)
  $(info GPU is set to $(GPU))
endif

ifeq ($(GPU),NONE)
  ifeq ($(notdir $(CC)),gcc)
    OFFLOAD_FL := -foffload=disable
  else
    OFFLOAD_FL :=
  endif
  DEFINES += -DNOGPU
  ifneq ($(strip $(NVIDIA_ARCH)$(AMD_ARCH)),)
    $(error GPU=NONE does not accept NVIDIA_ARCH or AMD_ARCH overrides)
  endif
else ifeq ($(GPU),NVIDIA)
  ifneq ($(strip $(AMD_ARCH)),)
    $(error GPU=NVIDIA does not accept AMD_ARCH; use NVIDIA_ARCH=sm_xy[,sm_ab,...])
  endif
  ifeq ($(notdir $(CC)),gcc)
    $(warning GPU=NVIDIA with CC=gcc is experimental; use CC=clang for reliable nvptx offload)
    OFFLOAD_FL := -foffload=nvptx-none $(NVIDIA_GCC_ARCH_FLAG)
  else ifeq ($(notdir $(CC)),clang)
    OFFLOAD_FL := -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=$(CUDA_PATH) $(NVIDIA_CLANG_ARCH_FLAGS) -Xopenmp-target=nvptx64-nvidia-cuda --no-cuda-version-check -foffload-lto
  else
    $(error GPU=NVIDIA is supported with gcc or clang only)
  endif
else ifeq ($(GPU),AMD)
  ifneq ($(strip $(NVIDIA_ARCH)),)
    $(error GPU=AMD does not accept NVIDIA_ARCH; use AMD_ARCH=gfx_target)
  endif
  ifneq ($(strip $(AMD_ARCH)),)
    ifneq ($(strip $(AMD_ARCH_INVALID)),)
      $(error AMD_ARCH must match gfx<target>, e.g. gfx90a; got '$(AMD_ARCH_INVALID)')
    endif
  endif
  ifeq ($(notdir $(CC)),gcc)
    $(warning GPU=AMD with CC=gcc may trigger compiler issues; prefer CC=clang)
    OFFLOAD_FL := -foffload=amdgcn-amdhsa $(foreach arch,$(amd_archs),-foffload-options=amdgcn-amdhsa="-march=$(arch)")
  else ifeq ($(notdir $(CC)),clang)
    OFFLOAD_FL := -fopenmp-targets=amdgcn-amd-amdhsa --rocm-path=$(ROCM_PATH) --rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH) $(foreach arch,$(amd_archs),-Xopenmp-target=amdgcn-amd-amdhsa -march=$(arch))
  else
    $(error GPU=AMD is supported with gcc or clang only)
  endif
else
  $(error GPU=$(GPU) is not one of NONE, NVIDIA, AMD)
endif

BUILD_DIR ?= build
$(shell mkdir -p $(BUILD_DIR))

CFLAGS0 := -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=199309L -std=c11 -fPIC -O3 -march=native -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable
OPENMP_FLAG := -fopenmp
OPENMP_LINK_LIBS := -lgomp
OPENMP_LINK_PRE := -Wl,--no-as-needed
OPENMP_LINK_POST := -Wl,--as-needed
REPORT_CFLAGS :=
ifeq ($(BUG_REPORT),1)
  ifeq ($(notdir $(CC)),gcc)
    REPORT_CFLAGS += -freport-bug
  else ifeq ($(notdir $(CC)),clang)
    REPORT_CFLAGS += -gen-reproducer -fcrash-diagnostics-dir=$(abspath $(BUG_REPORT_OUT))
  endif
endif

CFLAGS := $(DEFINES) $(OPENMP_FLAG) $(OFFLOAD_FL) $(CFLAGS0) $(REPORT_CFLAGS)
HOST_ONLY_CFLAGS := $(filter-out $(OFFLOAD_FL),$(CFLAGS))

COMPILE_CMD = $(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@
HOST_COMPILE_CMD = $(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -c $< -o $@

BUILD_RPATH_FLAG := -Wl,-rpath,$(shell pwd)/$(BUILD_DIR)
OMPTARGET_RPATH_FLAG :=

.PHONY: FORCE all clean distclean test test_offload bench bench_omp bug_report
CONFIG_STAMP := $(BUILD_DIR)/.config.stamp
.INTERMEDIATE: $(CONFIG_STAMP)
$(CONFIG_STAMP): FORCE
	@mkdir -p $(BUILD_DIR)
	@echo "CC=$(CC)" > $(CONFIG_STAMP).tmp
	@echo "GPU=$(GPU)" >> $(CONFIG_STAMP).tmp
	@echo "NVIDIA_ARCH=$(NVIDIA_ARCH)" >> $(CONFIG_STAMP).tmp
	@echo "AMD_ARCH=$(AMD_ARCH)" >> $(CONFIG_STAMP).tmp
	@echo "OFFLOAD_FL=$(OFFLOAD_FL)" >> $(CONFIG_STAMP).tmp
	@echo "CFLAGS=$(CFLAGS)" >> $(CONFIG_STAMP).tmp
	@if cmp -s $(CONFIG_STAMP).tmp $(CONFIG_STAMP) 2>/dev/null; then rm -f $(CONFIG_STAMP).tmp; else mv $(CONFIG_STAMP).tmp $(CONFIG_STAMP); fi

SRC := src/bit.c
OBJ := $(BUILD_DIR)/bit.o
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

all: $(TARGET) $(TARGET_STATIC)

$(BUILD_DIR)/%.o: src/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/%.o: tests/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(BUILD_DIR)/%.o: benchmark/%.c $(CONFIG_STAMP)
	$(COMPILE_CMD)

$(OPENMP_BIT_HELPERS_OBJ): benchmark/openmp_bit_helpers.c $(CONFIG_STAMP)
	$(HOST_COMPILE_CMD)

$(TARGET): $(OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -shared -o $@ $^ $(BUILD_RPATH_FLAG) $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST)

$(TARGET_STATIC): $(OBJ)
	ar rcs $@ $^

test: $(TARGET) $(TEST_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJ) $(BUILD_RPATH_FLAG) $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST)

test_offload: $(TEST_OFFLOAD_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_OFFLOAD_EXEC) $(TEST_OFFLOAD_OBJ) $(BUILD_RPATH_FLAG) $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST) -lm

bench: $(TARGET) $(BENCH_OBJ) bench_omp
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $(BENCH_EXEC) $(BENCH_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST) -lrt

ifeq ($(GPU),NONE)
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
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $@ $(BENCH_OMP_OBJ) $(OPENMP_BIT_HELPERS_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST) -lm -lrt

$(BENCH_OMP_GPU_EXEC): $(BENCH_OMP_GPU_OBJ) $(OPENMP_BIT_HELPERS_OBJ) $(TARGET)
	$(CC_ENV) $(CC) $(HOST_ONLY_CFLAGS) -o $@ $(BENCH_OMP_GPU_OBJ) $(OPENMP_BIT_HELPERS_OBJ) -L$(BUILD_DIR) -lbit $(BUILD_RPATH_FLAG) $(OPENMP_LINK_PRE) $(OPENMP_LINK_LIBS) $(OPENMP_LINK_POST) -lm -lrt

bug_report:
	@REPORT_PATH="$(BUG_REPORT_DIR)/$$(date +%Y%m%d-%H%M%S)-$(BUG_REPORT_TAG)" \
	CC="$(CC)" GPU="$(GPU)" BUG_TARGET="$(BUG_TARGET)" BUILD_DIR="$(BUILD_DIR)" \
	MAKE="$(MAKE)" OPENMP_FLAG="$(OPENMP_FLAG)" OFFLOAD_FL='$(OFFLOAD_FL)' \
	CFLAGS0='$(CFLAGS0)' CFLAGS='$(CFLAGS)' DEFINES='$(DEFINES)' CC_ENV='$(CC_ENV)' \
	CUDA_PATH="$(CUDA_PATH)" AMD_ARCH="$(AMD_ARCH)" NVIDIA_ARCH="$(NVIDIA_ARCH)" \
	NVIDIA_SYSTEM_ARCHES="$(NVIDIA_SYSTEM_ARCHES)" NVIDIA_EFFECTIVE_ARCHES="$(NVIDIA_EFFECTIVE_ARCHES)" \
	bash $(BUG_REPORT_SCRIPT)

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
