IS_CLEAN_GOAL := $(filter clean distclean,$(MAKECMDGOALS))

# Ensure plain `make` builds the libraries (not the first internal target).
.DEFAULT_GOAL := all

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
# Check for icx
else ifneq ($(shell which icx 2>/dev/null),)
$(info Using icx to compile)
CC=icx
endif
else
# For clean-only invocations we don't need compiler discovery.
CC=gcc
endif
endif

ifeq ($(origin GPU),undefined)
ifeq ($(IS_CLEAN_GOAL),)
$(info Default GPU offload not set, will set to NVIDIA)
GPU=NVIDIA
else
# For clean-only invocations, pick a quiet, valid default.
GPU=NONE
endif
endif

# Convert GPU to uppercase for case-insensitive comparison
GPU := $(shell echo $(GPU) | tr a-z A-Z)

# check to see if the CC is one of icx, clang, or gcc
# Skip validation/messages for clean-only invocations.
ifeq ($(IS_CLEAN_GOAL),)
ifneq ($(filter $(notdir $(CC)),gcc clang icx),)
$(info CC is set to $(CC), which is a supported compiler)
else
$(error CC=$(CC) is not one of the supported compilers (gcc, clang, icx))
endif
endif

## additional flags
DEFINES ?=
AMD_ARCH ?= gfx900
ROCM_PATH ?= /opt/rocm
ROCM_DEVICE_LIB_PATH ?= $(ROCM_PATH)/amdgcn/bitcode
ROCM_LLVM_BIN ?= $(ROCM_PATH)/lib/llvm/bin
CC_ENV :=
NVIDIA_ARCHES ?= sm_50 sm_52 sm_53 sm_60 sm_61 sm_62 sm_70 sm_72 sm_75
NVIDIA_ARCH ?=
CUDA_PATH ?= /usr/lib/cuda
BUG_REPORT ?= 0
BUG_REPORT_DIR ?= bug_reports
BUG_REPORT_OUT ?= $(BUG_REPORT_DIR)
BUG_TARGET ?= bench_omp
ifeq ($(GPU),AMD)
BUG_GPU_ARCH_TAG := $(AMD_ARCH)
else ifeq ($(GPU),NVIDIA)
ifneq ($(strip $(NVIDIA_ARCH)),)
BUG_GPU_ARCH_TAG := $(NVIDIA_ARCH)
else
BUG_GPU_ARCH_TAG := $(firstword $(NVIDIA_ARCHES))
endif
else ifeq ($(GPU),INTEL)
BUG_GPU_ARCH_TAG := spir64
else
BUG_GPU_ARCH_TAG := cpu
endif
BUG_REPORT_TAG ?= $(notdir $(CC))-$(GPU)-$(BUG_GPU_ARCH_TAG)

# Backward compatibility: if NVIDIA_ARCH is set, use it as a single target.
ifneq ($(strip $(NVIDIA_ARCH)),)
NVIDIA_ARCHES := $(NVIDIA_ARCH)
endif

NVIDIA_CLANG_ARCH_FLAGS := $(foreach arch,$(NVIDIA_ARCHES),-Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=$(arch))

# Set the appropriate OpenMP flag based on compiler
ifeq ($(notdir $(CC)),gcc)
OPENMP_FLAG = -fopenmp
else ifeq ($(notdir $(CC)),g++)
OPENMP_FLAG = -fopenmp
else ifeq ($(notdir $(CC)),clang)
OPENMP_FLAG = -fopenmp
else
# For clang-family and Intel compilers
OPENMP_FLAG = -qopenmp
endif

ifeq ($(IS_CLEAN_GOAL),)
$(info GPU is set to $(GPU))
endif
#Set the appropriate offload flag based on GPU type
ifeq ($(GPU),NONE)
ifeq ($(notdir $(CC)),gcc)
OFFLOAD_FL = -foffload=disable
else
OFFLOAD_FL =
endif
DEFINES += -DNOGPU
else ifeq ($(GPU),NVIDIA)
ifeq ($(notdir $(CC)),gcc)
OFFLOAD_FL = -fno-stack-protector -fcf-protection=none -foffload=nvptx-none
else ifeq ($(notdir $(CC)),clang)
OFFLOAD_FL = -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=$(CUDA_PATH) $(NVIDIA_CLANG_ARCH_FLAGS) -Xopenmp-target=nvptx64-nvidia-cuda --no-cuda-version-check -foffload-lto
else ifeq ($(notdir $(CC)),icx)
OFFLOAD_FL = -fopenmp-targets=nvptx64-nvidia-cuda
endif
else ifeq ($(GPU),AMD)
ifeq ($(notdir $(CC)),gcc)
$(warning GPU=AMD with CC=gcc can trigger GCC amdgcn internal compiler errors on this setup; prefer CC=clang)
OFFLOAD_FL = -fno-stack-protector -fcf-protection=none \
             -foffload=amdgcn-amdhsa \
             -foffload-options=amdgcn-amdhsa="-march=$(AMD_ARCH)"
else ifeq ($(notdir $(CC)),clang)
CC_ENV = PATH=$(ROCM_LLVM_BIN):$$PATH
OFFLOAD_FL = -fopenmp-targets=amdgcn-amd-amdhsa --rocm-path=$(ROCM_PATH) --rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH) -Xopenmp-target=amdgcn-amd-amdhsa -march=$(AMD_ARCH)
else
$(error GPU=AMD is not supported with CC=$(CC); use gcc or clang)
endif
else ifeq ($(GPU),INTEL)
$(info GPU is set to INTEL, so will use Intel oneAPI icx Compiler)
CC=icx
OPENMP_FLAG = -fiopenmp
OFFLOAD_FL = -fopenmp-targets=spir64
else
$(error GPU = $(GPU) is not one of the supported GPUs (NVIDIA, AMD, INTEL))
endif
# Allow custom build directory, default to 'build' folder
BUILD_DIR ?= build

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

## compiler flags
CFLAGS0 = -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=199309L -std=c11 -fPIC -O3 -march=native \
-Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable

REPORT_CFLAGS :=
ifeq ($(BUG_REPORT),1)
ifeq ($(notdir $(CC)),gcc)
REPORT_CFLAGS += -freport-bug
else ifeq ($(notdir $(CC)),clang)
REPORT_CFLAGS += -gen-reproducer -fcrash-diagnostics-dir=$(abspath $(BUG_REPORT_OUT))
endif
endif

CFLAGS += $(DEFINES) $(OPENMP_FLAG) $(OFFLOAD_FL) $(CFLAGS0) $(REPORT_CFLAGS)

# Runtime search paths - occasionally you have to do this
BUILD_RPATH_FLAG := -Wl,-rpath,$(shell pwd)/$(BUILD_DIR)
OMPTARGET_LIBDIR :=
ifeq ($(notdir $(CC)),clang)
CLANG_LIBRARY_DIRS := $(subst :, ,$(patsubst libraries: =%,%,$(shell $(CC) -print-search-dirs 2>/dev/null | grep '^libraries: =')))
OMPTARGET_LIBDIR := $(firstword $(foreach d,$(CLANG_LIBRARY_DIRS),$(if $(wildcard $(d)/libomptarget.so*),$(d),)))
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
		echo "DEFINES=$(DEFINES)"; \
		echo "ROCM_PATH=$(ROCM_PATH)"; \
		echo "ROCM_DEVICE_LIB_PATH=$(ROCM_DEVICE_LIB_PATH)"; \
		echo "OPENMP_FLAG=$(OPENMP_FLAG)"; \
		echo "OFFLOAD_FL=$(OFFLOAD_FL)"; \
		echo "OMPTARGET_LIBDIR=$(OMPTARGET_LIBDIR)"; \
		echo "CFLAGS=$(CFLAGS)"; \
	} > $(CONFIG_STAMP).tmp
	@cmp -s $(CONFIG_STAMP).tmp $(CONFIG_STAMP) 2>/dev/null || mv $(CONFIG_STAMP).tmp $(CONFIG_STAMP)
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
BENCH_OMP_NO_CPU_EXEC = $(BUILD_DIR)/openmp_bit_nocpu

.PHONY: all clean test test_offload bench LIBPOPCNT bug_report

# Default targets are to build the shared and static libraries
all: $(TARGET) $(TARGET_STATIC)

# Rule to build object files in the build directory
$(BUILD_DIR)/%.o: src/%.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: tests/%.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

# Add pattern rule for benchmark source files
$(BUILD_DIR)/%.o: benchmark/%.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/openmp_bit.o: benchmark/openmp_bit.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/openmp_bit_nogpu.o: benchmark/openmp_bit_nogpu.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/openmp_bit_nocpu.o: benchmark/openmp_bit_nocpu.c $(CONFIG_STAMP)
	$(CC_ENV) $(CC) $(CFLAGS) -c $< -o $@

# Change from static to shared library compilation
$(TARGET): $(OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS) $(OMPTARGET_RPATH_FLAG)

# Build the static library as well
$(TARGET_STATIC): $(OBJ)
	$(AR) rcs $@ $^
	
# Update test to use shared library
test: $(TARGET) $(TEST_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJ) -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG) -lbit

test_offload: $(TEST_OFFLOAD_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(TEST_OFFLOAD_EXEC) $(TEST_OFFLOAD_OBJ) $(OPENMP_FLAG) -lm


# Add target to build the benchmark executable and the OpenMP benchmark
bench: $(TARGET) $(BENCH_OBJ) bench_omp
	$(CC_ENV) $(CC) $(CFLAGS) -o $(BENCH_EXEC) $(BENCH_OBJ) -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG) -lbit -lrt

# Conditional bench_omp target based on GPU setting
ifeq ($(GPU),NONE)
bench_omp: $(TARGET) $(BENCH_OMP_GPU_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(BENCH_OMP_GPU_EXEC) $(BENCH_OMP_GPU_OBJ) -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG) -lbit $(OPENMP_FLAG) -lrt
else
bench_omp: $(TARGET) $(BENCH_OMP_OBJ) $(BENCH_OMP_GPU_OBJ) $(BENCH_OMP_NO_CPU_OBJ)
	$(CC_ENV) $(CC) $(CFLAGS) -o $(BENCH_OMP_EXEC) $(BENCH_OMP_OBJ) -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG) -lbit $(OPENMP_FLAG) -lrt
	$(CC_ENV) $(CC) $(CFLAGS) -o $(BENCH_OMP_GPU_EXEC) $(BENCH_OMP_GPU_OBJ) -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG) -lbit $(OPENMP_FLAG) -lrt
	$(CC_ENV) $(CC) $(CFLAGS) -o $(BENCH_OMP_NO_CPU_EXEC) $(BENCH_OMP_NO_CPU_OBJ) -L$(BUILD_DIR) $(BUILD_RPATH_FLAG) $(OMPTARGET_RPATH_FLAG) -lbit $(OPENMP_FLAG) -lrt
endif


bug_report:
	@REPORT_PATH="$(BUG_REPORT_DIR)/$$(date +%Y%m%d-%H%M%S)-$(BUG_REPORT_TAG)"; \
	mkdir -p "$$REPORT_PATH"; \
	echo "Compiler bug report generation" > "$$REPORT_PATH/config.txt"; \
	echo "Timestamp: $$(date -Is)" >> "$$REPORT_PATH/config.txt"; \
	echo "CC=$(CC)" >> "$$REPORT_PATH/config.txt"; \
	echo "GPU=$(GPU)" >> "$$REPORT_PATH/config.txt"; \
	echo "AMD_ARCH=$(AMD_ARCH)" >> "$$REPORT_PATH/config.txt"; \
	echo "OPENMP_FLAG=$(OPENMP_FLAG)" >> "$$REPORT_PATH/config.txt"; \
	echo "OFFLOAD_FL=$(OFFLOAD_FL)" >> "$$REPORT_PATH/config.txt"; \
	echo "BUG_TARGET=$(BUG_TARGET)" >> "$$REPORT_PATH/config.txt"; \
	echo "CFLAGS=$(CFLAGS)" >> "$$REPORT_PATH/config.txt"; \
	echo "Compiler version:" >> "$$REPORT_PATH/config.txt"; \
	$(CC) --version | head -n 1 >> "$$REPORT_PATH/config.txt"; \
	$(MAKE) clean >/dev/null; \
	BUILD_STATUS=0; \
	$(MAKE) BUG_REPORT=1 BUG_REPORT_OUT="$$REPORT_PATH" $(BUG_TARGET) > "$$REPORT_PATH/build.log" 2>&1 || BUILD_STATUS=$$?; \
	if [ "$(notdir $(CC))" = "gcc" ]; then \
		$(CC) -v > "$$REPORT_PATH/gcc-v.txt" 2>&1 || true; \
		printf '%s\n' "$(CC) -v -save-temps=obj $(CFLAGS) -c src/bit.c -o $(BUILD_DIR)/gcc-bug-report.o" > "$$REPORT_PATH/gcc-repro-command.txt"; \
		$(CC) -v -save-temps=obj $(CFLAGS) -c src/bit.c -o $(BUILD_DIR)/gcc-bug-report.o > "$$REPORT_PATH/gcc-save-temps.log" 2>&1 || true; \
		if [ -f "$(BUILD_DIR)/gcc-bug-report.i" ]; then \
			cp -f "$(BUILD_DIR)/gcc-bug-report.i" "$$REPORT_PATH/src-bit.preprocessed.i"; \
		else \
			echo "Failed to generate GCC preprocessed source with -save-temps." > "$$REPORT_PATH/src-bit.preprocessed.i"; \
		fi; \
		rm -f "$(BUILD_DIR)/gcc-bug-report.o" "$(BUILD_DIR)/gcc-bug-report.i" "$(BUILD_DIR)/gcc-bug-report.s"; \
	else \
		$(CC_ENV) $(CC) $(DEFINES) $(OPENMP_FLAG) $(OFFLOAD_FL) $(CFLAGS0) -E src/bit.c -o "$$REPORT_PATH/src-bit.preprocessed.i" >/dev/null 2>&1 || true; \
	fi; \
	if [ "$$BUILD_STATUS" -ne 0 ]; then \
		if command -v gdb >/dev/null 2>&1; then \
			gdb --batch \
				-ex "set pagination off" \
				-ex "set follow-fork-mode child" \
				-ex "set detach-on-fork off" \
				-ex "set print thread-events off" \
				-ex "run" \
				-ex "thread apply all bt full" \
				--args $(MAKE) BUG_REPORT=1 BUG_REPORT_OUT="$$REPORT_PATH" $(BUG_TARGET) \
				> "$$REPORT_PATH/backtrace.txt" 2>&1 || true; \
		else \
			echo "gdb not found; unable to collect full backtrace automatically." > "$$REPORT_PATH/backtrace.txt"; \
			echo "Install gdb and rerun make bug_report to include full backtrace." >> "$$REPORT_PATH/backtrace.txt"; \
		fi; \
	else \
		echo "Build completed successfully; no compiler crash occurred." > "$$REPORT_PATH/backtrace.txt"; \
		echo "No backtrace available because there was no failing process." >> "$$REPORT_PATH/backtrace.txt"; \
	fi; \
	find $(BUILD_DIR) -maxdepth 1 \( -name '*.i' -o -name '*.ii' -o -name '*.s' -o -name '*.bc' -o -name '*.cui' \) -delete 2>/dev/null || true; \
	echo "Saved bug report files under $$REPORT_PATH"; \
	echo "- Build log: $$REPORT_PATH/build.log"; \
	echo "- Config: $$REPORT_PATH/config.txt"; \
	if [ "$(notdir $(CC))" = "gcc" ]; then \
		echo "- GCC -v details: $$REPORT_PATH/gcc-v.txt"; \
		echo "- GCC repro command: $$REPORT_PATH/gcc-repro-command.txt"; \
		echo "- GCC save-temps log: $$REPORT_PATH/gcc-save-temps.log"; \
	fi; \
	echo "- Backtrace: $$REPORT_PATH/backtrace.txt"; \
	echo "- Preprocessed source: $$REPORT_PATH/src-bit.preprocessed.i"


clean:
	@rm -rf $(BUILD_DIR)
    
# Additional target to clean everything including dependencies
distclean: clean
