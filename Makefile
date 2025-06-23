# Check for available compilers if user hasn't specified one
ifeq ($(origin CC),default)
$(info Default compiler not set, checking for available compilers...)
# Check for icx first
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
endif

ifeq ($(origin GPU),undefined)
$(info Default GPU offload not set, will set to NVIDIA)
GPU=NVIDIA
endif

# check to see if the CC is one of icx, clang, or gcc
ifneq ($(filter $(CC),gcc clang icx),)
$(info CC is set to $(CC), which is a supported compiler)
else
$(error CC=$(CC) is not one of the supported compilers (gcc, clang, icx))
endif

# Set the appropriate OpenMP flag based on compiler
ifeq ($(CC),gcc)
OPENMP_FLAG = -fopenmp
else ifeq ($(CC),g++)
OPENMP_FLAG = -fopenmp
else ifeq ($(CC),clang)
OPENMP_FLAG = -fopenmp
else
# For clang, icx, icc
OPENMP_FLAG = -qopenmp
endif

$(info GPU is set to $(GPU))
#Set the appropriate offload flag based on GPU type
ifeq ($(GPU),NVIDIA)
ifeq ($(CC),gcc)
OFFLOAD_FL = -fno-stack-protector -fcf-protection=none -foffload=nvptx-none
else ifeq ($(CC),clang)
OFFLOAD_FL = -fopenmp-targets=nvptx64 -foffload-lto
else ifeq ($(CC),icx)
OFFLOAD_FL = -fopenmp-targets=nvptx64-nvidia-cuda
endif
else ifeq ($(GPU),AMD)
ifeq ($(CC),gcc)
OFFLOAD_FL = -foffload=amdgcn-amd-amdhsa
else ifeq ($(CC),clang)
OFFLOAD_FL = -foffload=amdgcn-amd-amdhsa
else ifeq ($(CC),icx)
OFFLOAD_FL = -foffload=amdgcn-amd-amdhsa
endif
else ifeq ($(GPU),INTEL)
ifeq ($(CC),gcc)
OFFLOAD_FL = -foffload=spir64-unknown-unknown
else ifeq ($(CC),clang)
OFFLOAD_FL = -foffload=spir64-unknown-unknown
else ifeq ($(CC),icx)	
OFFLOAD_FL = -foffload=spir64-unknown-unknown
endif
else
$(error GPU = $(GPU) is not one of the supported GPUs (NVIDIA, AMD, INTEL))
endif
# Allow custom build directory, default to 'build' folder
BUILD_DIR ?= build

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

## compiler flags
CFLAGS0 = -Wall -Wextra -Iinclude -std=c11 -fPIC -O3 -march=native \
-Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable

## additional flags - may be removed at some point
DEFINES ?=
CFLAGS += $(DEFINES) $(OPENMP_FLAG) $(OFFLOAD_FL) $(CFLAGS0)


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

# Add benchmark source and executable
BENCH_SRC = benchmark/benchmark.c
BENCH_OBJ = $(BUILD_DIR)/benchmark.o
BENCH_EXEC = $(BUILD_DIR)/benchmark

# Add OpenMP benchmark source and executable
BENCH_OMP_SRC = benchmark/openmp_bit.c
BENCH_OMP_OBJ = $(BUILD_DIR)/openmp_bit.o
BENCH_OMP_EXEC = $(BUILD_DIR)/openmp_bit

.PHONY: all clean test bench LIBPOPCNT

# Default targets are to build the shared and static libraries
all: $(TARGET) $(TARGET_STATIC)

# Rule to build object files in the build directory
$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) $(OPENMP_FLAG) -c $< -o $@

$(BUILD_DIR)/%.o: tests/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Add pattern rule for benchmark source files
$(BUILD_DIR)/%.o: benchmark/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/openmp_bit.o: benchmark/openmp_bit.c
	$(CC) $(CFLAGS)  $(OPENMP_FLAG) -c $< -o $@

# Change from static to shared library compilation
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

# Build the static library as well
$(TARGET_STATIC): $(OBJ)
	$(AR) rcs $@ $^
	
# Update test to use shared library
test: $(TARGET) $(TEST_OBJ)
	$(CC) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJ) -L$(BUILD_DIR) -Wl,-rpath,$(shell pwd)/$(BUILD_DIR) -lbit


# Add target to build the benchmark executable and the OpenMP benchmark
bench: $(TARGET) $(BENCH_OBJ) bench_omp
	$(CC) $(CFLAGS) -o $(BENCH_EXEC) $(BENCH_OBJ) -L$(BUILD_DIR) -Wl,-rpath,$(shell pwd)/$(BUILD_DIR) -lbit -lrt

bench_omp: $(BENCH_OMP_OBJ)
	$(CC) $(CFLAGS) -o $(BENCH_OMP_EXEC) $(BENCH_OMP_OBJ) -L$(BUILD_DIR) -Wl,-rpath,$(shell pwd)/$(BUILD_DIR) -lbit $(OPENMP_FLAG) -lrt


clean:
	@echo "Cleaning up build directory..."
	rm -rf $(BUILD_DIR)
    
# Additional target to clean everything including dependencies
distclean: clean

