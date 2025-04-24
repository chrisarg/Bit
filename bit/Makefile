# Check for available compilers if user hasn't specified one
ifeq ($(origin CC),default)
# Check for icx first
ifneq ($(shell which icx 2>/dev/null),)
CC = icx
# Then check for clang
else ifneq ($(shell which clang 2>/dev/null),)
CC = clang
# Default to gcc
else
CC = gcc
endif
endif

# Set the appropriate OpenMP flag based on compiler
ifeq ($(CC),gcc)
OPENMP_FLAG = -fopenmp
else ifeq ($(CC),g++)
OPENMP_FLAG = -fopenmp
else
# For clang, icx, icc
OPENMP_FLAG = -qopenmp
endif

# Allow custom build directory, default to 'build' folder
BUILD_DIR ?= build

# Ensure the build directory exists
$(shell mkdir -p $(BUILD_DIR))

## compiler flags
CFLAGS = -Wall -Wextra -Iinclude -std=c11 -fPIC -O3 -march=native

## additional flags - may be removed at some point
DEFINES ?=
CFLAGS += $(DEFINES)


# Handle libpopcnt integration
LIBPOPCNT ?= 0
ifneq ($(filter-out 0,$(LIBPOPCNT)),)
CFLAGS += -DUSE_LIBPOPCNT
ifeq ($(filter clean test bench distclean,$(MAKECMDGOALS)),)
$(info Using libpopcnt for population count)
endif
endif

SRC = src/bit.c
OBJ = $(BUILD_DIR)/bit.o
# Change from static to shared library
TARGET = $(BUILD_DIR)/libbit.so
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

# Default target
all: $(TARGET)


# Rule to build object files in the build directory
$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

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


# Update test to use shared library
test: $(TARGET) $(TEST_OBJ)
	$(CC) $(CFLAGS) -o $(TEST_EXEC) $(TEST_OBJ) -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -lbit
	$(TEST_EXEC)

# Add target to build the benchmark executable and the OpenMP benchmark
bench: $(TARGET) $(BENCH_OBJ) bench_omp
	$(CC) $(CFLAGS) -o $(BENCH_EXEC) $(BENCH_OBJ) -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -lbit -lrt

bench_omp: $(BENCH_OMP_OBJ)
	$(CC) $(CFLAGS) -o $(BENCH_OMP_EXEC) $(BENCH_OMP_OBJ) -L$(BUILD_DIR) -Wl,-rpath,$(BUILD_DIR) -lbit $(OPENMP_FLAG) -lrt


clean:
	@echo "Cleaning up build directory..."
	rm -rf $(BUILD_DIR)
    
# Additional target to clean everything including dependencies
distclean: clean

# Special target that enables libpopcnt
LIBPOPCNT:
	$(MAKE) LIBPOPCNT=1