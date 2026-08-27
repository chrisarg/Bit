#include "bit.h"
#include <stdbool.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define SIZE_OF_TEST_BIT 65536
typedef struct {
  int total;
  int passed;
  int failed;
} TestResults;

typedef struct {
  Bit_T set;
  int visits;
} MapClosure;

// Initialize test results
TestResults results = {0, 0, 0};

static void test_map_apply(int n, int bit, void *cl) {
  MapClosure *closure = (MapClosure *)cl;
  closure->visits++;
  Bit_put(closure->set, n, 1 - bit);
}

static Bit_T make_bit_with_indices(int length, const int *indices, size_t count) {
  Bit_T bit = Bit_new(length);
  for (size_t i = 0; i < count; ++i) {
    Bit_bset(bit, indices[i]);
  }
  return bit;
}

static void set_bits(Bit_T bit, const int *indices, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Bit_bset(bit, indices[i]);
  }
}

static void clear_bits(Bit_T bit, const int *indices, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Bit_bclear(bit, indices[i]);
  }
}

static bool expect_bits(Bit_T bit, const int *indices, size_t count, int expected) {
  for (size_t i = 0; i < count; ++i) {
    if (Bit_get(bit, indices[i]) != expected) {
      return false;
    }
  }
  return true;
}

static bool expect_assertion(void (*fn)(void)) {
  pid_t pid = fork();
  if (pid < 0) {
    return false;
  }

  if (pid == 0) {
    fn();
    _exit(0);
  }

  int status = 0;
  (void)waitpid(pid, &status, 0);
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void invalid_bit_get(void) {
  Bit_T bit = Bit_new(8);
  (void)Bit_get(bit, 8);
  Bit_free(&bit);
}

static void invalid_bit_bset(void) {
  Bit_T bit = Bit_new(8);
  Bit_bset(bit, -1);
  Bit_free(&bit);
}

static void invalid_bit_put(void) {
  Bit_T bit = Bit_new(8);
  (void)Bit_put(bit, 8, 0);
  Bit_free(&bit);
}

static void invalid_bitdb_count_at(void) {
  Bit_DB_T db = BitDB_new(8, 2);
  (void)BitDB_count_at(db, 2);
  BitDB_free(&db);
}

static void invalid_bitdb_get_from(void) {
  Bit_DB_T db = BitDB_new(8, 2);
  (void)BitDB_get_from(db, 2);
  BitDB_free(&db);
}

// Utility function to print test results
void report_test(const char *test_name, bool passed) {
  results.total++;
  if (passed) {
    printf("PASS: %s\n", test_name);
    results.passed++;
  } else {
    printf("FAIL: %s\n", test_name);
    results.failed++;
  }
}

// Basic operations tests
bool test_bit_new() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  bool success = (bit != NULL);
  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_extract() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bit, 2);
  Bit_bset(bit, 0);
  unsigned char buffer[Bit_buffer_size(SIZE_OF_TEST_BIT)];
  Bit_extract(bit, (void *)buffer);
  bool success = (buffer[0] == 0b00000101);
  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_load() {
  unsigned char buffer[Bit_buffer_size(SIZE_OF_TEST_BIT)];
  buffer[0] = 0b00000101;
  Bit_T bit = Bit_load(SIZE_OF_TEST_BIT, buffer);
  bool success = (Bit_get(bit, 0) == 1 && Bit_get(bit, 2) == 1);
  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_set() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bit, 2);
  bool success = (Bit_get(bit, 2) == 1);
  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_clear() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bit, 2);
  Bit_bclear(bit, 2);
  bool success = (Bit_get(bit, 2) == 0);
  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_put() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  int prev = Bit_put(bit, 3, 1);
  bool success = (prev == 0 && Bit_get(bit, 3) == 1);

  prev = Bit_put(bit, 3, 0);
  success = success && (prev == 1 && Bit_get(bit, 3) == 0);

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_set_range() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  Bit_set(bit, 2, SIZE_OF_TEST_BIT / 2); // Set bits 2,3,4,5

  bool success = 1;

  for (int index = 2; index <= SIZE_OF_TEST_BIT / 2; index++)
    success = success && Bit_get(bit, index) == 1;

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_clear_range() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  // Set all bits
  for (int i = 0; i < SIZE_OF_TEST_BIT / 2; i++) {
    Bit_bset(bit, i);
  }

  Bit_clear(bit, 2, 5); // Clear bits 2,3,4,5

  bool success =
      (Bit_get(bit, 2) == 0 && Bit_get(bit, 3) == 0 && Bit_get(bit, 4) == 0 &&
       Bit_get(bit, 5) == 0 && Bit_get(bit, 1) == 1);
  for (int index = 6; index < SIZE_OF_TEST_BIT / 2; index++)
    success = success && (Bit_get(bit, index) == 1);

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_count() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bit, 1);
  Bit_bset(bit, 3);
  Bit_bset(bit, SIZE_OF_TEST_BIT / 2);

  int count = Bit_count(bit);
  bool success = (count == 3);

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_aset_aclear() {
  Bit_T bit = Bit_new(16);
  int set_indices[] = {0, 2, 7, 13};
  int zero_indices[] = {1, 3, 4, 5};
  int clear_indices[] = {2, 13};
  int preserved_indices[] = {0, 7};

  Bit_aset(bit, set_indices, 4);
  bool success = expect_bits(bit, set_indices, 4, 1) &&
                 expect_bits(bit, zero_indices, 4, 0);

  Bit_aclear(bit, clear_indices, 2);
  success = success && expect_bits(bit, clear_indices, 2, 0) &&
            expect_bits(bit, preserved_indices, 2, 1);

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_not_and_map() {
  Bit_T bit = Bit_new(16);
  int initial_indices[] = {0, 2, 4, 6, 8, 10, 12, 14};
  int one_indices[] = {0, 1, 3, 5, 6};
  int zero_indices[] = {2, 4};
  int reset_indices[] = {0, 2};
  int mapped_one_indices[] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int mapped_zero_indices[] = {0, 2};

  set_bits(bit, initial_indices, 8);

  Bit_not(bit, 1, 5);
  bool success = expect_bits(bit, one_indices, 5, 1) &&
                 expect_bits(bit, zero_indices, 2, 0);

  Bit_clear(bit, 0, 15);
  set_bits(bit, reset_indices, 2);

  MapClosure closure = {bit, 0};
  Bit_map(bit, test_map_apply, &closure);
  success = success && (closure.visits == 16) &&
            expect_bits(bit, mapped_zero_indices, 2, 0) &&
            expect_bits(bit, mapped_one_indices, 14, 1);

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_buffer_size_and_length() {
  Bit_T bit = Bit_new(65);
  bool success = (Bit_length(bit) == 65 && Bit_buffer_size(1) == 8 &&
                  Bit_buffer_size(65) == 16);

  report_test(__func__, success);
  Bit_free(&bit);
  return success;
}

bool test_bit_invalid_index_handling() {
  bool success = expect_assertion(invalid_bit_get) &&
                 expect_assertion(invalid_bit_bset) &&
                 expect_assertion(invalid_bit_put);

  report_test(__func__, success);
  return success;
}

// Comparison tests
bool test_bit_eq() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);

  Bit_bset(bit2, 1);
  Bit_bset(bit2, 3);

  bool success = Bit_eq(bit1, bit2);
  Bit_bset(bit2, 8);
  success = success && !Bit_eq(bit1, bit2);
  Bit_bclear(bit2, 8);
  Bit_bset(bit2, 75);
  success = success && !Bit_eq(bit1, bit2);
  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  return success;
}

bool test_bit_leq() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);

  Bit_bset(bit2, 1);
  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);

  bool success = Bit_leq(bit1, bit2) && !Bit_leq(bit2, bit1);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  return success;
}

bool test_bit_lt() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);

  Bit_bset(bit2, 1);
  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);

  bool success = Bit_lt(bit1, bit2) && !Bit_lt(bit2, bit1);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  return success;
}

// Set operation tests
bool test_bit_union() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);

  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);

  Bit_T union_bit = Bit_union(bit1, bit2);

  bool success = (Bit_get(union_bit, 1) == 1 && Bit_get(union_bit, 3) == 1 &&
                  Bit_get(union_bit, 5) == 1 && Bit_get(union_bit, 0) == 0 &&
                  Bit_get(union_bit, 2) == 0 && Bit_get(union_bit, 4) == 0);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  Bit_free(&union_bit);
  return success;
}

bool test_bit_inter() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);
  Bit_bset(bit1, 5);

  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);
  Bit_bset(bit2, 7);

  Bit_T inter_bit = Bit_inter(bit1, bit2);

  bool success = (Bit_get(inter_bit, 3) == 1 && Bit_get(inter_bit, 5) == 1 &&
                  Bit_get(inter_bit, 1) == 0 && Bit_get(inter_bit, 7) == 0);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  Bit_free(&inter_bit);
  return success;
}

bool test_bit_minus() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);
  Bit_bset(bit1, 5);

  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);
  Bit_bset(bit2, 7);

  Bit_T minus_bit = Bit_minus(bit1, bit2);

  bool success = (Bit_get(minus_bit, 1) == 1 && Bit_get(minus_bit, 3) == 0 &&
                  Bit_get(minus_bit, 5) == 0 && Bit_get(minus_bit, 7) == 0);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  Bit_free(&minus_bit);
  return success;
}

bool test_bit_diff() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);
  Bit_bset(bit1, 5);

  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);
  Bit_bset(bit2, 7);

  Bit_T diff_bit = Bit_diff(bit1, bit2);

  bool success = (Bit_get(diff_bit, 1) == 1 && Bit_get(diff_bit, 7) == 1 &&
                  Bit_get(diff_bit, 3) == 0 && Bit_get(diff_bit, 5) == 0);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  Bit_free(&diff_bit);
  return success;
}

// Count operation tests
bool test_bit_count_operations() {
  Bit_T bit1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_T bit2 = Bit_new(SIZE_OF_TEST_BIT);

  Bit_bset(bit1, 1);
  Bit_bset(bit1, 3);
  Bit_bset(bit1, 5);

  Bit_bset(bit2, 3);
  Bit_bset(bit2, 5);
  Bit_bset(bit2, 7);

  // set some extra bits to test final bits
  int num_of_final_bits = SIZE_OF_TEST_BIT - 8;
  for (int i = 8; i < SIZE_OF_TEST_BIT; i++) {
    Bit_bset(bit1, i);
    Bit_bset(bit2, i);
  }

  int union_count = Bit_union_count(bit1, bit2);
  int inter_count = Bit_inter_count(bit1, bit2);
  int minus_count = Bit_minus_count(bit1, bit2);
  int diff_count = Bit_diff_count(bit1, bit2);

  bool success = (union_count == 4 + num_of_final_bits &&
                  inter_count == 2 + num_of_final_bits && minus_count == 1 &&
                  diff_count == 2);

  report_test(__func__, success);
  Bit_free(&bit1);
  Bit_free(&bit2);
  return success;
}

// Edge case tests
bool test_bit_null_handling() {
  Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bit, 1);
  Bit_bset(bit, 3);

  // Test operations with one NULL operand
  Bit_T union_result = Bit_union(bit, NULL);
  Bit_T inter_result = Bit_inter(bit, NULL);
  Bit_T minus_result = Bit_minus(bit, NULL);

  bool success = (Bit_count(union_result) == Bit_count(bit) &&
                  Bit_count(inter_result) == 0 &&
                  Bit_count(minus_result) == Bit_count(bit));

  Bit_free(&bit);
  Bit_free(&union_result);
  Bit_free(&inter_result);
  Bit_free(&minus_result);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_new() {
  Bit_DB_T bit = BitDB_new(SIZE_OF_TEST_BIT, 10);
  bool success = (bit != NULL);
  report_test(__func__, success);
  BitDB_free(&bit);
  return success;
}

bool test_bitDB_properties() {
  Bit_DB_T bit = BitDB_new(SIZE_OF_TEST_BIT, 10);
  bool success =
      (BitDB_length(bit) == SIZE_OF_TEST_BIT && BitDB_nelem(bit) == 10);

  BitDB_free(&bit);
  report_test(__func__, success);
  return success;
}

bool test_bitDB_count_at_and_count() {
  Bit_DB_T bit = BitDB_new(64, 3);
  int first_indices[] = {1, 63};
  int second_indices[] = {0, 1, 2};
  Bit_T first = make_bit_with_indices(64, first_indices, 2);
  Bit_T second = make_bit_with_indices(64, second_indices, 3);
  Bit_T third = Bit_new(64);

  BitDB_put_at(bit, 0, first);
  BitDB_put_at(bit, 1, second);
  BitDB_put_at(bit, 2, third);

  int *counts = BitDB_count(bit);
  bool success = (BitDB_count_at(bit, 0) == 2 && BitDB_count_at(bit, 1) == 3 &&
                  BitDB_count_at(bit, 2) == 0 && counts[0] == 2 &&
                  counts[1] == 3 && counts[2] == 0);

  free(counts);
  Bit_free(&first);
  Bit_free(&second);
  Bit_free(&third);
  BitDB_free(&bit);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_clear_ops() {
  Bit_DB_T bit = BitDB_new(64, 2);
  int first_indices[] = {1, 3};
  int second_indices[] = {5, 7};
  Bit_T first = make_bit_with_indices(64, first_indices, 2);
  Bit_T second = make_bit_with_indices(64, second_indices, 2);

  BitDB_put_at(bit, 0, first);
  BitDB_put_at(bit, 1, second);

  BitDB_clear_at(bit, 1);
  bool success = (BitDB_count_at(bit, 0) == 2 && BitDB_count_at(bit, 1) == 0);

  BitDB_clear(bit);
  success =
      success && (BitDB_count_at(bit, 0) == 0 && BitDB_count_at(bit, 1) == 0);

  Bit_free(&first);
  Bit_free(&second);
  BitDB_free(&bit);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_get_put() {
  Bit_DB_T bit = BitDB_new(SIZE_OF_TEST_BIT, 10);
  Bit_T bitset = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bitset, 1);
  Bit_bset(bitset, 3);

  BitDB_put_at(bit, 0, bitset);
  Bit_T retrieved = BitDB_get_from(bit, 0);

  bool success = (Bit_get(retrieved, 1) == 1 && Bit_get(retrieved, 3) == 1);

  Bit_free(&bitset);
  Bit_free(&retrieved);
  BitDB_free(&bit);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_extract_replace() {
  Bit_DB_T bit = BitDB_new(SIZE_OF_TEST_BIT, 10);
  Bit_T bitset = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bitset, 1);
  Bit_bset(bitset, 3);

  BitDB_put_at(bit, 0, bitset);

  unsigned char buffer[SIZE_OF_TEST_BIT / 8];
  BitDB_extract_from(bit, 0, buffer);

  bool success = (buffer[0] == (unsigned char)((1 << 1) | (1 << 3)));

  BitDB_replace_at(bit, 0, buffer);

  Bit_T retrieved = BitDB_get_from(bit, 0);

  success =
      success && (Bit_get(retrieved, 1) == 1 && Bit_get(retrieved, 3) == 1);

  Bit_free(&bitset);
  Bit_free(&retrieved);
  BitDB_free(&bit);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_load_from_buffer() {
  unsigned char buffer[2 * 8] = {0};
  buffer[0] = 0x03;
  buffer[8] = 0x0C;

  Bit_DB_T bit = BitDB_load(16, 2, buffer);
  Bit_T first = BitDB_get_from(bit, 0);
  Bit_T second = BitDB_get_from(bit, 1);

  bool success = (Bit_get(first, 0) == 1 && Bit_get(first, 1) == 1 &&
                  Bit_get(second, 2) == 1 && Bit_get(second, 3) == 1 &&
                  Bit_get(second, 0) == 0);

  Bit_free(&first);
  Bit_free(&second);
  BitDB_free(&bit);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_invalid_index_handling() {
  bool success = expect_assertion(invalid_bitdb_count_at) &&
                 expect_assertion(invalid_bitdb_get_from);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_count_macro_variants() {
  Bit_DB_T left = BitDB_new(128, 2);
  Bit_DB_T right = BitDB_new(128, 2);
  Bit_T value = Bit_new(128);

  Bit_bset(value, 1);
  Bit_bset(value, 3);
  BitDB_put_at(left, 0, value);
  Bit_clear(value, 0, 127);
  Bit_bset(value, 2);
  BitDB_put_at(left, 1, value);

  Bit_clear(value, 0, 127);
  Bit_bset(value, 3);
  BitDB_put_at(right, 0, value);
  Bit_clear(value, 0, 127);
  Bit_bset(value, 1);
  Bit_bset(value, 2);
  BitDB_put_at(right, 1, value);

  const SETOP_COUNT_OPTS opts = {.num_cpu_threads = 1};
  const int expected_inter[4] = {1, 1, 0, 1};
  const int expected_union[4] = {2, 3, 2, 2};
  const int expected_diff[4] = {1, 2, 2, 1};
  const int expected_minus[4] = {1, 1, 1, 0};
  bool success = true;

  int *inter_counts = BitDB_inter_count(left, right, opts, cpu);
  for (int i = 0; i < 4; ++i) {
    success = success && inter_counts[i] == expected_inter[i];
  }
  free(inter_counts);

  int *union_counts = BitDB_union_count(left, right, opts, cpu);
  for (int i = 0; i < 4; ++i) {
    success = success && union_counts[i] == expected_union[i];
  }
  free(union_counts);

  int *diff_counts = BitDB_diff_count(left, right, opts, cpu);
  for (int i = 0; i < 4; ++i) {
    success = success && diff_counts[i] == expected_diff[i];
  }
  free(diff_counts);

  int *minus_counts = BitDB_minus_count(left, right, opts, cpu);
  for (int i = 0; i < 4; ++i) {
    success = success && minus_counts[i] == expected_minus[i];
  }
  free(minus_counts);

  Bit_free(&value);
  BitDB_free(&right);
  BitDB_free(&left);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_inter_count() {
#define SIZEOF_BITDB 45
  Bit_DB_T bit1 = BitDB_new(SIZE_OF_TEST_BIT, SIZEOF_BITDB);
  Bit_DB_T bit2 = BitDB_new(SIZE_OF_TEST_BIT, SIZEOF_BITDB);

  Bit_T bitset1 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bitset1, 1);
  Bit_bset(bitset1, 3);

  Bit_T bitset2 = Bit_new(SIZE_OF_TEST_BIT);
  Bit_bset(bitset2, 3);
  Bit_bset(bitset2, 5);

  BitDB_put_at(bit1, 0, bitset1);
  BitDB_put_at(bit2, 0, bitset2);

  Bit_bset(bitset1, 7); // Add an extra bit to bitset1
  Bit_bset(bitset2, 7); // Add an extra bit to bitset2

  BitDB_put_at(bit1, 1, bitset1);
  BitDB_put_at(bit2, 1, bitset2);

  int *count = BitDB_count(bit1);
  int *count2 = BitDB_count(bit2);

  int *inter_count = BitDB_inter_count(bit1, bit2, (SETOP_COUNT_OPTS){}, cpu);
  bool success = (*inter_count == 1) &&
                 (inter_count[1] == 1 && inter_count[SIZEOF_BITDB] == 1 &&
                  inter_count[SIZEOF_BITDB + 1] == 2);

  free(inter_count);
  free(count2);
  free(count);
  Bit_free(&bitset1);
  Bit_free(&bitset2);
  BitDB_free(&bit1);
  BitDB_free(&bit2);

  report_test(__func__, success);
  return success;
}

bool test_bitDB_store_macros() {
  Bit_DB_T left = BitDB_new(128, 2);
  Bit_DB_T right = BitDB_new(128, 2);
  Bit_T value = Bit_new(128);

  Bit_bset(value, 1);
  Bit_bset(value, 3);
  BitDB_put_at(left, 0, value);
  Bit_clear(value, 0, 127);
  Bit_bset(value, 2);
  BitDB_put_at(left, 1, value);

  Bit_clear(value, 0, 127);
  Bit_bset(value, 3);
  BitDB_put_at(right, 0, value);
  Bit_clear(value, 0, 127);
  Bit_bset(value, 1);
  Bit_bset(value, 2);
  BitDB_put_at(right, 1, value);

  const SETOP_COUNT_OPTS opts = {.num_cpu_threads = 1};
  int actual[4] = {0};
  const int expected_inter[4] = {1, 1, 0, 1};
  const int expected_union[4] = {2, 3, 2, 2};
  const int expected_diff[4] = {1, 2, 2, 1};
  const int expected_minus[4] = {1, 1, 1, 0};
  bool success = true;

  BitDB_inter_count_store(left, right, actual, opts, cpu);
  for (int i = 0; i < 4; ++i)
    success = success && actual[i] == expected_inter[i];

  BitDB_union_count_store(left, right, actual, opts, cpu);
  for (int i = 0; i < 4; ++i)
    success = success && actual[i] == expected_union[i];

  BitDB_diff_count_store(left, right, actual, opts, cpu);
  for (int i = 0; i < 4; ++i)
    success = success && actual[i] == expected_diff[i];

  BitDB_minus_count_store(left, right, actual, opts, cpu);
  for (int i = 0; i < 4; ++i)
    success = success && actual[i] == expected_minus[i];

  BitDB_inter_count_store(left, right, actual, opts, gpu);
  for (int i = 0; i < 4; ++i)
    success = success && actual[i] == expected_inter[i];

  Bit_free(&value);
  BitDB_free(&right);
  BitDB_free(&left);
  report_test(__func__, success);
  return success;
}

void run_tests() {
  printf("Running bit library tests...\n\n");

  // Basic operations
  test_bit_new();
  test_bit_set();
  test_bit_clear();
  test_bit_put();
  test_bit_set_range();
  test_bit_clear_range();
  test_bit_count();
  test_bit_aset_aclear();
  test_bit_not_and_map();
  test_bit_buffer_size_and_length();
  test_bit_invalid_index_handling();

  // Comparison operations
  test_bit_eq();
  test_bit_leq();
  test_bit_lt();

  // Set operations
  test_bit_union();
  test_bit_inter();
  test_bit_minus();
  test_bit_diff();

  // Count operations
  test_bit_count_operations();

  // Use external buffers
  test_bit_extract();
  test_bit_load();

  // Edge cases
  test_bit_null_handling();

  // BitDB tests
  test_bitDB_new();
  test_bitDB_properties();
  test_bitDB_count_at_and_count();
  test_bitDB_clear_ops();
  test_bitDB_get_put();
  test_bitDB_load_from_buffer();
  test_bitDB_invalid_index_handling();
  test_bitDB_count_macro_variants();
  test_bitDB_extract_replace();
  test_bitDB_inter_count();
  test_bitDB_store_macros();

  // Print summary
  printf("\nTest Summary:\n");
  printf("  Total:  %d\n", results.total);
  printf("  Passed: %d\n", results.passed);
  printf("  Failed: %d\n", results.failed);
}

int main() {
  // test whether we are in DEBUG mode;
#ifndef NDEBUG
  printf("Debug mode is enabled.\n");
#else
  printf("Debug mode is disabled.\n");
#endif

  run_tests();

  // Return non-zero exit code if any tests failed
  if (results.failed > 0) {
    printf("\nSome tests failed!\n");
    return 1;
  } else {
    printf("\nAll tests passed!\n");
    return 0;
  }
}