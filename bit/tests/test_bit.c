#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "bit.h"

#define SIZE_OF_TEST_BIT 2048
typedef struct {
    int total;
    int passed;
    int failed;
} TestResults;

// Initialize test results
TestResults results = {0, 0, 0};

// Utility function to print test results
void report_test(const char* test_name, bool passed) {
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
    report_test("test_bit_new", success);
    Bit_free(bit);
    return success;
}

bool test_bit_set() {
    Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
    Bit_bset(bit, 2);
    bool success = (Bit_get(bit, 2) == 1);
    report_test("test_bit_set", success);
    Bit_free(bit);
    return success;
}

bool test_bit_clear() {
    Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
    Bit_bset(bit, 2);
    Bit_bclear(bit, 2);
    bool success = (Bit_get(bit, 2) == 0);
    report_test("test_bit_clear", success);
    Bit_free(bit);
    return success;
}

bool test_bit_put() {
    Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
    int prev = Bit_put(bit, 3, 1);
    bool success = (prev == 0 && Bit_get(bit, 3) == 1);
    
    prev = Bit_put(bit, 3, 0);
    success = success && (prev == 1 && Bit_get(bit, 3) == 0);
    
    report_test("test_bit_put", success);
    Bit_free(bit);
    return success;
}

bool test_bit_set_range() {
    Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
    Bit_set(bit, 2, SIZE_OF_TEST_BIT/2); // Set bits 2,3,4,5
    
    bool success = 1 ;
    
    for(int index = 2; index <=SIZE_OF_TEST_BIT/2; index++)
        success = success && Bit_get(bit,index) == 1; 
                   
    report_test("test_bit_set_range", success);
    Bit_free(bit);
    return success;
}

bool test_bit_clear_range() {
    Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
    // Set all bits
    for (int i = 0; i < SIZE_OF_TEST_BIT/2; i++) {
        Bit_bset(bit, i);
    }
    
    Bit_clear(bit, 2, 5); // Clear bits 2,3,4,5
    
    bool success = (Bit_get(bit, 2) == 0 && 
                   Bit_get(bit, 3) == 0 &&
                   Bit_get(bit, 4) == 0 &&
                   Bit_get(bit, 5) == 0 &&
                   Bit_get(bit, 1) == 1);
    for(int index = 6; index < SIZE_OF_TEST_BIT/2; index++)
        success = success && (Bit_get(bit, index) == 1);
            
    report_test("test_bit_clear_range", success);
    Bit_free(bit);
    return success;
}



bool test_bit_count() {
    Bit_T bit = Bit_new(SIZE_OF_TEST_BIT);
    Bit_bset(bit, 1);
    Bit_bset(bit, 3);
    Bit_bset(bit, SIZE_OF_TEST_BIT/2);
    
    int count = Bit_count(bit);
    bool success = (count == 3);
    
    report_test("test_bit_count", success);
    Bit_free(bit);
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
    Bit_bclear(bit2,8);
    Bit_bset(bit2,75);
    success = success && !Bit_eq(bit1, bit2);
    report_test("test_bit_eq", success);
    Bit_free(bit1);
    Bit_free(bit2);
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
    
    report_test("test_bit_leq", success);
    Bit_free(bit1);
    Bit_free(bit2);
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
    
    report_test("test_bit_lt", success);
    Bit_free(bit1);
    Bit_free(bit2);
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
    
    bool success = (Bit_get(union_bit, 1) == 1 &&
                   Bit_get(union_bit, 3) == 1 &&
                   Bit_get(union_bit, 5) == 1 &&
                   Bit_get(union_bit, 0) == 0 &&
                   Bit_get(union_bit, 2) == 0 &&
                   Bit_get(union_bit, 4) == 0);
                   
    report_test("test_bit_union", success);
    Bit_free(bit1);
    Bit_free(bit2);
    Bit_free(union_bit);
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
    
    bool success = (Bit_get(inter_bit, 3) == 1 &&
                   Bit_get(inter_bit, 5) == 1 &&
                   Bit_get(inter_bit, 1) == 0 &&
                   Bit_get(inter_bit, 7) == 0);
                   
    report_test("test_bit_inter", success);
    Bit_free(bit1);
    Bit_free(bit2);
    Bit_free(inter_bit);
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
    
    bool success = (Bit_get(minus_bit, 1) == 1 &&
                   Bit_get(minus_bit, 3) == 0 &&
                   Bit_get(minus_bit, 5) == 0 &&
                   Bit_get(minus_bit, 7) == 0);
                   
    report_test("test_bit_minus", success);
    Bit_free(bit1);
    Bit_free(bit2);
    Bit_free(minus_bit);
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
    
    bool success = (Bit_get(diff_bit, 1) == 1 &&
                   Bit_get(diff_bit, 7) == 1 &&
                   Bit_get(diff_bit, 3) == 0 &&
                   Bit_get(diff_bit, 5) == 0);
                   
    report_test("test_bit_diff", success);
    Bit_free(bit1);
    Bit_free(bit2);
    Bit_free(diff_bit);
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
    
    int union_count = Bit_union_count(bit1, bit2);
    int inter_count = Bit_inter_count(bit1, bit2);
    int minus_count = Bit_minus_count(bit1, bit2);
    int diff_count = Bit_diff_count(bit1, bit2);
    
    bool success = (union_count == 4 &&
                   inter_count == 2 &&
                   minus_count == 1 &&
                   diff_count == 2);
                   
    report_test("test_bit_count_operations", success);
    Bit_free(bit1);
    Bit_free(bit2);
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
    
    Bit_free(bit);
    Bit_free(union_result);
    Bit_free(inter_result);
    Bit_free(minus_result);
    
    report_test("test_bit_null_handling", success);
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
    
    // Edge cases
    test_bit_null_handling();
    
    // Print summary
    printf("\nTest Summary:\n");
    printf("  Total:  %d\n", results.total);
    printf("  Passed: %d\n", results.passed);
    printf("  Failed: %d\n", results.failed);
}

int main() {
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