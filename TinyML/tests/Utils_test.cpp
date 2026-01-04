#include "unity.h"
#include "Utils.h"
#include <limits.h>

/* -------------------------------------------------
 * Dot_Q16_Q16_TO_Q16
 * ------------------------------------------------- */

void Test_TypicalSympnetDotProduct(void)
{
    // weights = [-1.0, 0.2]
    const int32_t weights[] = {
        -65536,
         13107
    };

    // coordinates = [1.0, 0.0]
    const int32_t hamiltonianCoordinates[] = {
        65536,
        0
    };

    uint8_t length = 2;

    // Expected:
    // (-1.0 * 1.0) + (0.2 * 0.0) = -1.0
    // Q16.16 → -65536
    int32_t result = Dot_Q16_Q16_TO_Q16(
        weights,
        hamiltonianCoordinates,
        length
    );

    TEST_ASSERT_INT32_WITHIN(1, -65536, result);
}

void Test_AllZerosDotProduct(void)
{
    const int32_t weights[] = { 0, 0 };
    const int32_t hamiltonianCoordinates[] = {
        123456,
        -98765
    };

    uint8_t length = 2;

    int32_t result = Dot_Q16_Q16_TO_Q16(
        weights,
        hamiltonianCoordinates,
        length
    );

    TEST_ASSERT_EQUAL_INT32(0, result);
}

void Test_SignHandlingDotProduct(void)
{
    // weights = [0.5, -0.5]
    const int32_t weights[] = {
        32768,
       -32768
    };

    // coordinates = [2.0, 1.0]
    const int32_t hamiltonianCoordinates[] = {
        131072,
        65536
    };

    uint8_t length = 2;

    // (0.5 * 2.0) + (-0.5 * 1.0) = 0.5
    // → 32768
    int32_t result = Dot_Q16_Q16_TO_Q16(
        weights,
        hamiltonianCoordinates,
        length
    );

    TEST_ASSERT_INT32_WITHIN(1, 32768, result);
}
