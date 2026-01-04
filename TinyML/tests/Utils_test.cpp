#include "unity.h"
#include "Utils.h"
#include <cmath>

void Test_TypicalSympnetDotProduct(void)
{
    const int8_t weights[] = {-127, 13};
    const int32_t hamiltonianCoordinates[] = {65536, 0};
    uint8_t length = 2;
    uint8_t fractionBits = 7;

    int32_t result = Dot_I8_I32_TO_I32(weights, hamiltonianCoordinates, length, fractionBits);
    TEST_ASSERT_EQUAL_INT32(-65024, result);
}

void Test_AllZerosDotProduct(void)
{
    const int8_t weights[] = {0 , 0};
    const int32_t hamiltonianCoordinates[] = {123456, -98765};
    uint8_t length = 2;
    uint8_t fractionBits = 7;

    int32_t result = Dot_I8_I32_TO_I32(weights, hamiltonianCoordinates, length, fractionBits);
    TEST_ASSERT_EQUAL_INT32(0, result);
}

void Test_MaxMagnitudeDotProduct(void)
{
    const int8_t weights[] = {127 , -128};
    const int32_t hamiltonianCoordinates[] = {INT32_MAX >> 1, INT32_MIN >> 1};
    uint8_t length = 2;
    uint8_t fractionBits = 7;

    int32_t result = Dot_I8_I32_TO_I32(weights, hamiltonianCoordinates, length, fractionBits);
    
    int64_t actual = 0;
    actual += (int64_t)weights[0] * hamiltonianCoordinates[0];
    actual += (int64_t)weights[1] * hamiltonianCoordinates[1];
    int32_t expected = (int32_t)(actual >> fractionBits);
    TEST_ASSERT_EQUAL_INT32(expected, result);
}