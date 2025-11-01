#include "unity.h"
#include "activations.h"
#include "utils.h"
#include <cmath>


void setUp(void) 
{
    // Optional: code to run before each test
}

void tearDown(void) 
{
    // Optional: code to run after each test
}

void Test_Relu_Positive(void) 
{
    TEST_ASSERT_EQUAL_UINT8(5, Relu(5));
}

void Test_Relu_Zero(void) 
{
    TEST_ASSERT_EQUAL_UINT8(0, Relu(0));
}

void Test_Relu_Negative(void) 
{
    TEST_ASSERT_EQUAL_UINT8(0, Relu(-7));
}

void Test_Exp_Approx_4(void)
{
    float expected = std::exp(4);
    int32_t actual_q16 = Exp_Approx(4 * 65536); // Q16 fixed-point result
    double actual = actual_q16 / 65536.0;      // Convert Q16 back to float
    double rel_error = std::abs(actual - expected) / expected;
    TEST_ASSERT_TRUE(rel_error < 0.02); // 2% tolerance
}

void Test_Exp_Approx_Minus4(void)
{
    float expected = std::exp(-4);
    int32_t actual_q16 = Exp_Approx(-4 * 65536); // Q16 fixed-point result
    double actual = actual_q16 / 65536.0;
    double rel_error = std::abs(actual - expected) / expected;
    TEST_ASSERT_TRUE(rel_error < 0.02); // 2% tolerance
}

void Test_Exp_Approx_0(void)
{
    float expected = std::exp(0);
    int32_t actual_q16 = Exp_Approx(0 * 65536); // Q16 fixed-point result
    double actual = actual_q16 / 65536.0;
    double rel_error = std::abs(actual - expected) / expected;
    TEST_ASSERT_TRUE(rel_error < 0.02); // 2% tolerance
}

void Test_Exp_Approx_8(void)
{
    float expected = std::exp(8);
    int32_t actual_q16 = Exp_Approx(8 * 65536); // Q16 fixed-point result
    double actual = actual_q16 / 65536.0;
    double rel_error = std::abs(actual - expected) / expected;
    TEST_ASSERT_TRUE(rel_error < 0.02); // 2% tolerance
}

void Test_Exp_Approx_Minus8(void)
{
    float expected = std::exp(-8);
    int32_t actual_q16 = Exp_Approx(-8 * 65536); // Q16 fixed-point result
    double actual = actual_q16 / 65536.0;
    double rel_error = std::abs(actual - expected) / expected;
    TEST_ASSERT_TRUE(rel_error < 0.02); // 2% tolerance
}



int main(void) 
{
    UNITY_BEGIN();
    RUN_TEST(Test_Relu_Positive);
    RUN_TEST(Test_Relu_Zero);
    RUN_TEST(Test_Relu_Negative);
    RUN_TEST(Test_Exp_Approx_4);
    RUN_TEST(Test_Exp_Approx_Minus4);
    RUN_TEST(Test_Exp_Approx_0);
    RUN_TEST(Test_Exp_Approx_8);
    RUN_TEST(Test_Exp_Approx_Minus8);
    return UNITY_END();
}