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

void Test_Exp_Approx_0(void)
{
    float expected = std::exp(0);
    int32_t actual_q16 = Exp_Approx(0 * 65536); // Q16 fixed-point result
    double actual = actual_q16 / 65536.0;
    double rel_error = std::abs(actual - expected) / expected;
    TEST_ASSERT_TRUE(rel_error < 0.02); // 2% tolerance
}

void Test_Exp_Approx_Full_Range(void)
{
    const int32_t min_q16 = -8 * 65536;
    const int32_t max_q16 =  8 * 65536;
    const int32_t step = 1024;

    for (int32_t x_q16 = min_q16; x_q16 <= max_q16; x_q16 += step) {
        float x = x_q16 / 65536.0f;
        float expected = std::exp(x);
        int32_t actual_q16 = Exp_Approx(x_q16);
        float actual = actual_q16 / 65536.0f;
        float rel_error = fabsf(actual - expected) / expected;
        if (rel_error >= 0.035) {
            printf("FAIL: x=%f (q16=%d), expected=%f, actual=%f, rel_error=%f\n", x, x_q16, expected, actual, rel_error);
            UNITY_TEST_FAIL(__LINE__, "Exp_Approx relative error too high");
        }
    }
}

void Test_Softmax_SineDataset_SingleLogit_Returns255(void)
{
    int16_t logits[] = {1234};   // arbitrary value
    Softmax(logits, 1);
    TEST_ASSERT_EQUAL_UINT16(255, logits[0]);
}

void test_Softmax_LorenzDataset_3Values_NormalizedAndScaled(void)
{
    int16_t logits[] = { 1200, 800, 400 };
    Softmax(logits, 3);
    uint16_t sum = logits[0] + logits[1] + logits[2];
    TEST_ASSERT_INT_WITHIN(2, 255, sum);   // allow rounding error ±2
    TEST_ASSERT_TRUE(logits[0] > logits[1]);
    TEST_ASSERT_TRUE(logits[1] >= logits[2]);
}

void test_Softmax_NegativeValues_StableNormalization(void)
{
    int16_t logits[] = { -300, -400, -500 };
    Softmax(logits, 3);

    for (int i = 0; i < 3; i++)
    {
        TEST_ASSERT_TRUE(logits[i] >= 0);
        TEST_ASSERT_TRUE(logits[i] <= 255);
    }

    uint16_t sum = logits[0] + logits[1] + logits[2];
    TEST_ASSERT_INT_WITHIN(2, 255, sum);
}

void test_Softmax_LargeValues_NoOverflow(void)
{
    int16_t logits[] = { 30000, 29000, 28000 };
    Softmax(logits, 3);

    for (int i = 0; i < 3; i++)
        TEST_ASSERT_TRUE(logits[i] <= 255);

    uint16_t sum = logits[0] + logits[1] + logits[2];
    TEST_ASSERT_INT_WITHIN(2, 255, sum);
}

int main(void) 
{
    UNITY_BEGIN();
    RUN_TEST(Test_Relu_Positive);
    RUN_TEST(Test_Relu_Zero);
    RUN_TEST(Test_Relu_Negative);
    RUN_TEST(Test_Exp_Approx_0);
    RUN_TEST(Test_Exp_Approx_Full_Range);
    RUN_TEST(Test_Softmax_SineDataset_SingleLogit_Returns255);
    RUN_TEST(test_Softmax_LorenzDataset_3Values_NormalizedAndScaled);
    RUN_TEST(test_Softmax_NegativeValues_StableNormalization);
    RUN_TEST(test_Softmax_LargeValues_NoOverflow);
    return UNITY_END();
}