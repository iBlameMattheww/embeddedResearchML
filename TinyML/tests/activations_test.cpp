#include "unity.h"
#include "activations.h"
#include "utils.h"
#include <cmath>

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
    TEST_ASSERT_TRUE(rel_error < 0.0001); // 0.1% tolerance
}

void Test_Exp_Approx_Full_Range(void)
{
    const int32_t min_q16 = -8 * 65536;
    const int32_t max_q16 =  8 * 65536;
    const int32_t step = 1024;

    for (int32_t x_q16 = min_q16; x_q16 <= max_q16; x_q16 += step) {
        float x = x_q16 / 65536.0f;
        float expected = std::exp(x);
        float actual = Exp_Approx(x_q16) / 65536.0f;

        float abs_err = fabsf(actual - expected);
        float rel_err = fabsf(actual - expected) / expected;

        // absolute error OK for tiny exp(-x)
        bool ok_abs = abs_err < 1e-4f;

        // relative error OK for normal values
        bool ok_rel = rel_err < 0.008f;

        if (!(ok_abs || ok_rel)) {
            printf("FAIL: x=%f expected=%f actual=%f abs_err=%f rel_err=%f\n",
                   x, expected, actual, abs_err, rel_err);
            UNITY_TEST_FAIL(__LINE__, "Exp_Approx error too high");
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
