#include "unity.h"
#include "Sympnet.h"

/* -------------------------------------------------
 * PolynomialDerivation
 * ------------------------------------------------- */

void Test_SingleCoefficientPolynomialDerivation(void)
{
    // a0 = -0.5 in Q16.16
    const int32_t coefficients[] = { -32768 };
    int32_t m = 6554;                 // ≈ 0.1
    uint8_t n = 1;

    // dH = 2 * a0 * m
    // = 2 * (-0.5) * 0.1 = -0.1
    // ≈ -6554
    int32_t result = PolynomialDerivation(coefficients, n, m);

    TEST_ASSERT_INT32_WITHIN(2, -6554, result);
}

void Test_ZeroMPolynomialDerivation(void)
{
    const int32_t coefficients[] = {
        -32768,   // -0.5
         65536,   // 1.0
        -131072   // -2.0
    };

    int32_t m = 0;
    uint8_t n = 3;

    int32_t result = PolynomialDerivation(coefficients, n, m);
    TEST_ASSERT_EQUAL_INT32(0, result);
}

void Test_NegativeMPolynomialDerivation(void)
{
    const int32_t coefficients[] = { -32768 };
    int32_t m = -6554;    // -0.1
    uint8_t n = 1;

    // dH = 2 * (-0.5) * (-0.1) = +0.1
    int32_t result = PolynomialDerivation(coefficients, n, m);

    TEST_ASSERT_INT32_WITHIN(2, 6554, result);
}

/* -------------------------------------------------
 * SymplecticTimeScale
 * ------------------------------------------------- */

void Test_TypicalSymplecticTimeScale(void)
{
    int32_t h  = 3277;    // ≈ 0.05
    int32_t dH = -13107;  // ≈ -0.2

    // h * dH ≈ -0.01 → -655
    int32_t result = SymplecticTimeScale(h, dH);

    TEST_ASSERT_INT32_WITHIN(2, -655, result);
}

void Test_ZeroDerivativeSymplecticTimeScale(void)
{
    int32_t h  = 3277;
    int32_t dH = 0;

    TEST_ASSERT_EQUAL_INT32(0, SymplecticTimeScale(h, dH));
}

void Test_SignHandlingSymplecticTimeScale(void)
{
    int32_t h  = -3277;
    int32_t dH = -13107;

    TEST_ASSERT_TRUE(SymplecticTimeScale(h, dH) > 0);
}

/* -------------------------------------------------
 * State Update
 * ------------------------------------------------- */

void Test_TypicalStateUpdate(void)
{
    phaseState_t state = {
        .p = 65536,   // 1.0
        .q = 0
    };

    int32_t scale = -655;   // ≈ -0.01

    // weights = [-2.0, 0.25]
    int32_t weights[2] = {
        -131072,
         16384
    };

    SympnetStateUpdate(&state, scale, weights);

    int32_t expected_p = 65536 + ((int32_t)(((int64_t)scale * weights[1]) >> 16));
    int32_t expected_q = 0     - ((int32_t)(((int64_t)scale * weights[0]) >> 16));

    TEST_ASSERT_INT32_WITHIN(2, expected_p, state.p);
    TEST_ASSERT_INT32_WITHIN(2, expected_q, state.q);
}

void Test_ZeroScaleStateUpdate(void)
{
    phaseState_t state = { .p = 12345, .q = -6789 };
    int32_t scale = 0;

    int32_t weights[2] = { 32768, -32768 };

    SympnetStateUpdate(&state, scale, weights);

    TEST_ASSERT_EQUAL_INT32(12345, state.p);
    TEST_ASSERT_EQUAL_INT32(-6789, state.q);
}

void Test_SignBehaviorStateUpdate(void)
{
    phaseState_t state = { .p = 0, .q = 0 };

    int32_t scale = 1000;
    int32_t weights[2] = { -32768, 65536 };

    SympnetStateUpdate(&state, scale, weights);

    TEST_ASSERT_TRUE(state.p > 0);
    TEST_ASSERT_TRUE(state.q > 0);
}

/* -------------------------------------------------
 * Layer + Model
 * ------------------------------------------------- */

void Test_SingleLayerSympnetStep(void)
{
    phaseState_t state = { .p = 65536, .q = 0 };

    static const int32_t coeffs[]  = { -32768 };      // -0.5
    static const int32_t weights[] = { -65536, 32768 }; // [-1.0, 0.5]

    symplecticLayer_t layer = {
        ._private = {
            .coefficients    = coeffs,
            .weights         = weights,
            .numCoefficients = 1
        }
    };

    int32_t stepSize = 3277;

    SympnetLayerStep(&state, &layer, stepSize);

    TEST_ASSERT_NOT_EQUAL(65536, state.p);
    TEST_ASSERT_NOT_EQUAL(0,     state.q);
}