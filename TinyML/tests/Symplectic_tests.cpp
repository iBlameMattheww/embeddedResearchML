#include "unity.h"
#include "Sympnet.h"


void Test_SingleCoefficientPolynomialDerivation(void)
{
    const int16_t coefficients[] = {-32767};
    int32_t m = 6554;
    uint8_t numberOfCoefficients = 1;

    int32_t result = PolynomialDerivation(coefficients, numberOfCoefficients, m);
    TEST_ASSERT_EQUAL_INT32(-13108, result);
}

void Test_ZeroMPolynomialDerivation(void)
{
    const int16_t coefficients[] = {-32767, 1000, -2000};
    int32_t m = 0;
    uint8_t numberOfCoefficients = 3;

    int32_t result = PolynomialDerivation(coefficients, numberOfCoefficients, m);
    TEST_ASSERT_EQUAL_INT32(0, result);
}

void Test_NegativeMPolynomialDerivation(void)
{
    const int16_t coefficients[] = {-32767};
    int32_t m = -6554;
    uint8_t numberOfCoefficients = 1;

    int32_t result = PolynomialDerivation(coefficients, numberOfCoefficients, m);
    TEST_ASSERT_EQUAL_INT32(13106, result);
}

void Test_TypicalSymplecticTimeScale(void)
{
    int32_t h = 3277;
    int32_t dH = -13107;

    int32_t result = SymplecticTimeScale(h, dH);
    TEST_ASSERT_INT32_WITHIN(2, -655, result);
}

void Test_ZeroDerivativeSymplecticTimeScale(void)
{
    int32_t h = 3277;
    int32_t dH = 0;

    int32_t result = SymplecticTimeScale(h, dH);
    TEST_ASSERT_EQUAL_INT32(0, result);
}

void Test_SignHandlingSyplecticTimeScale(void)
{
    int32_t h = -3277;
    int32_t dH = -13107;

    int32_t result = SymplecticTimeScale(h, dH);
    TEST_ASSERT_TRUE(result > 0);
}

void Test_TypicalStateUpdate(void)
{
    phaseState_t state = {.p = 65536, .q = 0};
    int32_t scale = -655;
    int8_t weights[2] = {-127, 13};

    SympnetStateUpdate(&state, scale, weights);

    int32_t expected_p = 65536 + ((scale * 13) >> 7);
    int32_t expected_q = 0 - ((scale * -127) >> 7);

    TEST_ASSERT_INT32_WITHIN(2, expected_p, state.p);
    TEST_ASSERT_INT32_WITHIN(2, expected_q, state.q);
}

void Test_ZeroScaleStateUpdate(void)
{
    phaseState_t state = {.p = 12345, .q = -6789};
    int32_t scale = 0;
    int8_t weights[2] = {50, -50};

    SympnetStateUpdate(&state, scale, weights);
    TEST_ASSERT_EQUAL_INT32(12345, state.p);
    TEST_ASSERT_EQUAL_INT32(-6789, state.q);
}

void Test_SignBehaviorStateUpdate(void)
{
    phaseState_t state = {.p = 0, .q = 0};
    int32_t scale = 1000;
    int8_t weights[2] = {-10, 20};

    SympnetStateUpdate(&state, scale, weights);

    TEST_ASSERT_TRUE(state.p > 0);   // +scale * +w1
    TEST_ASSERT_TRUE(state.q > 0);   // -scale * -w0
}

void Test_SingleLayerSympnetStep(void)
{
    phaseState_t state = {.p = 65536, .q = 0};

    symplecticLayer_t layer = {._private = {
        .coefficients = (const int16_t[]){-32767},
        .weights = (const int8_t[]){-127, 13},      
        .numCoefficients = 1
    }};

    int32_t stepSize = 3277;

    SympnetLayerStep(&state, &layer, stepSize);

    TEST_ASSERT_INT32_WITHIN(16, 66197, state.p);
    TEST_ASSERT_INT32_WITHIN(16, 6464,  state.q);
}

// void Test_MultiLayerSympnetStep(void)
// {
//     phaseState_t state = {.p = 65536, .q = 0};

//     symplecticLayer_t layers[1] = {
//      {._private = {
//         .coefficients = (const int16_t[]){-32767},
//         .weights = (const int8_t[]){-127, 13},      
//         .numCoefficients = 1
//     }}
//     };

//     symplecticModel_t model = {._private = {
//         .layers = layers,
//         .numLayers = 1
//     }};

//     int32_t stepSize = 3277;

//     SympnetRollout(&model, &state, stepSize);

//     TEST_ASSERT_NOT_EQUAL(65536, state.p);
//     TEST_ASSERT_NOT_EQUAL(0,     state.q);

//     TEST_ASSERT_GREATER_THAN(65536, state.p);
//     TEST_ASSERT_GREATER_THAN(0,     state.q);

//     TEST_ASSERT_INT32_WITHIN(16, 66197, state.p);
//     TEST_ASSERT_INT32_WITHIN(16, 6464,  state.q);
// }