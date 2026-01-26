#include "unity.h"
#include "Activations.h"
#include <stdint.h>

/* ============================================================
 * Helpers
 * ============================================================ */

#define Q16(x) ((int32_t)((x) * 65536))

/* ============================================================
 * Test: Vanilla_Init wires model metadata
 * ============================================================ */

void Test_Vanilla_Init_SetsLayerCount(void)
{
    vanillaModel_t model = {0};

    Vanilla_Init(&model, 2);

    TEST_ASSERT_EQUAL_UINT8(2, model._private.numLayers);
    TEST_ASSERT_NOT_NULL(model._private.layers);
}

/* ============================================================
 * Test: VanillaStep is deterministic
 * ============================================================ */

void Test_VanillaStep_IsDeterministic(void)
{
    phaseState_t a = { .p = Q16(0.5), .q = Q16(-0.25) };
    phaseState_t b = a;

    VanillaStep(NULL, &a, Q16(1.0));
    VanillaStep(NULL, &b, Q16(1.0));

    TEST_ASSERT_EQUAL_INT32(a.p, b.p);
    TEST_ASSERT_EQUAL_INT32(a.q, b.q);
}

/* ============================================================
 * Test: Step size scaling behaves linearly (Euler property)
 * ============================================================ */

void Test_VanillaStep_StepSizeScaling(void)
{
    /*
        Euler integration property:
        dx(h) ≈ 2 * dx(h/2)
    */

    phaseState_t s_half = { .p = Q16(1.0), .q = Q16(1.0) };
    phaseState_t s_full = s_half;

    VanillaStep(NULL, &s_half, Q16(0.5));
    VanillaStep(NULL, &s_full, Q16(1.0));

    int32_t dp_half = s_half.p - Q16(1.0);
    int32_t dq_half = s_half.q - Q16(1.0);

    int32_t dp_full = s_full.p - Q16(1.0);
    int32_t dq_full = s_full.q - Q16(1.0);

    /* Allow small fixed-point rounding error */
    TEST_ASSERT_INT32_WITHIN(4, dp_full / 2, dp_half);
    TEST_ASSERT_INT32_WITHIN(4, dq_full / 2, dq_half);
}

/* ============================================================
 * Test: Zero step size produces no state change
 * ============================================================ */

void Test_VanillaStep_ZeroStepSize_NoChange(void)
{
    phaseState_t state = { .p = Q16(2.0), .q = Q16(-3.0) };
    phaseState_t original = state;

    VanillaStep(NULL, &state, Q16(0.0));

    TEST_ASSERT_EQUAL_INT32(original.p, state.p);
    TEST_ASSERT_EQUAL_INT32(original.q, state.q);
}

/* ============================================================
 * Test: State remains finite (no overflow / NaN behavior)
 * ============================================================ */

void Test_VanillaStep_StateRemainsFinite(void)
{
    phaseState_t state = {
        .p = Q16(10.0),
        .q = Q16(-10.0)
    };

    for (int i = 0; i < 100; i++)
    {
        VanillaStep(NULL, &state, Q16(0.1));
    }

    TEST_ASSERT_TRUE(state.p > INT32_MIN);
    TEST_ASSERT_TRUE(state.p < INT32_MAX);
    TEST_ASSERT_TRUE(state.q > INT32_MIN);
    TEST_ASSERT_TRUE(state.q < INT32_MAX);
}
