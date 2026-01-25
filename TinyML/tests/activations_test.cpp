#include "unity.h"
#include "Activations.h"
#include <stdint.h>

/* ============================================================
 * Helpers
 * ============================================================ */

#define Q16(x) ((int32_t)((x) * 65536))

/* ============================================================
 * Test: Vanilla_Init wires layers correctly
 * ============================================================ */

void Test_Vanilla_Init_SetsLayerCount(void)
{
    vanillaModel_t model = {0};

    Vanilla_Init(&model, 2);

    TEST_ASSERT_EQUAL_UINT8(2, model._private.numLayers);
    TEST_ASSERT_NOT_NULL(model._private.layers);
}

/* ============================================================
 * Test: Single-layer Euler step, positive region (ReLU active)
 * ============================================================ */

void Test_VanillaStep_SingleLayer_PositiveReLU(void)
{
    /*
        dx = W * [p q] + b
        Use identity weights so dx = [p q]
        stepSize = 1.0

        p_next = p + p
        q_next = q + q
    */

    static int32_t weights[] = {
        Q16(1.0), Q16(0.0),
        Q16(0.0), Q16(1.0)
    };

    static int32_t biases[] = {
        Q16(0.0),
        Q16(0.0)
    };

    static vanillaLayer_t layers[] = {
        { ._private = {
            .weights = weights,
            .biases = biases,
            .inputSize = 2,
            .outputSize = 2
        }}
    };

    vanillaModel_t model = {
        ._private = {
            .layers = layers,
            .numLayers = 1
        }
    };

    phaseState_t state = {
        .p = Q16(2.0),
        .q = Q16(3.0)
    };

    VanillaStep(&model, &state, Q16(1.0));

    TEST_ASSERT_EQUAL_INT32(Q16(4.0), state.p);
    TEST_ASSERT_EQUAL_INT32(Q16(6.0), state.q);
}

/* ============================================================
 * Test: ReLU clamps negative derivative to zero
 * ============================================================ */

void Test_VanillaStep_ReLU_ClampsNegative(void)
{
    /*
        dx = -p, -q  → ReLU → 0
        state should not change
    */

    static int32_t weights[] = {
        Q16(-1.0), Q16(0.0),
        Q16(0.0),  Q16(-1.0)
    };

    static int32_t biases[] = {
        Q16(0.0),
        Q16(0.0)
    };

    static vanillaLayer_t layers[] = {
        { ._private = {
            .weights = weights,
            .biases = biases,
            .inputSize = 2,
            .outputSize = 2
        }}
    };

    vanillaModel_t model = {
        ._private = {
            .layers = layers,
            .numLayers = 1
        }
    };

    phaseState_t state = {
        .p = Q16(2.0),
        .q = Q16(3.0)
    };

    VanillaStep(&model, &state, Q16(1.0));

    TEST_ASSERT_EQUAL_INT32(Q16(2.0), state.p);
    TEST_ASSERT_EQUAL_INT32(Q16(3.0), state.q);
}

/* ============================================================
 * Test: Step size scaling works correctly
 * ============================================================ */

void Test_VanillaStep_StepSizeScaling(void)
{
    /*
        dx = [1, 1]
        stepSize = 0.5

        p_next = p + 0.5
        q_next = q + 0.5
    */

    static int32_t weights[] = {
        Q16(0.0), Q16(0.0),
        Q16(0.0), Q16(0.0)
    };

    static int32_t biases[] = {
        Q16(1.0),
        Q16(1.0)
    };

    static vanillaLayer_t layers[] = {
        { ._private = {
            .weights = weights,
            .biases = biases,
            .inputSize = 2,
            .outputSize = 2
        }}
    };

    vanillaModel_t model = {
        ._private = {
            .layers = layers,
            .numLayers = 1
        }
    };

    phaseState_t state = {
        .p = Q16(1.0),
        .q = Q16(1.0)
    };

    VanillaStep(&model, &state, Q16(0.5));

    TEST_ASSERT_EQUAL_INT32(Q16(1.5), state.p);
    TEST_ASSERT_EQUAL_INT32(Q16(1.5), state.q);
}

/* ============================================================
 * Test: Multiple layers compose correctly
 * ============================================================ */

void Test_VanillaStep_TwoLayers_Compose(void)
{
    /*
        Layer 1: dx = [1, 1]
        Layer 2: dx = [1, 1]
        step = 1

        Net effect: +2 on p and q
    */

    static int32_t weights1[] = {
        Q16(0.0), Q16(0.0),
        Q16(0.0), Q16(0.0)
    };

    static int32_t biases1[] = {
        Q16(1.0),
        Q16(1.0)
    };

    static int32_t weights2[] = {
        Q16(0.0), Q16(0.0),
        Q16(0.0), Q16(0.0)
    };

    static int32_t biases2[] = {
        Q16(1.0),
        Q16(1.0)
    };

    static vanillaLayer_t layers[] = {
        { ._private = {
            .weights = weights1,
            .biases = biases1,
            .inputSize = 2,
            .outputSize = 2
        }},
        { ._private = {
            .weights = weights2,
            .biases = biases2,
            .inputSize = 2,
            .outputSize = 2
        }}
    };

    vanillaModel_t model = {
        ._private = {
            .layers = layers,
            .numLayers = 2
        }
    };

    phaseState_t state = {
        .p = Q16(0.0),
        .q = Q16(0.0)
    };

    VanillaStep(&model, &state, Q16(1.0));

    TEST_ASSERT_EQUAL_INT32(Q16(2.0), state.p);
    TEST_ASSERT_EQUAL_INT32(Q16(2.0), state.q);
}
