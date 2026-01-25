#include "Activations.h"

static int32_t Relu(int32_t x)
{
    x = CLAMP(x, 0, INT32_MAX);
    return x;
}

static void Dense_ReLU(
    const int32_t *input,
    int32_t *output,
    const int32_t *weights,
    const int32_t *biases,
    uint8_t inputSize,
    uint8_t outputSize
)
{
    for (uint8_t i = 0; i < outputSize; i++)
    {
        int64_t acc = biases[i];

        for (uint8_t j = 0; j < inputSize; j++)
        {
            acc += ((int64_t)weights[i * inputSize + j] * input[j]) >> 16;
        }

        acc = Relu((int32_t)acc);
        output[i] = CLAMP((int32_t)acc, INT32_MIN, INT32_MAX);
    }
}

static void Dense_Linear(
    const int32_t *input,
    int32_t *output,
    const int32_t *weights,
    const int32_t *biases,
    uint8_t inputSize,
    uint8_t outputSize
)
{
    for (uint8_t i = 0; i < outputSize; i++)
    {
        int64_t acc = biases[i];

        for (uint8_t j = 0; j < inputSize; j++)
        {
            acc += ((int64_t)weights[i * inputSize + j] * input[j]) >> 16;
        }

        output[i] = CLAMP((int32_t)acc, INT32_MIN, INT32_MAX);
    }
}

static void VanillaForward(
    const phaseState_t *state,
    int32_t *dx
)
{
    int32_t x0[2] = { state->p, state->q };  // Q16.16
    int32_t h1[FC1_OUT_DIM];
    int32_t out[2];

    Dense_ReLU(
    x0,
    h1,
    (const int32_t *)fc1_weights,
    fc1_biases,
    FC1_IN_DIM,
    FC1_OUT_DIM
    );


    Dense_Linear(
        h1,
        out,
        (const int32_t *)fc2_weights,
        fc2_biases,
        FC2_IN_DIM,
        FC2_OUT_DIM
    );

    dx[0] = out[0];
    dx[1] = out[1];
}

static void VanillaEulerUpdate(phaseState_t *state, const int32_t *dx, int32_t stepSize)
{
    state->p += (((int64_t)stepSize * dx[0]) >> 16);
    state->q += (((int64_t)stepSize * dx[1]) >> 16);

    state->p = CLAMP(state->p, INT32_MIN, INT32_MAX);
    state->q = CLAMP(state->q, INT32_MIN, INT32_MAX);
}

void VanillaStep(vanillaModel_t *model, phaseState_t *state, int32_t stepSize)
{
    (void) model;
    int32_t dx[2] = {0};

    VanillaForward(state, dx);
    VanillaEulerUpdate(state, dx, stepSize);
}

void Vanilla_Init(vanillaModel_t *model, uint8_t numLayers)
{
    static vanillaLayer_t layers[] = {
        {._private = {
            .biases = fc1_biases,
            .weights = (const int32_t*)fc1_weights,
            .inputSize = FC1_IN_DIM,
            .outputSize = FC1_OUT_DIM
        }},
        {._private = {
            .biases = fc2_biases,
            .weights = (const int32_t*)fc2_weights,
            .inputSize = FC2_IN_DIM,
            .outputSize = FC2_OUT_DIM
        }}
    };
    model->_private.layers = layers;
    model->_private.numLayers = numLayers;
}
