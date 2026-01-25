#include "Activations.h"

static int32_t Relu(int32_t x)
{
    x = CLAMP(x, 0, INT32_MAX);
    return x;
}

static void VanillaLayerDerivative(const phaseState_t *state, const vanillaLayer_t *layer, int32_t *dx)
{
    int32_t input[2] = { state->p, state->q };  // Q16.16

    for (uint8_t i = 0; i < layer->_private.outputSize; i++)
    {
        int64_t acc = layer->_private.biases[i];  // Q16.16
        
        for (uint8_t j = 0; j < layer->_private.inputSize; j++)
        {
            int32_t weight = layer->_private.weights[i * layer->_private.inputSize + j];
            acc += ((int64_t)weight * input[j]) >> 16;  // Q16.16
        }

        acc = Relu((int32_t)acc);  // Q16.16
        dx[i] = CLAMP((int32_t)acc, INT32_MIN, INT32_MAX);  // Q16.16
    }
}

static void VanillaEulerUpdate(phaseState_t *state, const int32_t *dx, int32_t stepSize)
{
    state->p += (((int64_t)stepSize * dx[0]) >> 16);
    state->q += (((int64_t)stepSize * dx[1]) >> 16);

    state->p = CLAMP(state->p, INT32_MIN, INT32_MAX);
    state->q = CLAMP(state->q, INT32_MIN, INT32_MAX);
}

static void VanillaLayerStep(phaseState_t *state, const vanillaLayer_t *layer, int32_t stepSize)
{
    int32_t dx[2] = {0};

    VanillaLayerDerivative(state, layer, dx);
    VanillaEulerUpdate(state, dx, stepSize);
}

void VanillaStep(vanillaModel_t *model, phaseState_t *state, int32_t stepSize)
{
    for (uint8_t i = 0; i < model->_private.numLayers; i++)
    {
        VanillaLayerStep(state, &model->_private.layers[i], stepSize);
    }
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
