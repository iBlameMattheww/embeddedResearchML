#include "Sympnet.h"

int32_t PolynomialDerivation(const int16_t* coefficients, uint8_t numberOfCoefficients, int32_t m)
{
    int64_t accumulator = 0;
    int32_t m_power = m;

    for (uint8_t k = 0; k < numberOfCoefficients; k++)
    {
        uint8_t degree = 2 + k;
        int32_t term = (int32_t)coefficients[k] * m_power;

        term >>= COEFFICIENT_FRACTIONAL_BITS;
        term *= degree;
        accumulator += term;
        m_power = (int32_t)(((int64_t)m_power * (int64_t)m) >> STATE_FRACTIONAL_BITS);
    }
    return (int32_t)accumulator;
}

int32_t SymplecticTimeScale(int32_t h, int32_t dH)
{
    int64_t product = (int64_t)h * (int64_t)dH;
    return (int32_t)(product >> STATE_FRACTIONAL_BITS);
}

void SympnetStateUpdate(phaseState_t *state, int32_t scale, const int8_t *weights)
{
    state->p += (int32_t)(((int64_t)scale * (int64_t)weights[1]) >> WEIGHT_FRACTIONAL_BITS);
    state->q -= (int32_t)(((int64_t)scale * (int64_t)weights[0]) >> WEIGHT_FRACTIONAL_BITS);
}

void SympnetLayerStep(phaseState_t *state, const symplecticLayer_t *layer, int32_t stepSize)
{
    int32_t x[2] = {state->p, state->q};
    int32_t m = Dot_I8_I32_TO_I32(layer->_private.weights, x, 2, WEIGHT_FRACTIONAL_BITS);
    int32_t dH = PolynomialDerivation(layer->_private.coefficients, layer->_private.numCoefficients, m);
    int32_t scale = SymplecticTimeScale(stepSize, dH);
    SympnetStateUpdate(state, scale, layer->_private.weights);
}

void SympnetStep(symplecticModel_t *model, phaseState_t *state, int32_t stepSize)
{
    for (uint8_t layerIndex = 0; layerIndex < model->_private.numLayers; layerIndex++)
    {
        SympnetLayerStep(state, &model->_private.layers[layerIndex], stepSize);
    }
}

void Symplectic_Init(symplecticModel_t *model, uint8_t numLayers, int32_t stepSize)
{
    static symplecticLayer_t layers[] = {
        {._private = {
            .coefficients = Sympnet_layer_0_a,
            .weights = Sympnet_layer_0_w,
            .numCoefficients = COEFFICIENTS_LEN
        }},
        {._private = {
            .coefficients = Sympnet_layer_1_a,
            .weights = Sympnet_layer_1_w,
            .numCoefficients = COEFFICIENTS_LEN
        }}
    };
    model->_private.layers = layers;
    model->_private.numLayers = numLayers;
}