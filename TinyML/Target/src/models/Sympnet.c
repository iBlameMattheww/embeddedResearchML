#include "Sympnet.h"
#include "Utils.h"

int32_t PolynomialDerivation(
    const int32_t* a,   // Q16.16 coefficients
    uint8_t n,
    int32_t m           // Q16.16
)
{
    int64_t acc = 0;          // Q32.32 accumulator
    int64_t m_power = m;      // m^1 in Q16.16

    for (uint8_t k = 0; k < n; k++)
    {
        // term = (2+k) * a[k] * m^(k+1)
        int64_t term = (int64_t)(2 + k) * a[k]; // Q16.16
        term = (term * m_power) >> 16;          // back to Q16.16

        acc += term;
        CLAMP(acc, INT32_MIN, INT32_MAX);

        // m_power *= m  → m^(k+2)
        m_power = (m_power * m) >> 16;
        CLAMP(m_power, INT32_MIN, INT32_MAX);
    }

    return (int32_t)acc;  // Q16.16
}

int32_t SymplecticTimeScale(int32_t h, int32_t dH)
{
    // h: Q16.16, dH: Q16.16 → product: Q32.32
    return (int32_t)(((int64_t)h * dH) >> 16);
}

void SympnetStateUpdate(
    phaseState_t *state,
    int32_t scale,           // Q16.16 = h * dH
    const int32_t *weights   // Q16.16
)
{
    state->p += (int32_t)(((int64_t)scale * weights[1]) >> 16);
    CLAMP(state->p, INT32_MIN, INT32_MAX);
    state->q -= (int32_t)(((int64_t)scale * weights[0]) >> 16);
    CLAMP(state->q, INT32_MIN, INT32_MAX);
}


void SympnetLayerStep(phaseState_t *state,
                      const symplecticLayer_t *layer,
                      int32_t stepSize)
{
    int32_t x[2] = { state->p, state->q };  // Q16.16

    int32_t m = Dot_Q16_Q16_TO_Q16(
        layer->_private.weights, x, 2
    );  // Q16.16

    int32_t dH = PolynomialDerivation(
        layer->_private.coefficients,
        layer->_private.numCoefficients,
        m
    );  // Q16.16

    int32_t scale = SymplecticTimeScale(stepSize, dH); // Q16.16

    SympnetStateUpdate(state, scale, layer->_private.weights);
}

void SympnetStep(symplecticModel_t *model, phaseState_t *state, int32_t stepSize)
{
    for (uint8_t layerIndex = 0; layerIndex < model->_private.numLayers; layerIndex++)
    {
        SympnetLayerStep(state, &model->_private.layers[layerIndex], stepSize);
    }
}

void Symplectic_Init(symplecticModel_t *model, uint8_t numLayers)
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