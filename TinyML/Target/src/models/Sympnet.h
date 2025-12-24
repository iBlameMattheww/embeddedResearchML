#ifndef SYMPNET_H
#define SYMPNET_H

#include <stdint.h>
#include "utils.h"

#define WEIGHT_FRACTIONAL_BITS  7
#define COEFFICIENT_FRACTIONAL_BITS  15
#define STATE_FRACTIONAL_BITS  16

typedef struct
{
    int32_t p;
    int32_t q;
} phaseState_t;

typedef struct 
{
    struct
    {
        const int16_t* coefficients;
        const int8_t* weights;
        uint8_t numCoefficients;
    } _private; 
} layer_t;

typedef struct 
{
    struct
    {
        layer_t *layers;
        uint8_t numLayers;
    } _private;
} model_t;

int32_t PolynomialDerivation(const int16_t* coefficients, uint8_t numberOfCoefficients, int32_t m);
int32_t SymplecticTimeScale(int32_t h, int32_t dH);
void SympnetStateUpdate(phaseState_t *state, int32_t scale, const int8_t *weights);
void SympnetLayerStep(phaseState_t *state, const layer_t *layer, int32_t stepSize);
void Symplectic_init(model_t *model, uint8_t numLayers, int32_t stepSize);

#endif // SYMPNET_H