#ifndef SYMPNET_H
#define SYMPNET_H

#include <stdint.h>
#include "SNN_Model_layers.h"
#include "Utils.h"

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
} symplecticLayer_t;

typedef struct 
{
    struct
    {
        symplecticLayer_t *layers;
        uint8_t numLayers;
    } _private;
} symplecticModel_t;

int32_t PolynomialDerivation(const int16_t* coefficients, uint8_t numberOfCoefficients, int32_t m);
int32_t SymplecticTimeScale(int32_t h, int32_t dH);
void SympnetStateUpdate(phaseState_t *state, int32_t scale, const int8_t *weights);
void SympnetLayerStep(phaseState_t *state, const symplecticLayer_t *layer, int32_t stepSize);
void SympnetStep(symplecticModel_t *model, phaseState_t *state, int32_t stepSize);
void Symplectic_Init(symplecticModel_t *model, uint8_t numLayers, int32_t stepSize);

#endif // SYMPNET_H