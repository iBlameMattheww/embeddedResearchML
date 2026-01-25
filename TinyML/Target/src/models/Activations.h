#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <stdint.h>
#include "VNN_Model_Layers.h"
#include "Utils.h"

typedef struct
{
    int32_t p;
    int32_t q;
} phaseState_t;

typedef struct 
{
    struct
    {
        const int32_t* biases;
        const int32_t* weights;
        uint8_t inputSize;
        uint8_t outputSize;
    } _private; 
} vanillaLayer_t;

typedef struct 
{
    struct
    {
        vanillaLayer_t *layers;
        uint8_t numLayers;
    } _private;
} vanillaModel_t;

void VanillaStep(vanillaModel_t *model, phaseState_t *state, int32_t stepSize);
void Vanilla_Init(vanillaModel_t *model, uint8_t numLayers);

#endif