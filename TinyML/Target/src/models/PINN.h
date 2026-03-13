#ifndef PINN_H
#define PINN_H

#include <stdint.h>
#include <math.h>
#include "PINN_Model_Layers.h"
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
} PINN_Layer_t;

typedef struct 
{
    struct
    {
        PINN_Layer_t *layers;
        uint8_t numLayers;
    } _private;
} PINN_Model_t;

void PINN_Step(PINN_Model_t *model, phaseState_t *state, int32_t stepSize);
void PINN_Init(PINN_Model_t *model, uint8_t numLayers);

#endif // PINN_H