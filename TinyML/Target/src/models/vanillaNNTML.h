#ifndef VANILLANNTML_H
#define VANILLANNTML_H
#include "dense_0.h"
#include "dense_1.h"
#include "dense_2.h"
#include "activations.h"
#include <stdint.h>

typedef enum
{
    relu,
    softmax
} activation_t;

typedef struct 
{
    struct
    {
        const int8_t* weights;
        const int16_t* biases;
        uint8_t inputSize;
        uint8_t outputSize;
        activation_t activation;
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

void VanillaNNTML_init(model_t *model, int16_t *input, int16_t *output);

#endif