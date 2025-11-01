#ifndef INFERENCE_H
#define INFERENCE_H
#include <stdint.h>

typedef struct 
{
    struct
    {
        const int8_t* weights;
        const int16_t* biases;
        int inputSize;
        int outputSize;
    } _private; 
} inference_t;

void inference_Init(inference_t* instance);

#endif