#ifndef INFERENCE_H
#define INFERENCE_H
#include <stdint.h>
#include "vanillaNNTML.h"


typedef struct
{
    struct{    
        int16_t *inputBuffer;
        int16_t *outputBuffer;
        model_t *model;
    } _private;
} inference_t;

void Inference_Init(inference_t* instance);

#endif