#ifndef INFERENCE_H
#define INFERENCE_H
#include <stdint.h>



typedef struct
{
    int16_t *inputBuffer;
    int16_t *outputBuffer;
    model_t *model;
} inference_t;


void inference_Init(inference_t* instance);

#endif