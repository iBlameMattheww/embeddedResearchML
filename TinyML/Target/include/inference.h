#ifndef INFERENCE_H
#define INFERENCE_H
#include <stdint.h>


typedef struct
{
    struct{    
        int16_t *inputBuffer;
        int16_t *outputBuffer;
    } _private;
} inference_t;

void Inference_Init(inference_t* instance);

#endif