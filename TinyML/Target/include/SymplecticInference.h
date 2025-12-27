#ifndef SYMPLECTICINFERENCE_H
#define SYMPLECTICINFERENCE_H
#include <stdint.h>


typedef struct 
{
    int8_t p0; 
    int8_t q0; 
    int32_t stepSize;
    uint32_t numSteps;
} runPayload_t;

typedef struct 
{
    int32_t p;
    int32_t q;
} phasePacket_t;

void SymplecticInference_Init();

#endif