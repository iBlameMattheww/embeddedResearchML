#ifndef SYMPLECTICINFERENCE_H
#define SYMPLECTICINFERENCE_H
#include <stdint.h>
#include "Serial.h"
#include "Sympnet.h"


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

typedef enum
{
    InferenceIdle,
    InferenceRunning,
} inferenceState_t;

void SymplecticInference_Task();
void SymplecticInference_Init(serial_t *serial, symplecticModel_t *model);

#endif