#ifndef VanillaInference_H
#define VanillaInference_H
#include <stdint.h>
#include "Serial.h"
#include "Activations.h"

#define MAX_STEPS 1024

typedef struct 
{
    int32_t stepSize;
    uint32_t numSteps;
    int32_t p0; 
    int32_t q0; 
} runPayload_t;

typedef struct 
{
    int32_t p;
    int32_t q;
} PhasePacket_t;

typedef enum
{
    InferenceIdle,
    InferenceRunning,
    InferenceTransmit,
} inferenceState_t;

void VanillaInference_Task();
void VanillaInference_Init(serial_t *serial, vanillaModel_t *model);

#endif