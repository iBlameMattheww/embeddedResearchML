#ifndef PINN_INFERENCE_H
#define PINN_INFERENCE_H
#include <stdint.h>
#include "Serial.h"
#include "PINN.h"

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

void PINN_Inference_Task();
void PINN_Inference_Init(serial_t *serial, PINN_Model_t *model);

#endif // PINN_INFERENCE_H