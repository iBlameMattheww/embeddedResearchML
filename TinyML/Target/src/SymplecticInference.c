#include "SymplecticInference.h"
#include <string.h>

static struct 
{
    serial_t *serial;
    symplecticModel_t *model;
    inferenceState_t state;
    phaseState_t phase;
    int32_t stepSize;
    uint32_t remainingSteps;
} symplecticInferenceContext;

void SymplecticInference_Task()
{
    if (symplecticInferenceContext.state == InferenceIdle 
    && IsSerialCommandAvailable(symplecticInferenceContext.serial))
    {
        runPayload_t payload;
        serialCmd_t cmd = GetSerialCommand(symplecticInferenceContext.serial);

        if (cmd == Cmd_Run)
        {
            SerialCopyPayload(
                symplecticInferenceContext.serial, 
                &payload);

            symplecticInferenceContext.phase.p = (int32_t)payload.p0 << 16;
            symplecticInferenceContext.phase.q = (int32_t)payload.q0 << 16;
            symplecticInferenceContext.stepSize = payload.stepSize;
            symplecticInferenceContext.remainingSteps = payload.numSteps;
            symplecticInferenceContext.state = InferenceRunning;
        }
    }
    if (symplecticInferenceContext.state == InferenceRunning)
    {
        SympnetStep(
            symplecticInferenceContext.model, 
            &symplecticInferenceContext.phase, 
            symplecticInferenceContext.stepSize);
        
        SerialSendPhasePacket(
            symplecticInferenceContext.phase.p, 
            symplecticInferenceContext.phase.q);

        if (--symplecticInferenceContext.remainingSteps == 0)
        {
            SerialSendDone();
            symplecticInferenceContext.state = InferenceIdle;
        }
    }
}


void SymplecticInference_Init(serial_t *serial, symplecticModel_t *model)
{
    symplecticInferenceContext.serial = serial;
    symplecticInferenceContext.model = model;
    symplecticInferenceContext.state = InferenceIdle;
    symplecticInferenceContext.phase.p = 0;
    symplecticInferenceContext.phase.q = 0;
    symplecticInferenceContext.stepSize = 0;
    symplecticInferenceContext.remainingSteps = 0;
}