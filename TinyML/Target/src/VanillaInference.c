#include "VanillaInference.h"
#include <tusb.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static struct 
{
    serial_t *serial;
    vanillaModel_t *model;
    inferenceState_t state;
    phaseState_t phase;
    int32_t phaseBufferP[MAX_STEPS];
    int32_t phaseBufferQ[MAX_STEPS];
    int32_t stepSize;
    uint32_t totalSteps;
    uint32_t bufferedSteps;
    uint32_t TX_Index;
    bool runAccepted;
} vanillaInferenceContext;

static void VanillaInference_Reset(void)
{
    vanillaInferenceContext.state = InferenceIdle;
    vanillaInferenceContext.runAccepted = false;
    vanillaInferenceContext.bufferedSteps = 0;
    vanillaInferenceContext.TX_Index = 0;
    vanillaInferenceContext.totalSteps = 0;
    memset(&vanillaInferenceContext.phase, 0, sizeof(phaseState_t));
    memset(vanillaInferenceContext.phaseBufferP, 0, sizeof(int32_t) * MAX_STEPS);
    memset(vanillaInferenceContext.phaseBufferQ, 0, sizeof(int32_t) * MAX_STEPS);
    SerialReset(vanillaInferenceContext.serial);
}

void VanillaInference_Task(void)
{
    if (vanillaInferenceContext.serial->_private.resetRequested)
    {
        VanillaInference_Reset();
        vanillaInferenceContext.serial->_private.resetRequested = false;
    }

    /* ---------------- IDLE: wait for RUN ---------------- */
    if (vanillaInferenceContext.state == InferenceIdle &&
        IsSerialCommandAvailable(vanillaInferenceContext.serial) &&
        !vanillaInferenceContext.runAccepted)
    {
        runPayload_t payload;
        serialCmd_t cmd = GetSerialCommand(vanillaInferenceContext.serial);
        if (cmd == Cmd_Run)
        {
            /* COPY PAYLOAD FIRST */
            SerialCopyPayload(
                vanillaInferenceContext.serial,
                &payload
            );

            if (payload.numSteps > MAX_STEPS)
            {
                /* Invalid number of steps, ignore */
                SerialSendDone(vanillaInferenceContext.serial);
                ClearSerialCommand(vanillaInferenceContext.serial);
                vanillaInferenceContext.state = InferenceIdle;
                vanillaInferenceContext.runAccepted = false;
                return;
            }

            /* NOW clear command */
            ClearSerialCommand(vanillaInferenceContext.serial);

            /* Zero-step guard */
            if (payload.numSteps == 0)
            {
                SerialSendDone(vanillaInferenceContext.serial);
                ClearSerialCommand(vanillaInferenceContext.serial);
                vanillaInferenceContext.state = InferenceIdle;
                vanillaInferenceContext.runAccepted = false;
                return;
            }

            /* Latch state */
            vanillaInferenceContext.serial->_private.sequenceNumber = 0;
            vanillaInferenceContext.serial->_private.acknowledged = false;
            vanillaInferenceContext.bufferedSteps = 0;
            vanillaInferenceContext.TX_Index = 0;
            
            vanillaInferenceContext.runAccepted = true;
            vanillaInferenceContext.state = InferenceRunning;

            vanillaInferenceContext.phase.p = payload.p0;
            vanillaInferenceContext.phase.q = payload.q0;

            vanillaInferenceContext.stepSize = payload.stepSize;

            vanillaInferenceContext.totalSteps = payload.numSteps;
        }
    }

    /* ---------------- RUNNING: step + stream ---------------- */
    if (vanillaInferenceContext.state == InferenceRunning)
    {
        if (vanillaInferenceContext.bufferedSteps < vanillaInferenceContext.totalSteps)
        {
            VanillaStep(
                vanillaInferenceContext.model,
                &vanillaInferenceContext.phase,
                vanillaInferenceContext.stepSize
            );

            vanillaInferenceContext.phaseBufferP[vanillaInferenceContext.bufferedSteps] =
                vanillaInferenceContext.phase.p;
            vanillaInferenceContext.phaseBufferQ[vanillaInferenceContext.bufferedSteps] =
                vanillaInferenceContext.phase.q;

            vanillaInferenceContext.bufferedSteps++;
        }
        else
        {
            vanillaInferenceContext.state = InferenceTransmit;
            vanillaInferenceContext.TX_Index = 0;
        }
    }

    /* ---------------- TRANSMIT: send buffered data ---------------- */
    if (vanillaInferenceContext.state == InferenceTransmit)
    {
        tud_task();
        if (vanillaInferenceContext.TX_Index < vanillaInferenceContext.bufferedSteps)
        {
            SerialSendPhasePacket(
                vanillaInferenceContext.serial,
                vanillaInferenceContext.phaseBufferP[vanillaInferenceContext.TX_Index],
                vanillaInferenceContext.phaseBufferQ[vanillaInferenceContext.TX_Index]
            );

            if (vanillaInferenceContext.serial->_private.acknowledged == true)
            {
                vanillaInferenceContext.TX_Index++;
                vanillaInferenceContext.serial->_private.sequenceNumber++;
                vanillaInferenceContext.serial->_private.acknowledged = false;           
            }  
        }
        
        else
        {
            SerialSendDone(vanillaInferenceContext.serial);
            
            if (vanillaInferenceContext.serial->_private.acknowledged == true)
            {
                /* DONE acknowledged, finish */
                tud_cdc_n_write_flush(CDC_ITF);
                vanillaInferenceContext.state = InferenceIdle;
                vanillaInferenceContext.runAccepted = false;
                vanillaInferenceContext.bufferedSteps = 0;
                vanillaInferenceContext.serial->_private.acknowledged = false;
            } 
        }
    }
}

void VanillaInference_Init(serial_t *serial, vanillaModel_t *model)
{
    memset(&vanillaInferenceContext, 0, sizeof(vanillaInferenceContext));
    vanillaInferenceContext.serial = serial;
    vanillaInferenceContext.model = model;
    vanillaInferenceContext.state = InferenceIdle;
}