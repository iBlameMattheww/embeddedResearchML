#include "PINN_Inference.h"
#include <tusb.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static struct 
{
    serial_t *serial;
    PINN_Model_t *model;
    inferenceState_t state;
    phaseState_t phase;
    int32_t phaseBufferP[MAX_STEPS];
    int32_t phaseBufferQ[MAX_STEPS];
    int32_t stepSize;
    uint32_t totalSteps;
    uint32_t bufferedSteps;
    uint32_t TX_Index;
    bool runAccepted;
} pinnInferenceContext;

static void PINN_Inference_Reset(void)
{
    pinnInferenceContext.state = InferenceIdle;
    pinnInferenceContext.runAccepted = false;
    pinnInferenceContext.bufferedSteps = 0;
    pinnInferenceContext.TX_Index = 0;
    pinnInferenceContext.totalSteps = 0;
    memset(&pinnInferenceContext.phase, 0, sizeof(phaseState_t));
    memset(pinnInferenceContext.phaseBufferP, 0, sizeof(int32_t) * MAX_STEPS);
    memset(pinnInferenceContext.phaseBufferQ, 0, sizeof(int32_t) * MAX_STEPS);
    SerialReset(pinnInferenceContext.serial);
}

void PINN_Inference_Task(void)
{
    if (pinnInferenceContext.serial->_private.resetRequested)
    {
        PINN_Inference_Reset();
        pinnInferenceContext.serial->_private.resetRequested = false;
    }

    /* ---------------- IDLE: wait for RUN ---------------- */
    if (pinnInferenceContext.state == InferenceIdle &&
        IsSerialCommandAvailable(pinnInferenceContext.serial) &&
        !pinnInferenceContext.runAccepted)
    {
        runPayload_t payload;
        serialCmd_t cmd = GetSerialCommand(pinnInferenceContext.serial);
        if (cmd == Cmd_Run)
        {
            /* COPY PAYLOAD FIRST */
            SerialCopyPayload(
                pinnInferenceContext.serial,
                &payload
            );

            if (payload.numSteps > MAX_STEPS)
            {
                /* Invalid number of steps, ignore */
                SerialSendDone(pinnInferenceContext.serial);
                ClearSerialCommand(pinnInferenceContext.serial);
                pinnInferenceContext.state = InferenceIdle;
                pinnInferenceContext.runAccepted = false;
                return;
            }

            /* NOW clear command */
            ClearSerialCommand(pinnInferenceContext.serial);

            /* Zero-step guard */
            if (payload.numSteps == 0)
            {
                SerialSendDone(pinnInferenceContext.serial);
                ClearSerialCommand(pinnInferenceContext.serial);
                pinnInferenceContext.state = InferenceIdle;
                pinnInferenceContext.runAccepted = false;
                return;
            }

            /* Latch state */
            pinnInferenceContext.serial->_private.sequenceNumber = 0;
            pinnInferenceContext.serial->_private.acknowledged = false;
            pinnInferenceContext.bufferedSteps = 0;
            pinnInferenceContext.TX_Index = 0;
            
            pinnInferenceContext.runAccepted = true;
            pinnInferenceContext.state = InferenceRunning;

            pinnInferenceContext.phase.p = payload.p0;
            pinnInferenceContext.phase.q = payload.q0;

            pinnInferenceContext.stepSize = payload.stepSize;

            pinnInferenceContext.totalSteps = payload.numSteps;
        }
    }

    /* ---------------- RUNNING: step + stream ---------------- */
    if (pinnInferenceContext.state == InferenceRunning)
    {
        if (pinnInferenceContext.bufferedSteps < pinnInferenceContext.totalSteps)
        {
            PINN_Step(
                pinnInferenceContext.model,
                &pinnInferenceContext.phase,
                pinnInferenceContext.stepSize
            );

            pinnInferenceContext.phaseBufferP[pinnInferenceContext.bufferedSteps] =
                pinnInferenceContext.phase.p;
            pinnInferenceContext.phaseBufferQ[pinnInferenceContext.bufferedSteps] =
                pinnInferenceContext.phase.q;

            pinnInferenceContext.bufferedSteps++;
        }
        else
        {
            pinnInferenceContext.state = InferenceTransmit;
            pinnInferenceContext.TX_Index = 0;
        }
    }

    /* ---------------- TRANSMIT: send buffered data ---------------- */
    if (pinnInferenceContext.state == InferenceTransmit)
    {
        tud_task();
        if (pinnInferenceContext.TX_Index < pinnInferenceContext.bufferedSteps)
        {
            SerialSendPhasePacket(
                pinnInferenceContext.serial,
                pinnInferenceContext.phaseBufferP[pinnInferenceContext.TX_Index],
                pinnInferenceContext.phaseBufferQ[pinnInferenceContext.TX_Index]
            );

            if (pinnInferenceContext.serial->_private.acknowledged == true)
            {
                pinnInferenceContext.TX_Index++;
                pinnInferenceContext.serial->_private.sequenceNumber++;
                pinnInferenceContext.serial->_private.acknowledged = false;           
            }  
        }
        
        else
        {
            SerialSendDone(pinnInferenceContext.serial);
            
            if (pinnInferenceContext.serial->_private.acknowledged == true)
            {
                /* DONE acknowledged, finish */
                tud_cdc_n_write_flush(CDC_ITF);
                pinnInferenceContext.state = InferenceIdle;
                pinnInferenceContext.runAccepted = false;
                pinnInferenceContext.bufferedSteps = 0;
                pinnInferenceContext.serial->_private.acknowledged = false;
            } 
        }
    }
}

void PINN_Inference_Init(serial_t *serial, PINN_Model_t *model)
{
    memset(&pinnInferenceContext, 0, sizeof(pinnInferenceContext));
    pinnInferenceContext.serial = serial;
    pinnInferenceContext.model = model;
    pinnInferenceContext.state = InferenceIdle;
}