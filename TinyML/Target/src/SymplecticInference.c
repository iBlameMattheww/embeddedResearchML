#include "SymplecticInference.h"
#include <tusb.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static struct 
{
    serial_t *serial;
    symplecticModel_t *model;
    inferenceState_t state;
    phaseState_t phase;
    int32_t phaseBufferP[MAX_STEPS];
    int32_t phaseBufferQ[MAX_STEPS];
    int32_t stepSize;
    uint32_t totalSteps;
    uint32_t bufferedSteps;
    uint32_t TX_Index;
    bool runAccepted;
} symplecticInferenceContext;

static void SymplecticInference_Reset(void)
{
    symplecticInferenceContext.state = InferenceIdle;
    symplecticInferenceContext.runAccepted = false;
    symplecticInferenceContext.bufferedSteps = 0;
    symplecticInferenceContext.TX_Index = 0;
    symplecticInferenceContext.totalSteps = 0;
    memset(&symplecticInferenceContext.phase, 0, sizeof(phaseState_t));
    memset(symplecticInferenceContext.phaseBufferP, 0, sizeof(int32_t) * MAX_STEPS);
    memset(symplecticInferenceContext.phaseBufferQ, 0, sizeof(int32_t) * MAX_STEPS);
    SerialReset(symplecticInferenceContext.serial);
}

void SymplecticInference_Task(void)
{
    if (symplecticInferenceContext.serial->_private.resetRequested)
    {
        SymplecticInference_Reset();
        symplecticInferenceContext.serial->_private.resetRequested = false;
    }

    /* ---------------- IDLE: wait for RUN ---------------- */
    if (symplecticInferenceContext.state == InferenceIdle &&
        IsSerialCommandAvailable(symplecticInferenceContext.serial) &&
        !symplecticInferenceContext.runAccepted)
    {
        runPayload_t payload;
        serialCmd_t cmd = GetSerialCommand(symplecticInferenceContext.serial);

        if (cmd == Cmd_Run)
        {
            /* COPY PAYLOAD FIRST */
            SerialCopyPayload(
                symplecticInferenceContext.serial,
                &payload
            );

            if (payload.numSteps > MAX_STEPS)
            {
                /* Invalid number of steps, ignore */
                SerialSendDone(symplecticInferenceContext.serial);
                symplecticInferenceContext.state = InferenceIdle;
                symplecticInferenceContext.runAccepted = false;
                return;
            }

            /* NOW clear command */
            ClearSerialCommand(symplecticInferenceContext.serial);

            /* Zero-step guard */
            if (payload.numSteps == 0)
            {
                SerialSendDone(symplecticInferenceContext.serial);
                return;
            }

            /* Latch state */
            symplecticInferenceContext.serial->_private.sequenceNumber = 0;
            symplecticInferenceContext.serial->_private.acknowledged = false;
            symplecticInferenceContext.bufferedSteps = 0;
            symplecticInferenceContext.TX_Index = 0;
            
            symplecticInferenceContext.runAccepted = true;
            symplecticInferenceContext.state = InferenceRunning;

            symplecticInferenceContext.phase.p = payload.p0;
            symplecticInferenceContext.phase.q = payload.q0;

            symplecticInferenceContext.stepSize = payload.stepSize;

            symplecticInferenceContext.totalSteps = payload.numSteps;
        }
    }

    /* ---------------- RUNNING: step + stream ---------------- */
    if (symplecticInferenceContext.state == InferenceRunning)
    {
        if (symplecticInferenceContext.bufferedSteps < symplecticInferenceContext.totalSteps)
        {
            SympnetStep(
                symplecticInferenceContext.model,
                &symplecticInferenceContext.phase,
                symplecticInferenceContext.stepSize
            );

            symplecticInferenceContext.phaseBufferP[symplecticInferenceContext.bufferedSteps] =
                symplecticInferenceContext.phase.p;
            symplecticInferenceContext.phaseBufferQ[symplecticInferenceContext.bufferedSteps] =
                symplecticInferenceContext.phase.q;

            symplecticInferenceContext.bufferedSteps++;
        }
        else
        {
            symplecticInferenceContext.state = InferenceTransmit;
            symplecticInferenceContext.TX_Index = 0;
        }
    }

    /* ---------------- TRANSMIT: send buffered data ---------------- */
    if (symplecticInferenceContext.state == InferenceTransmit)
    {
        tud_task();
        if (symplecticInferenceContext.TX_Index < symplecticInferenceContext.bufferedSteps)
        {
            SerialSendPhasePacket(
                symplecticInferenceContext.serial,
                symplecticInferenceContext.phaseBufferP[symplecticInferenceContext.TX_Index],
                symplecticInferenceContext.phaseBufferQ[symplecticInferenceContext.TX_Index]
            );

            if (symplecticInferenceContext.serial->_private.acknowledged == true)
            {
                symplecticInferenceContext.TX_Index++;
                symplecticInferenceContext.serial->_private.sequenceNumber++;
                symplecticInferenceContext.serial->_private.acknowledged = false;           
            }  
        }
        
        else
        {
            SerialSendDone(symplecticInferenceContext.serial);
            
            if (symplecticInferenceContext.serial->_private.acknowledged == true)
            {
                /* DONE acknowledged, finish */
                tud_cdc_n_write_flush(CDC_ITF);
                symplecticInferenceContext.state = InferenceIdle;
                symplecticInferenceContext.runAccepted = false;
                symplecticInferenceContext.bufferedSteps = 0;
                symplecticInferenceContext.serial->_private.acknowledged = false;
            } 
        }
    }
}

void SymplecticInference_Init(serial_t *serial, symplecticModel_t *model)
{
    memset(&symplecticInferenceContext, 0, sizeof(symplecticInferenceContext));
    symplecticInferenceContext.serial = serial;
    symplecticInferenceContext.model = model;
    symplecticInferenceContext.state = InferenceIdle;
}