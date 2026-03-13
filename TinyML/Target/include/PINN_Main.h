#ifndef PINN_MAIN_H
#define PINN_MAIN_H

#include "PINN_Inference.h"
#include "Serial.h"
#include "Heartbeat.h"

enum
{
    PINN_ModelNumLayers = 4,
    StepSize = 16384,
};

#endif // PINN_MAIN_H