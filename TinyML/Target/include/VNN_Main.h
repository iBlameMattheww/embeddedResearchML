#ifndef VNN_MAIN_H
#define VNN_MAIN_H

#include "VanillaInference.h"
#include "Serial.h"
#include "Heartbeat.h"

enum
{
    VanillaModelNumLayers = 2,
    StepSize = 16384,
};

int main();

#endif