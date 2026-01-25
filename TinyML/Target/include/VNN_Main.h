#ifndef VNN_MAIN_H
#define VNN_MAIN_H

// #include VanillaInference.h --- IGNORE ---
#include "Serial.h"
#include "Heartbeat.h"

enum
{
    VanillaModelNumLayers = 3,
    StepSize = 16384,
};

int main();

#endif