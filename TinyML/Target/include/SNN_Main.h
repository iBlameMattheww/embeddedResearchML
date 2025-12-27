#ifndef SNN_MAIN_H
#define SNN_MAIN_H

#include "Activations.h"
#include "SymplecticInference.h"
#include "Serial.h"
#include "Heartbeat.h"

enum
{
    SymplecticModelNumLayers = 2,
    StepSize = 16384,
};

int main();

#endif 