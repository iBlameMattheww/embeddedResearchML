#include "SNN_Main.h"
#include <stdio.h>
#include <stdbool.h>
#include "tusb.h"
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"

static serial_t serial;
static symplecticModel_t sympModel;

int main() 
{
    stdio_init_all();
    Heartbeat_init();
    Serial_Init(&serial);
    Symplectic_Init(&sympModel, SymplecticModelNumLayers, StepSize);
    SymplecticInference_Init(&serial, &sympModel);

    while (true) 
    {
        tud_task();
        SerialTask(&serial);
        SymplecticInference_Task();
        tight_loop_contents();
    }
    return 0;
}

