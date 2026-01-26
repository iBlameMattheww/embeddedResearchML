#include "VNN_Main.h"
#include <stdbool.h>
#include "tusb.h"
#include "bsp/board.h"
#include "pico/stdlib.h"

static serial_t serial;
static vanillaModel_t vanillaModel;

int main()
{
    board_init();
    tusb_init();
    Heartbeat_init();

    Serial_Init(&serial);
    Vanilla_Init(&vanillaModel, VanillaModelNumLayers);
    VanillaInference_Init(&serial, &vanillaModel);

    while (true)
    {
        tud_task();              
        SerialTask(&serial);     
        VanillaInference_Task();
        tight_loop_contents();
    }
}