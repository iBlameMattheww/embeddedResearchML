#include "SNN_Main.h"
#include <stdbool.h>
#include "tusb.h"
#include "bsp/board.h"
#include "pico/stdlib.h"

static serial_t serial;
static symplecticModel_t sympModel;

int main()
{
    board_init();
    tusb_init();
    Heartbeat_init();

    Serial_Init(&serial);
    Symplectic_Init(&sympModel, SymplecticModelNumLayers);
    SymplecticInference_Init(&serial, &sympModel);

    while (true)
    {
        tud_task();              
        SerialTask(&serial);     
        SymplecticInference_Task();
        tight_loop_contents();
    }
}