#include "PINN_Main.h"
#include <stdbool.h>
#include "tusb.h"
#include "bsp/board.h"
#include "pico/stdlib.h"

static serial_t serial;
static PINN_Model_t pinnModel;

int main()
{
    board_init();
    tusb_init();
    Heartbeat_init();

    Serial_Init(&serial);
    PINN_Init(&pinnModel, PINN_ModelNumLayers);
    PINN_Inference_Init(&serial, &pinnModel);

    while (true)
    {
        tud_task();              
        SerialTask(&serial);     
        PINN_Inference_Task();
        tight_loop_contents();
    }
}