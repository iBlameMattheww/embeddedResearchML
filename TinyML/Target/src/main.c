#include "Main.h"
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"

int main() 
{
    stdio_init_all();
    Heartbeat_init();

    while (true) 
    {
        printf("HELLO USB CDC\n");
        tight_loop_contents();
    }
    return 0;
}

