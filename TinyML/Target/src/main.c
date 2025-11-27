#include "main.h"
#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"

int main() {
    stdio_init_all();
    heartbeat_init();

    while (true) {
        printf("HELLO USB CDC\n");
        tight_loop_contents();
    }
    return 0;
}
