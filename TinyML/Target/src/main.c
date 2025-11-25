#include <stdio.h>
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"

int main() {
    stdio_init_all();

    const uint LED = 25;
    gpio_init(LED);
    gpio_set_dir(LED, GPIO_OUT);

    while (true) {
        printf("HELLO USB CDC\n");

        gpio_put(LED, 1);
        sleep_ms(250);
        gpio_put(LED, 0);
        sleep_ms(250);
    }
    return 0;
}
