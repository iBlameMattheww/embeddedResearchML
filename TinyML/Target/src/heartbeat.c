#include "Heartbeat.h"

static heartbeat_t led = { ._private = { .ledState = false } };

static bool ToggleLed(repeating_timer_t *rt)
{
    led._private.ledState = !led._private.ledState;
    gpio_put(HEARTBEAT_DEFAULT_LED_PIN, led._private.ledState);
    return true;
}

void Heartbeat_init() {
    const uint LED_PIN = HEARTBEAT_DEFAULT_LED_PIN;
    const uint INTERVAL_MS = HEARTBEAT_DEFAULT_INTERVAL_MS;

    gpio_init(HEARTBEAT_DEFAULT_LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    add_repeating_timer_ms(
        INTERVAL_MS,
        ToggleLed,
        NULL,
        &led._private.timer
    );
}