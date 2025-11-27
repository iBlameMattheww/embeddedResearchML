#ifndef HEARTBEAT_H
#define HEARTBEAT_H

#include "pico/stdlib.h"
#include "pico/time.h"

#define HEARTBEAT_DEFAULT_LED_PIN 25
#define HEARTBEAT_DEFAULT_INTERVAL_MS 500

typedef struct 
{
    struct
    {
        repeating_timer_t timer;
        bool ledState;
    } _private;
} heartbeat_t;


void heartbeat_init();

#endif // HEARTBEAT_H