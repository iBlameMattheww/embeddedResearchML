#include "activations.h"
#include "utils.h"

uint8_t relu(int8_t x)
{
    return CLAMP(x, 0, INT8_MAX);
}

int8_t softmax(int8_t x [])
{
    uint8_t length = sizeof(x) / sizeof(x[0]);
    int8_t maxVal = max(x, length);
    for (uint8_t i = 0; i < length; i++)
    {
        maxVal = MAX(maxVal, x[i]);
    }
    for (uint8_t i = 0; i < length; i++)
    {
        x[i] = x[i] - maxVal;
    }
}