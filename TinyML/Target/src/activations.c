#include "activations.h"
#include "utils.h"

uint8_t Relu(int8_t x)
{
    return CLAMP(x, 0, INT8_MAX);
}

void Softmax(int8_t *x, uint8_t length)
{
    int8_t maxVal = x[0];
    int32_t sum = 0;
    for (uint8_t i = 0; i < length; i++)
    {
        maxVal = MAX(maxVal, x[i]);
    }
    for (uint8_t i = 0; i < length; i++)
    {
        x[i] = (x[i] - maxVal) * 256;
        x[i] = Exp_Approx(x[i]);
        sum += x[i];
    }
    for (uint8_t i = 0; i < length; i++)
    {
        x[i] = (x[i] * 255) / sum;
    }
}