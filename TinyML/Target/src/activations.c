#include "activations.h"
#include "utils.h"
#include <stdio.h>

uint16_t Relu(int16_t x)
{
    return CLAMP(x, 0, INT16_MAX);
}

void Softmax(int16_t *x, uint8_t length)
{
    int16_t maxVal = x[0];
    for (uint8_t i = 0; i < length; i++)
    {
        maxVal = MAX(maxVal, x[i]);
    }
        int32_t exp_sum = 0;
        int32_t exp_vals[16];
        for (uint8_t i = 1; i < length; i++) {
            if (x[i] > maxVal) {
                maxVal = x[i];
            }
        }
        for (uint8_t i = 0; i < length; i++) {
            int32_t shifted = (x[i] - maxVal) * 256;
            if (shifted < -60000) {
                exp_vals[i] = 0;
            } else {
                shifted = CLAMP(shifted, INT16_MIN, INT16_MAX);
                exp_vals[i] = Exp_Approx(shifted); // Q16
            }
            exp_sum += exp_vals[i];
        }
    if (exp_sum == 0) 
    {
        for (uint8_t i = 0; i < length; i++)
        {
            x[i] = (i == 0) ? 255 : 0; 
        }
        return;
    }
    for (uint8_t i = 0; i < length; i++)
    {
        x[i] = (uint16_t)(((int64_t)exp_vals[i] * 255 + (exp_sum/2)) / exp_sum);
    }
}