#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>
#include "Exp_LUT.h"

#define CLAMP(x, lo, hi)   ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#define MAX(a, b)          ((a) > (b) ? (a) : (b))
#define MIN(a, b)          ((a) < (b) ? (a) : (b))
#define LENGTH(arr)      (sizeof(arr) / sizeof(arr[0]))

static inline uint8_t CRC_8(const uint8_t *data, uint8_t len)
{
    uint8_t crc = 0xFF;

    for (uint8_t i = 0; i < len; i++)
    {
        crc ^= data[i];
        for (uint8_t j = 0; j < 8; j++)
        {
            if (crc & 0x80)
            {
                crc = (crc << 1) ^ 0x31;
            }
            else
            {
                crc <<= 1;
            }
        }
    }
    return crc;
}

static inline int32_t Exp_Approx(int32_t x_q16)
{
    if (x_q16 <= EXP_MIN_Q16)
        return exp_lut[0];
    if (x_q16 >= EXP_MAX_Q16)
        return exp_lut[EXP_SIZE - 1];

    // FLOOR 
    int32_t idx = (x_q16 - EXP_MIN_Q16) / EXP_STEP_Q16;

    int32_t base_x = EXP_MIN_Q16 + idx * EXP_STEP_Q16;
    int32_t delta  = x_q16 - base_x;

    // frac in Q16
    int32_t frac_q16 = (delta << 16) / EXP_STEP_Q16;

    uint32_t y0 = exp_lut[idx];
    uint32_t y1 = exp_lut[idx + 1];

    int32_t dy = (int32_t)y1 - (int32_t)y0;

    // Use 64-bit multiply to be safe
    int32_t interp = ((int64_t)dy * frac_q16) >> 16;

    return y0 + interp;
}

static inline int32_t Dot_Q16_Q16_TO_Q16(
    const int32_t *w,   // Q16.16
    const int32_t *x,   // Q16.16
    uint8_t length
)
{
    int64_t acc = 0;

    for (uint8_t i = 0; i < length; i++)
    {
        acc += (int64_t)w[i] * (int64_t)x[i];
    }

    acc >>= 16;
    CLAMP(acc, INT32_MIN, INT32_MAX);
    return (int32_t)acc;
}

static inline int32_t Tanh_Approx(int32_t x_q16)
{
    int32_t sign = 1;

    if (x_q16 < 0)
    {
        sign = -1;
        x_q16 = -x_q16;
    }

    int64_t widened = (int64_t)x_q16 << 1;
    int32_t exp2x = Exp_Approx((int32_t)widened);   // e^(2x)

    int32_t num = exp2x - (1 << 16);          // e^(2x) - 1
    int32_t den = exp2x + (1 << 16);          // e^(2x) + 1

    if (den == 0)
    {
        return sign << 16; // tanh approaches 1 as x -> +inf
    }
        
    int32_t tanh_q16 = ((int64_t)num << 16) / den;

    return sign * tanh_q16;
}

#endif
