#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>
#include "Exp_LUT.h"

#define CLAMP(x, lo, hi)   ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#define MAX(a, b)          ((a) > (b) ? (a) : (b))
#define MIN(a, b)          ((a) < (b) ? (a) : (b))
#define LENGTH(arr)      (sizeof(arr) / sizeof(arr[0]))

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

    // Q16.16 * Q16.16 → Q32.32 → back to Q16.16
    return (int32_t)(acc >> 16);
}

#endif
