#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>
#include "exp_lut.h"

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

#endif
