#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>
#include "exp_lut.h"

#define CLAMP(x, lo, hi)   ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#define MAX(a, b)          ((a) > (b) ? (a) : (b))
#define MIN(a, b)          ((a) < (b) ? (a) : (b))
#define LENGTH(arr)      (sizeof(arr) / sizeof(arr[0]))

static inline int32_t Exp_Approx(float x_q8) {
    if (x_q8 <= EXP_MIN_Q8) return exp_lut[0];
    if (x_q8 >= EXP_MAX_Q8) return exp_lut[EXP_SIZE - 1];

    // Integer index = (x - min)/step
    int32_t index_q8 = (x_q8 - EXP_MIN_Q8) / EXP_STEP_Q8;
    int i = index_q8; // integer part
    int32_t frac_q8 = (x_q8 - (EXP_MIN_Q8 + i * EXP_STEP_Q8)) * 256 / EXP_STEP_Q8; // 0–256 range

    uint32_t y0 = exp_lut[i];
    uint32_t y1 = exp_lut[i+1];

    // Linear interpolation: y = y0 + (y1 - y0) * frac/256
    return y0 + ((int64_t)(y1 - y0) * frac_q8 >> 8);
}

#endif
