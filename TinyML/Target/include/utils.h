#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <math.h>

#define CLAMP(x, lo, hi)   ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#define MAX(a, b)          ((a) > (b) ? (a) : (b))
#define MIN(a, b)          ((a) < (b) ? (a) : (b))

static inline float exp_approx(float x) {
    // 3rd-order Taylor around 0
    if (x < -8.0f) return 0.0f;
    if (x >  8.0f) return expf(8.0f);   // or pre-clamped constant
    return 1.0f + x + 0.5f*x*x + 0.1667f*x*x*x;
}

#endif
