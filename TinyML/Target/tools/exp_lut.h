// Exponential Lookup Table
#include <stdint.h>

#define EXP_MIN -8
#define EXP_STEP_Q8 64
#define EXP_SIZE 65
#define EXP_MIN_Q8   (EXP_MIN * 256)
#define EXP_MAX_Q8   ((EXP_MIN + (EXP_SIZE - 1) * 0.25) * 256)

static const uint32_t exp_lut[EXP_SIZE] = {
0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21, 27, 35, 44, 57, 73, 94, 121, 155, 199, 256, 329, 422, 542, 696, 894, 1147, 1473, 1892, 2429, 3119, 4005, 5142, 6602, 8478, 10885, 13977, 17947, 23044, 29590, 37994, 48785, 62641, 80433, 103278, 132611, 170276, 218639, 280738, 360475, 462859, 594323, 763125};
