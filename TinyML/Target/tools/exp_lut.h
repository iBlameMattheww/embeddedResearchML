// Exponential Lookup Table
#include <stdint.h>

#define EXP_MIN -8f
#define EXP_STEP 0.25f
#define EXP_SIZE 65
static const uint8_t exp_lut[EXP_SIZE] = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 36, 47, 60, 77, 99, 127};
