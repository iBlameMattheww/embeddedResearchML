#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <stdint.h>

uint16_t Relu(int16_t x);
void Softmax(int16_t x [], uint8_t length);

#endif