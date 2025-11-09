#include "inference.h"

void Inference_Init(inference_t* instance)
{
    static model_t model;
    static int16_t inputBuffer[L0_IN];
    static int16_t outputBuffer[L2_OUT];

    instance->_private.inputBuffer = inputBuffer;
    instance->_private.outputBuffer = outputBuffer;
    instance->_private.model = &model;

    VanillaNNTML_init(&model, inputBuffer, outputBuffer);
}