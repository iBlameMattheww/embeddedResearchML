#include "vanillaNNTML.h"
#define MAX_LAYER_SIZE 32

static void VanillaNNTML_infer(model_t *model, int16_t *input, int16_t *output)
{
    int16_t bufferA[MAX_LAYER_SIZE];
    int16_t bufferB[MAX_LAYER_SIZE];
    int16_t *src = input;
    int16_t *dst = bufferA;

    for (uint8_t l = 0; l < model->_private.numLayers; l++)
    {
        layer_t *layer = &model->_private.layers[l];
        for (uint8_t o = 0; o < layer->_private.outputSize; o++)
        {
            int32_t acc = 0;
            for (uint8_t i = 0; i < layer->_private.inputSize; i++)
            {
                acc += layer->_private.weights[o * layer->_private.inputSize + i] * src[i];
            }
            acc += layer->_private.biases[o];
            dst[o] = (int16_t)acc;
        }
        switch (layer->_private.activation)
        {
            case relu:
                for (uint8_t o = 0; o < layer->_private.outputSize; o++)
                {
                    dst[o] = Relu(dst[o]);
                }
                break;
            case softmax:
                Softmax(dst, layer->_private.outputSize);
                break;
        }
        int16_t *temp = src;
        src = dst;
        dst = temp;
        
    }
    for (uint8_t o = 0; 
        o < model->_private.layers[model->_private.numLayers - 1]._private.outputSize;
        o++) 
    {
        output[o] = src[o];
    }
}

void VanillaNNTML_init(model_t *model, int16_t *input, int16_t *output)
{
    static layer_t layers[3];

    // Layer 0
    layers[0]._private.weights = (const int8_t*)W0;
    layers[0]._private.biases = (const int16_t*)B0;
    layers[0]._private.inputSize = L0_IN;
    layers[0]._private.outputSize = L0_OUT;
    layers[0]._private.activation = relu;

    // Layer 1
    layers[1]._private.weights = (const int8_t*)W1;
    layers[1]._private.biases = (const int16_t*)B1;
    layers[1]._private.inputSize = L1_IN;
    layers[1]._private.outputSize = L1_OUT;
    layers[1]._private.activation = relu;

    // Layer 2
    layers[2]._private.weights = (const int8_t*)W2;
    layers[2]._private.biases = (const int16_t*)B2;
    layers[2]._private.inputSize = L2_IN;
    layers[2]._private.outputSize = L2_OUT;
    layers[2]._private.activation = softmax;

    model->_private.layers = layers;
    model->_private.numLayers = 3;

    VanillaNNTML_infer(model, input, output); // Warm-up call (if needed)
}
