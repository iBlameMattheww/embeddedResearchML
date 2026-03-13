#include "PINN.h"

static void MatrixVectorMultiply(
    const int32_t *input,
    int32_t *output,
    const int32_t *weights,
    const int32_t *biases,
    uint8_t inputSize,
    uint8_t outputSize
)
{
    for (uint8_t i = 0; i < outputSize; i++)
    {
        int64_t acc = biases[i];

        for (uint8_t j = 0; j < inputSize; j++)
        {
            acc += ((int64_t)weights[i * inputSize + j] * input[j]) >> 16;
        }

        output[i] = CLAMP((int32_t)acc, INT32_MIN, INT32_MAX);
    }
}

static void ApplyTanh(int32_t *data, uint8_t length)
{
    for (uint8_t i = 0; i < length; i++)
    {
        data[i] = Tanh_Approx(data[i]);
    }
}

static void PINN_Forward(
    const phaseState_t *state,
    int32_t *dx
)
{
    int32_t x0[2] = { state->p, state->q };  // Q16.16
    int32_t h1[FC1_OUT_DIM];
    int32_t h2[FC2_OUT_DIM];
    int32_t h3[FC3_OUT_DIM];
    int32_t out[FC4_OUT_DIM];

    MatrixVectorMultiply(
        x0,
        h1,
        (const int32_t *)fc1_weights,
        fc1_biases,
        FC1_IN_DIM,
        FC1_OUT_DIM
    );

    ApplyTanh(h1, FC1_OUT_DIM);

    MatrixVectorMultiply(
        h1,
        h2,
        (const int32_t *)fc2_weights,
        fc2_biases,
        FC2_IN_DIM,
        FC2_OUT_DIM
    );

    ApplyTanh(h2, FC2_OUT_DIM);

    MatrixVectorMultiply(
        h2,
        h3,
        (const int32_t *)fc3_weights,
        fc3_biases,
        FC3_IN_DIM,
        FC3_OUT_DIM
    );

    ApplyTanh(h3, FC3_OUT_DIM);

    MatrixVectorMultiply(
        h3,
        out,
        (const int32_t *)fc4_weights,
        fc4_biases,
        FC4_IN_DIM,
        FC4_OUT_DIM
    );

    dx[0] = out[0];
    dx[1] = out[1];
}

static void RungaKutta4_Update(
    phaseState_t *state,
    int32_t stepSize
)
{
    int32_t k1[2], k2[2], k3[2], k4[2];
    int32_t halfStep = stepSize >> 1;

    phaseState_t tempState;

    // k1 = f(x)
    PINN_Forward(state, k1);

    // k2 = f(x + dt/2 * k1)
    tempState.p = state->p + ((int64_t)halfStep * k1[0] >> 16);
    tempState.q = state->q + ((int64_t)halfStep * k1[1] >> 16);
    PINN_Forward(&tempState, k2);

    // k3 = f(x + dt/2 * k2)
    tempState.p = state->p + ((int64_t)halfStep * k2[0] >> 16);
    tempState.q = state->q + ((int64_t)halfStep * k2[1] >> 16);
    PINN_Forward(&tempState, k3);

    // k4 = f(x + dt * k3)
    tempState.p = state->p + ((int64_t)stepSize * k3[0] >> 16);
    tempState.q = state->q + ((int64_t)stepSize * k3[1] >> 16);
    PINN_Forward(&tempState, k4);

    int64_t dp = (int64_t)k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0];
    int64_t dq = (int64_t)k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1];

    int32_t dt_over_6 = stepSize / 6;

    state->p += ((int64_t)dt_over_6 * dp) >> 16;
    state->q += ((int64_t)dt_over_6 * dq) >> 16;
}

void PINN_Step(PINN_Model_t *model, phaseState_t *state, int32_t stepSize)
{
    RungaKutta4_Update(state, stepSize);
}

void PINN_Init(PINN_Model_t *model, uint8_t numLayers)
{
    static PINN_Layer_t layers[] = 
    {
        {
            ._private = 
            {
                .biases = fc1_biases,
                .weights = (const int32_t*)fc1_weights,
                .inputSize = FC1_IN_DIM,
                .outputSize = FC1_OUT_DIM
            }
        },
        {
            ._private = 
            {
                .biases = fc2_biases,
                .weights = (const int32_t*)fc2_weights,
                .inputSize = FC2_IN_DIM,
                .outputSize = FC2_OUT_DIM
            }
        },
        {
            ._private = 
            {
                .biases = fc3_biases,
                .weights = (const int32_t*)fc3_weights,
                .inputSize = FC3_IN_DIM,
                .outputSize = FC3_OUT_DIM
        
            }
        },
        {
            ._private = 
            {
                .biases = fc4_biases,
                .weights = (const int32_t*)fc4_weights,
                .inputSize = FC4_IN_DIM,
                .outputSize = FC4_OUT_DIM
            }
        }
    };
    model->_private.layers = layers;
    model->_private.numLayers = numLayers;
}