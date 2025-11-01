# 🧩 TinyML Pipeline Outline (Manual Export + MCU Inference)

---

## 1️⃣ Objective

- Build a **fully custom TinyML inference runtime** written in C.
- Avoid TensorFlow Lite Micro; instead, **manually export** trained parameters (weights, biases, and quantization scales).
- Maintain full control over layer math, quantization, and optimization.
- Use a simple **sine-wave regression model** as a validation platform before scaling to Lorenz or physics-informed networks.

---

## 2️⃣ Overall Flow

1. **Model training** in Python using TensorFlow (float model).
2. **Manual export** of weights, biases, and scale factors as C arrays.
3. **Construction of an embedded inference engine** in C using these exported parameters.
4. **Optimization of math routines** (exp, clamp, ReLU, etc.) for integer arithmetic.
5. **Deployment and testing** on a microcontroller (Arduino Nano 33 BLE Sense chosen first).

---

## 3️⃣ Model-Side Preparation

- Neural network is composed of **dense (fully-connected)** layers.
- Each layer’s parameters are saved into header files containing:
  - Input and output dimensions
  - Quantization scale (input-to-output)
  - Integer weights (`int8_t`)
  - Integer or short biases (`int16_t`)
- The model structure is mirrored in the C runtime by connecting layers sequentially.

---

## 4️⃣ Quantization Concept

- The Python side simulates **integer quantization**: activations and parameters scaled from float → int8.
- Each layer has a unique **scale ratio** that connects its input and output scales.
- Inference uses integer multiplication with accumulation in `int32_t` and rescales via pre-computed ratios.
- Quantization reduces memory, improves speed, and removes floating-point dependence.

---

## 5️⃣ Inference Runtime Architecture

### File Responsibilities

| File           | Purpose                                                      |
|----------------|--------------------------------------------------------------|
| `main.c`       | Entry point; initializes inference and feeds test data.      |
| `inference.c/h`| Implements NN forward pass and defines layer structures.     |
| `activations.c/h` | Stores activation functions (ReLU, softmax, etc.).        |
| `utils.c/h`    | Contains math macros and lookup tables.                      |
| `layer headers`| Exported parameters per layer.                               |

### Execution Flow

1. Quantize the input.
2. Pass data through each dense layer (matrix multiply + bias + rescale).
3. Apply the activation function.
4. Pass output to the next layer.
5. Dequantize the final result back to float for display or logging.

---

## 6️⃣ Dense Layer Logic (Conceptual)

- Each layer performs: input vector × weight matrix + bias + rescale.
- Rescaling is performed using a **layer-specific scale ratio**.
- Outputs are clamped within `[-128, 127]` to prevent overflow.
- Intermediate accumulators use higher precision (`int32_t`).
- Activation is applied element-wise after computation.

---

## 7️⃣ Activation Functions

- **ReLU:**
  - Zeroes out negative values.
  - Implemented as an inline function using integer math and clamping.
- **Softmax:**
  - Converts logits into probabilities.
  - Fully integer-only implementation: all operations (max, exp, sum, normalization) are performed using fixed-point math (Q8, scaled by 256).
  - Uses an exponential lookup table (LUT) with linear interpolation for efficiency and accuracy.
  - In-place operation: logits array is overwritten with softmax probabilities (range 0–255, sum ≈ 255).
- **Others (future):**
  - Tanh, Sigmoid, or custom activations using the same table-driven approach.

---

## 8️⃣ Utility Layer (`utils.h`)

- Central hub for math helpers used across all modules.
- Contains:
  - Clamp, max, min macros.
  - Integer saturation helpers.
  - Exponential approximation with lookup table + interpolation (Q8 fixed-point).
  - All helpers are designed for integer-only operation (no float required).

---

## 9️⃣ Exponential Lookup Table (exp_lut)

### Purpose

- Replace slow runtime `exp()` with a **precomputed lookup table** stored in Flash.
- Used for **softmax** and other activation functions requiring exponentials.

### Generation

- Generated **offline in Python** (inside the `/tools` directory).
- Script computes `exp(x)` between **−8 and 8** with a fixed step (e.g., 0.25).
- Results are normalized to integer range (Q8: 0–255 or higher for `uint32_t`) and written to `exp_lut.h`.
- Saved in `/src` and included in `utils.h`.

### Structure

- Constants:
  - `EXP_MIN`: lower bound of sampled range.
  - `EXP_STEP`: spacing between adjacent samples.
  - `EXP_SIZE`: total number of table entries.
  - `EXP_MIN_Q8`, `EXP_MAX_Q8`, `EXP_STEP_Q8`: Q8 fixed-point versions for integer math.
- Table values:
  - Start near 0 for negative inputs (e.g., e⁻⁸ ≈ 0).
  - Rise exponentially toward max for large positives.

### Runtime Behavior

1. Input `x` is **clamped** to table range.
2. Compute fractional index between nearest samples (Q8 math).
3. **Linearly interpolate** between two LUT values for smooth output.
4. Return approximated exponential in Q8 fixed-point.

### Design Details

- The LUT is **model-independent** (does not change between retrains).
- Regenerate only when:
  - The range changes (e.g., from −8→8 to −10→10).
  - The step size changes (affects accuracy).
  - Output format changes (e.g., move to `uint16_t` or different Q format).
- Zeros for small `x` values are expected — they represent near-zero exponentials.
- Interpolation gives smooth transitions with just **two table reads** and minimal arithmetic.

---

## 🔧 10️⃣ Design Philosophy

- **Portability:**
  Written purely in C with no TensorFlow Lite dependencies.
- **Transparency:**
  Every mathematical step is explicit and modifiable.
- **Modularity:**
  Model weights live independently from the inference logic.
- **Scalability:**
  Supports evolution into physics-informed, symplectic, or hybrid neural architectures.
- **Efficiency:**
  Heavy nonlinearities are replaced with lookup tables or integer math for real-time performance.
- **Integer-only inference:**
  All math (including softmax) is performed using fixed-point integer arithmetic—no floating-point required on MCU.

---

## 11️⃣ Current Status

- ✅ Model trained and parameters exported.
- ✅ Manual quantization tested and working.
- ✅ Inference runtime modularized and functional.
- ✅ ReLU activation implemented (integer-only).
- ✅ Exponential lookup table + interpolation integrated (Q8 fixed-point).
- ✅ Utility macros established for math operations.
- ✅ Integer-only softmax implemented and tested (in-place, no float).
- 🚧 Next: deploy on MCU and measure performance + precision.

---

## 12️⃣ Next Development Milestones

1. Implement integer-only (Q7/Q15) exponential interpolation for other activations.
2. Add tanh and sigmoid LUTs using the same approach.
3. Validate on Arduino Nano 33 BLE Sense, ESP32, and RPi Pico with test vectors.
4. Extend runtime for the Lorenz system model.
5. Benchmark latency, memory use, and MSE.
6. Package workflow into a reproducible embedded-ML template.

---