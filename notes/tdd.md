# TinyML TDD & Unit Testing

## How to Run Tests

1. Change directory to TinyML:
   ```sh
   cd TinyML
   ```
2. Remove previous build artifacts:
   ```sh
   rm -rf build
   ```
3. Clean with Makefile:
   ```sh
   make -f tdd.mk clean
   ```
4. Run all tests:
   ```sh
   make -f tdd.mk run
   ```

---

- All test results will be shown in the terminal.
- If you see errors about missing files, check that `unity/src` and `tests/` contain source files.

---

## Test Summary

- **Test_Relu_Positive:** Checks that ReLU returns the input for positive values.
- **Test_Relu_Zero:** Checks that ReLU returns zero for zero input.
- **Test_Relu_Negative:** Checks that ReLU clamps negative values to zero.
- **Test_Exp_Approx_0:** Verifies exponential approximation at zero is within 2% of the true value.
- **Test_Exp_Approx_Full_Range:** Sweeps input from -8 to +8, checking exponential approximation accuracy (absolute error for tiny values, relative error < 0.8% for normal values).
- **Test_Softmax_SineDataset_SingleLogit_Returns255:** Checks softmax of a single value returns 255.
- **test_Softmax_LorenzDataset_3Values_NormalizedAndScaled:** Checks softmax on three values: output is normalized, scaled to sum ≈ 255, and sorted in descending order.
- **test_Softmax_NegativeValues_StableNormalization:** Checks softmax on negative values: all outputs are in [0, 255] and sum ≈ 255.
- **test_Softmax_LargeValues_NoOverflow:** Checks softmax on large values: outputs do not overflow and sum ≈ 255.
