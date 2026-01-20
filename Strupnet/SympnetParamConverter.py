import numpy as np
import json
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PARAMS_PATH = os.path.join("Strupnet", "params", "sympnet_params.json") 
SYMPNET_WEIGHT_FOLDER = os.path.join("Strupnet", "Weights", "sympnet_int32_weights")

Q16_SCALE = 1 << 16  # 65536

# --------------------------------------------------
# Q16.16 QUANTIZATION
# --------------------------------------------------
def float_to_q16(x):
    return np.round(np.array(x, dtype=np.float64) * Q16_SCALE).astype(np.int32)

# --------------------------------------------------
# CONVERT ONE LAYER
# --------------------------------------------------
def QuantizeLayerQ16(layer):
    a_q16 = float_to_q16(layer["a"])
    w_q16 = float_to_q16(layer["w"])
    return a_q16, w_q16

# --------------------------------------------------
# MAIN CONVERSION
# --------------------------------------------------
def ConvertSympNetParamsToQ16(params_path, output_folder):
    if not os.path.exists(output_folder):
        print(f"[INFO] Creating directory: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

    with open(params_path, "r") as f:
        params = json.load(f)

    for i, layer in enumerate(params):
        a_q16, w_q16 = QuantizeLayerQ16(layer)

        out_path = os.path.join(output_folder, f"Sympnet_layer_{i}.h")
        with open(out_path, "w") as file:
            file.write(f"// SympNet Layer {i} (Q16.16)\n")
            file.write(f"#ifndef SYMPNET_LAYER_{i}_H\n")
            file.write(f"#define SYMPNET_LAYER_{i}_H\n\n")
            file.write(f"#include <stdint.h>\n\n")

            file.write(f"#define COEFFICIENTS_LEN {len(a_q16)}\n")
            file.write(f"#define WEIGHTS_LEN {len(w_q16)}\n\n")

            file.write(
                f"static const int32_t Sympnet_layer_{i}_a[COEFFICIENTS_LEN] = {{\n    "
            )
            file.write(", ".join(map(str, a_q16)))
            file.write("\n};\n\n")

            file.write(
                f"static const int32_t Sympnet_layer_{i}_w[WEIGHTS_LEN] = {{\n    "
            )
            file.write(", ".join(map(str, w_q16)))
            file.write("\n};\n\n")

            file.write(f"#endif // SYMPNET_LAYER_{i}_H\n")

        print(f"[OK] Wrote {out_path}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
def main():
    ConvertSympNetParamsToQ16(PARAMS_PATH, SYMPNET_WEIGHT_FOLDER)

if __name__ == "__main__":
    main()
