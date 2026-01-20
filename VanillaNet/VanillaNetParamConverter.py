import numpy as np
import json
import os

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PARAMS_PATH = os.path.join("VanillaNet", "params", "vanillanet_paramsV2.json")
VANILLANET_WEIGHT_FOLDER = os.path.join("VanillaNet", "Weights", "vanillanet_int32_weights")

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
    weight_q16 = float_to_q16(layer["weight"])
    bias_q16 = float_to_q16(layer["bias"])
    return weight_q16, bias_q16

def GetWeightShape2ndValue(weight):
    weight_array = np.array(weight)
    if weight_array.ndim < 2:
        return weight_array.shape[0]
    return weight_array.shape[1]

def ToCArray2D(arr):
    rows = []
    for row in arr:
        row_str = ", ".join(str(int(x)) for x in row)
        rows.append(f"{{ {row_str} }}")
    return "{\n    " + ",\n    ".join(rows) + "\n}"

def ToCArray1D(arr):
    return "{ " + ", ".join(str(int(x)) for x in arr) + " }"


# --------------------------------------------------
# MAIN CONVERSION
# --------------------------------------------------
def ConvertVanillaNetParamsToQ16(params_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(params_path, "r") as f:
        params = json.load(f)

    for layer_name, layer in params.items():
        weight_q16 = float_to_q16(layer["weight"])
        bias_q16   = float_to_q16(layer["bias"])

        weight_str = ToCArray2D(weight_q16)
        bias_str   = ToCArray1D(bias_q16)


        rows, cols = weight_q16.shape

        out_path = os.path.join(output_folder, f"{layer_name}_q16.h")
        with open(out_path, "w") as file:
            file.write(f"// {layer_name} (Q16.16)\n")
            file.write(f"#ifndef {layer_name.upper()}_Q16_H\n")
            file.write(f"#define {layer_name.upper()}_Q16_H\n\n")
            file.write("#include <stdint.h>\n\n")

            file.write(f"#define {layer_name.upper()}_OUT_DIM {rows}\n")
            file.write(f"#define {layer_name.upper()}_IN_DIM {cols}\n\n")

            file.write(
                f"static const int32_t {layer_name}_weights[{rows}][{cols}] = {weight_str};\n\n"
            )
            file.write(
                f"static const int32_t {layer_name}_biases[{rows}] = {bias_str};\n\n"
            )

            file.write(f"#endif // {layer_name.upper()}_Q16_H\n")

        print(f"[INFO] Saved {layer_name} to {out_path}")


# --------------------------------------------------
# RUN CONVERSION
# --------------------------------------------------
def main():
    ConvertVanillaNetParamsToQ16(PARAMS_PATH, VANILLANET_WEIGHT_FOLDER)

if __name__ == "__main__":
    main()