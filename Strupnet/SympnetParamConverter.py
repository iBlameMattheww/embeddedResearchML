import numpy as np
import json
import os

PARAMS_PATH = os.path.join("Strupnet", "params", "sympnet_params.json")
SYMPNET_WEIGHT_FOLDER = os.path.join("Strupnet", "Weights", "sympnet_int8_weights")

def Int8QuantizeParam(layer, param):
    param = np.array(layer[param], dtype=np.float32)
    scale = 127.0 / np.max(np.abs(param))
    param_int8 = np.round(param * scale).astype(np.int8)
    return param_int8

def Int16QuantizeParam(layer, param):
    param = np.array(layer[param], dtype=np.float32)
    scale = 32767.0 / np.max(np.abs(param))
    param_int16 = np.round(param * scale).astype(np.int16)
    return param_int16

def QuantizeLayer(layer):
    a_int16 = Int16QuantizeParam(layer, "a")
    w_int8 = Int8QuantizeParam(layer, "w")
    return (a_int16, w_int8)

def ConvertSympNetParamsToInt8(PARAMS_PATH, SYMPNET_WEIGHT_FOLDER):
    if not os.path.exists(SYMPNET_WEIGHT_FOLDER):
        print(f"[INFO] Creating directory at {SYMPNET_WEIGHT_FOLDER}")
        os.makedirs(SYMPNET_WEIGHT_FOLDER)
    
    with open(PARAMS_PATH, "r") as f:
        params = json.load(f)

    for i, layer in enumerate(params):
        with open(os.path.join(SYMPNET_WEIGHT_FOLDER, f"Sympnet_layer_{i}.h"), "w") as file:
            a_int16, w_int8 = QuantizeLayer(layer)
            
            file.write(f"// SympNet Layer {i} Weights\n")
            file.write(f"#ifndef SYMPNET_LAYER_{i}_H\n")
            file.write(f"#define SYMPNET_LAYER_{i}_H\n\n")
            file.write(f"#include <stdint.h>\n\n")
            file.write(f"#define COEFFICIENTS_LEN {len(a_int16)}\n\n")
            file.write(f"#define WEIGHTS_LEN {len(w_int8)}\n\n")

            file.write(f"static const int16_t Sympnet_layer_{i}_a[COEFFICIENTS_LEN] = {{")
            file.write(", ".join(map(str, a_int16)))
            file.write("};\n\n")

            file.write(f"static const int8_t Sympnet_layer_{i}_w[WEIGHTS_LEN] = {{")
            file.write(", ".join(map(str, w_int8)))
            file.write("};\n")

            file.write(f"\n#endif // SYMPNET_LAYER_{i}_H\n")

def main():
    ConvertSympNetParamsToInt8(PARAMS_PATH, SYMPNET_WEIGHT_FOLDER)

if __name__ == "__main__":
    main()