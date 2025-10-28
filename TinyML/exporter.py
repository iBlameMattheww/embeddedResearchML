import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model(r"C:\Users\Matthew\embeddedResearchML\TinyML\model.keras")
folder = r"C:\Users\Matthew\embeddedResearchML\TinyML\Target\include\exported_weights"
model_layers = r"C:\Users\Matthew\embeddedResearchML\TinyML\Target\include\model_layers.h"
lut_file = r"C:\Users\Matthew\embeddedResearchML\TinyML\Target\tools\exp_lut.h"
if not os.path.exists(folder):
    os.makedirs(folder)


for i, layer in enumerate(model.layers):
    if isinstance(layer, tf.keras.layers.Dense):
        W, b = layer.get_weights()
        np.savez(os.path.join(folder, f"dense_{i}.npz"), W=W, b=b)

        # Manual symmetric quantization to int8
        scale = np.max(np.abs(W))
        W_q = np.round((W / scale) * 127).astype(np.int8)
        b_q = np.round(b).astype(np.int16)  # keep bias higher precision

        with open(os.path.join(folder, f"dense_{i}.h"), "w") as f:
            f.write(f"// Layer {i}\n")
            f.write(f"#include <stdint.h>\n")
            f.write(f"#define L{i}_IN {W.shape[0]}\n")
            f.write(f"#define L{i}_OUT {W.shape[1]}\n")
            f.write(f"#define L{i}_SCALE {scale:.8f}\n\n")
            f.write("static const int8_t W%d[%d][%d] = {\n" %
                    (i, W.shape[1], W.shape[0]))
            for row in W_q.T:
                f.write("{" + ",".join(map(str, row)) + "},\n")
            f.write("};\n")
            f.write("static const int16_t B%d[%d] = {%s};\n\n" %
                    (i, W.shape[1], ",".join(map(str, b_q))))

with open(model_layers, "w") as f:
    f.write("#ifndef MODEL_LAYERS_H\n")
    f.write("#define MODEL_LAYERS_H\n\n")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            f.write(f"#include \"exported_weights/dense_{i}.h\"\n")
    f.write("\n#endif // MODEL_LAYERS_H\n")

with open(lut_file, "w") as f:
    f.write("// Exponential Lookup Table\n")
    f.write("#include <stdint.h>\n\n")
    x_min, x_max, step = -8, 8, 0.25
    xs = np.arange(x_min, x_max + step, step)
    ys = np.exp(xs)

    scale = 127 / np.max(ys)      # normalize to int8 range
    ys_q = np.round(ys * scale).astype(np.uint8)

    f.write(f"#define EXP_MIN {x_min}f\n")
    f.write(f"#define EXP_STEP {step}f\n")
    f.write(f"#define EXP_SIZE {len(xs)}\n")
    f.write("static const uint8_t exp_lut[EXP_SIZE] = {\n")
    f.write(", ".join(map(str, ys_q)) + "};\n")

