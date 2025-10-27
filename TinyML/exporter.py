import numpy as np
import tensorflow as tf
import os

model = tf.keras.models.load_model(r"C:\Users\Matthew\embeddedResearchML\TinyML\model.keras")
folder = r"C:\Users\Matthew\embeddedResearchML\TinyML\exported_weights"
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
