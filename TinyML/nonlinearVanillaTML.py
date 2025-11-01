import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==============================
#   Setup paths
# ==============================
base_dir = r"C:\Users\Matthew\embeddedResearchML\TinyML"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

MODEL_TF = os.path.join(base_dir, "model")
MODEL_TFLITE_FLOAT = os.path.join(base_dir, "model_float.tflite")
MODEL_TFLITE_INT8 = os.path.join(base_dir, "model_int8.tflite")
MODEL_TFLITE_INT8_CC = os.path.join(base_dir, "model_int8.cc")

# ==============================
#   Generate sine dataset
# ==============================
np.random.seed(42)
tf.random.set_seed(42)
samples = 1000

# Inputs in [-1, 1], outputs = sin(pi * x)
X = np.random.uniform(-1, 1, size=samples).astype(np.float32)
Y = np.sin(np.pi * X).astype(np.float32)
Y += 0.05 * np.random.randn(*Y.shape)  # small noise

plt.title("Training data (sine wave)")
plt.plot(X, Y, "b.")
plt.show()

# Split
X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42)

# ==============================
#   Define and train model
# ==============================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(X_train, Y_train, epochs=500, batch_size=32,
                    validation_data=(X_val, Y_val), verbose=0)

plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.legend()
plt.title("Training/Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.show()

# Export trained model
model.export(MODEL_TF)
model.save(os.path.join(base_dir, "model.keras"))

# ==============================
#   Convert to TFLite (float)
# ==============================
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
tflite_model_float = converter.convert()
open(MODEL_TFLITE_FLOAT, "wb").write(tflite_model_float)

# ==============================
#   Convert to TFLite (INT8)
# ==============================
def representative_dataset():
    for _ in range(500):
        x = np.random.uniform(-1, 1, size=(1, 1)).astype(np.float32)
        yield [x]

quant_converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quant_converter.representative_dataset = representative_dataset
quant_converter.inference_input_type = tf.int8
quant_converter.inference_output_type = tf.int8
tflite_model_int8 = quant_converter.convert()
open(MODEL_TFLITE_INT8, "wb").write(tflite_model_int8)

# ==============================
#   TFLite inference helper
# ==============================
def predict_tflite(tflite_model, X_input):
    X_input = X_input.reshape((-1, 1)).astype(np.float32)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    preds = np.zeros(X_input.shape[0], dtype=np.float32)
    for i in range(X_input.shape[0]):
        x = X_input[i:i+1]
        if input_details["dtype"] == np.int8:
            x = np.round(x / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] == np.int8:
            y = (y.astype(np.float32) - output_zero_point) * output_scale
        preds[i] = y
    return preds

# ==============================
#   Compare predictions
# ==============================
Y_pred_tf = model.predict(X_test)
Y_pred_tflite_float = predict_tflite(tflite_model_float, X_test)
Y_pred_tflite_int8 = predict_tflite(tflite_model_int8, X_test)

plt.title("Comparison of various models against actual values")
plt.plot(X_test, Y_test, "bo", label="Actual values")
plt.plot(X_test, Y_pred_tf, "ro", label="TF predictions")
plt.plot(X_test, Y_pred_tflite_float, "bx", label="TFLite predictions")
plt.plot(X_test, Y_pred_tflite_int8, "gx", label="TFLite quantized predictions")
plt.legend()
plt.show()

# ==============================
#   Evaluate losses
# ==============================
loss_tf, _ = model.evaluate(X_test, Y_test, verbose=0)
mse_float = tf.keras.losses.MeanSquaredError()(Y_test, Y_pred_tflite_float).numpy()
mse_int8 = tf.keras.losses.MeanSquaredError()(Y_test, Y_pred_tflite_int8).numpy()

print("\nModel Losses (MSE):")
print(f"TF Model:          {loss_tf:.6f}")
print(f"TFLite (Float):    {mse_float:.6f}")
print(f"TFLite (Quantized):{mse_int8:.6f}")

# ==============================
#   Check quantization scales
# ==============================
interpreter = tf.lite.Interpreter(model_content=tflite_model_int8)
interpreter.allocate_tensors()
print("\nQuantization parameters:")
print("Input quantization:", interpreter.get_input_details()[0]["quantization"])
print("Output quantization:", interpreter.get_output_details()[0]["quantization"])

# ==============================
#   Export to C array (.cc)
# ==============================
with open(MODEL_TFLITE_INT8_CC, "w") as f:
    f.write("// Automatically generated from model_int8.tflite\n")
    f.write("const unsigned char model_int8_tflite[] = {\n")
    for i, val in enumerate(tflite_model_int8):
        f.write(f"0x{val:02x},")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("};\n")
    f.write(f"const int model_int8_tflite_len = {len(tflite_model_int8)};\n")

print(f"\nExported C array to: {MODEL_TFLITE_INT8_CC}")