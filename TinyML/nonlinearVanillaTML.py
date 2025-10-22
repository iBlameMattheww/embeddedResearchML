import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

modelsDir = 'models/'
if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
MODEL_TF = modelsDir + 'model'
MODEL_NO_QUANT_TFLITE = modelsDir + 'model_no_quant.tflite'
MODEL_TFLITE = modelsDir + 'model.tflite'
MODEL_TFLITE_MICRO = modelsDir + 'model.cc'

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

samples = 1000
X = np.random.uniform(low = 0, high = 2*math.pi, size = samples).astype(np.float32)
np.random.shuffle(X)
Y = np.sin(X).astype(np.float32)
Y += 0.1 * np.random.randn(*Y.shape)

plt.plot(X, Y, 'b.')
plt.show()

X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=seed)
assert (X_train.size + X_val.size + X_test.size) == samples

plt.plot(X_train, Y_train, 'b.', label="Train")
plt.plot(X_test, Y_test, 'r.', label="Test")
plt.plot(X_val, Y_val, 'y.', label="Validate")
plt.legend()
plt.show()

model1 = tf.keras.Sequential()
model1.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)))
model1.add(tf.keras.layers.Dense(16, activation='relu'))
model1.add(tf.keras.layers.Dense(1))
model1.compile(optimizer='adam', loss='mse', metrics=['mae'])

history1 = model1.fit(X_train, Y_train, epochs = 500, batch_size=64,
                        validation_data=(X_val, Y_val))

plt.plot(history1.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'], label='Val Loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

test_loss, test_mae = model1.evaluate(X_test, Y_test)
y_test_pred = model1.predict(X_test)

plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot(X_test, Y_test, 'r.', label='Actual')
plt.plot(X_test, y_test_pred, 'b.', label='Predicted')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()