import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ---------- SETTINGS ----------
INPUT_DIM = 3
BATCH = 1024             # try 512, 1024, 2048
STEPS = 400              # short timed run; adjust up/down
USE_MIXED = True         # mixed precision on GPU
DEVICE = "/GPU:0"        # switch to "/CPU:0" for CPU test
# -----------------------------

# Mixed precision (helps on 4070)
if USE_MIXED:
    mixed_precision.set_global_policy('mixed_float16')

# Dummy data to benchmark pure compute/pipeline (no disk I/O)
X = np.random.randn(BATCH * STEPS, INPUT_DIM).astype(np.float32)
Y = np.random.randn(BATCH * STEPS, INPUT_DIM).astype(np.float32)

ds = tf.data.Dataset.from_tensor_slices((X, Y)) \
        .shuffle(10000) \
        .batch(BATCH) \
        .prefetch(tf.data.AUTOTUNE)

with tf.device(DEVICE):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        # keep output float32 for numerical stability under mixed precision
        tf.keras.layers.Dense(3, dtype='float32')
    ])
    model.compile(optimizer='adam', loss='mse')

    # Warm-up (avoid first-iteration jitters)
    model.fit(ds.take(10), epochs=1, verbose=0)

    # Timed run
    t0 = time.time()
    hist = model.fit(ds.take(STEPS), epochs=1, verbose=0)
    t1 = time.time()

elapsed = t1 - t0
batches_sec = STEPS / elapsed
samples_sec = batches_sec * BATCH
print(f"Device: {DEVICE} | mixed: {USE_MIXED}")
print(f"Batches/sec: {batches_sec:.1f} | Samples/sec: {samples_sec:,.0f}")

# Example: compute time per epoch for your real dataset
N_real = 500_000
time_per_epoch_sec = (N_real / BATCH) / batches_sec
print(f"With BATCH={BATCH}, estimated seconds/epoch ≈ {time_per_epoch_sec:.1f}")
