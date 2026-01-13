import numpy as np
import json
import os
import matplotlib.pyplot as plt

# =========================
# Load trained parameters
# =========================
def GetParams():
    path = "VanillaNet/params/vanillanet_paramsV2.json"
    if not os.path.exists(path):
        raise FileNotFoundError("Parameter file not found.")
    with open(path, "r") as f:
        return json.load(f)
    
# ======================================
# Single VanillaNet layer (NumPy version)
# ======================================
def VanillaNet_step(p, q, params, h):
    x = np.array([p, q])

    # Forward pass through the network
    # Layer 1
    w1 = np.array(params["fc1"]["weight"], dtype=float)
    b1 = np.array(params["fc1"]["bias"], dtype=float)
    z1 = np.dot(w1, x) + b1
    a1 = np.maximum(0, z1)  # ReLU activation

    # Layer 2
    w2 = np.array(params["fc2"]["weight"], dtype=float)
    b2 = np.array(params["fc2"]["bias"], dtype=float)
    dx = np.dot(w2, a1) + b2

    # Update step
    x_new = x + h * dx

    return x_new[0], x_new[1]

# =========================
# Main simulation
# =========================
def main():
    params = GetParams()    

    T = 1000
    dt = 0.05
    p0, q0 = 0.0, 1.0

    traj = np.zeros((T, 2))
    p, q = p0, q0

    for t in range(T):
        traj[t] = [p, q]
        p, q = VanillaNet_step(p, q, params, dt)

    return traj

# =========================
# Run + plots
# =========================
traj = main()

# Phase space
plt.figure()
plt.plot(traj[:, 1], traj[:, 0])
plt.xlabel("q (position)")
plt.ylabel("p (momentum)")
plt.axis("equal")
plt.title("Phase space: learned VanillaNet")
plt.show()


# Energy
def energy(p, q):
    return 0.5 * (p**2 + q**2)

E = energy(traj[:, 0], traj[:, 1])

plt.figure()
plt.plot(E)
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.title("Energy vs time")
plt.show()