import numpy as np
import json
import os
import matplotlib.pyplot as plt


# =========================
# Load trained parameters
# =========================
def GetParams():
    path = "Strupnet/params/sympnet_params.json"
    if not os.path.exists(path):
        raise FileNotFoundError("Parameter file not found.")
    with open(path, "r") as f:
        return json.load(f)


# ======================================
# Single P-SympNet layer (NumPy version)
# ======================================
def P_layer_step(p, q, layer, h):
    a = np.array(layer["a"], dtype=float)   # polynomial coeffs
    w = np.array(layer["w"], dtype=float)   # shape (2,)

    x = np.array([p, q])

    # scalar monomial m = w^T x
    m = np.dot(w, x)

    # polynomial derivative dH/dm
    min_degree = 2
    poly_deriv = 0.0
    for k, ak in enumerate(a):
        degree = min_degree + k
        poly_deriv += degree * ak * (m ** (degree - 1))

    # symplectic direction Jw
    Jw = np.array([w[1], -w[0]])

    # symplectic update
    x_new = x + h * poly_deriv * Jw

    return x_new[0], x_new[1]


# ======================================
# Full SympNet step (all layers)
# ======================================
def SymplecticStep(p, q, params, h):
    for layer in params:
        p, q = P_layer_step(p, q, layer, h)
    return p, q


# =========================
# Main simulation
# =========================
def main():
    params = GetParams()

    T = 1000
    h = 0.05          # MUST match training timestep
    p0, q0 = 0.0, 1.0

    traj = np.zeros((T, 2))
    p, q = p0, q0

    for t in range(T):
        traj[t] = [p, q]
        p, q = SymplecticStep(p, q, params, h)

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
plt.title("Phase space: learned SympNet integrator")
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
