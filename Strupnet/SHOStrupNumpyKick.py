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

def VelocityVerlet(state, dt):
    yHalf = state[1] - 0.5 * dt * state[0]
    xNew = state[0] + dt * yHalf
    yNew = yHalf - 0.5 * dt * xNew
    return [xNew, yNew] 

# =========================
# Main simulation
# =========================
def main():
    params = GetParams()

    T = 500
    h = 0.05          # MUST match training timestep
    p0, q0 = 0.0, 1.0

    learnedTraj = np.zeros((T, 2))
    groundTruthTraj = np.zeros((T, 2))
    
    p, q = p0, q0
    q_gt, p_gt = q0, p0

    for t in range(T):
        learnedTraj[t] = [p, q]
        groundTruthTraj[t] = [p_gt, q_gt]
        p, q = SymplecticStep(p, q, params, h)
        q_gt, p_gt = VelocityVerlet([q_gt, p_gt], h)

    return learnedTraj, groundTruthTraj


# =========================
# Run + plots
# =========================
learnedTraj, groundTruthTraj = main()

plt.figure(figsize=(12, 5))

# ======================
# Phase space (left)
# ======================
plt.subplot(1, 2, 1)
plt.plot(learnedTraj[:, 1], learnedTraj[:, 0],
         label='Learned Trajectory (SympNet)')
plt.plot(groundTruthTraj[:, 1], groundTruthTraj[:, 0],
         linestyle='dashed', label='Ground Truth')
plt.xlabel("q (position)")
plt.ylabel("p (momentum)")
plt.title("Phase Space")
plt.axis("equal")
plt.grid(True)
plt.legend()

# ======================
# Energy (right)
# ======================
def energy(p, q):
    return 0.5 * (p**2 + q**2)

E = energy(learnedTraj[:, 0], learnedTraj[:, 1])
E_gt = energy(groundTruthTraj[:, 0], groundTruthTraj[:, 1])

plt.subplot(1, 2, 2)
plt.plot(E, label='Learned Energy (SympNet)')
plt.plot(E_gt, linestyle='--', label='Ground Truth Energy')
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.title("Energy vs Time")
plt.grid(True)
plt.legend()

# Layout + show
plt.tight_layout()
plt.show()