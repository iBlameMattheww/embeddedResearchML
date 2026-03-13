import numpy as np
import json
import os
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
PARAMS_PATH = "PINN/params/h_pinn_params.json"


DT = 0.05
STEPS = 500

# =========================
# Load params
# =========================
def GetParams():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError("Parameter file not found.")
    with open(PARAMS_PATH, "r") as f:
        return json.load(f)

# =========================
# Forward pass (vector field)
# model(x) -> (dp/dt, dq/dt)
# =========================
def ForwardStep(x, params):

    layer_keys = sorted(params.keys(), key=lambda k: int(k.replace("fc", "")))

    for i, key in enumerate(layer_keys):
        W = np.array(params[key]["weight"]).T
        b = np.array(params[key]["bias"])
        x = x @ W + b

        if i < len(layer_keys) - 1:
            x = np.tanh(x)

    return x  # (dp/dt, dq/dt)

def RK4Step(x, dt, params):

    k1 = ForwardStep(x, params)
    k2 = ForwardStep(x + 0.5 * dt * k1, params)
    k3 = ForwardStep(x + 0.5 * dt * k2, params)
    k4 = ForwardStep(x + dt * k3, params)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# =========================
# Ground Truth (Velocity Verlet)
# =========================
def VelocityVerlet(state, dt):
    p, q = state
    p_half = p - 0.5 * dt * q
    q_new = q + dt * p_half
    p_new = p_half - 0.5 * dt * q_new
    return [p_new, q_new]

# =========================
# Rollout
# =========================
def main():

    params = GetParams()

    learned = np.zeros((STEPS, 2))
    gt = np.zeros((STEPS, 2))

    x = np.array([[0.0, 1.3]])   # (p,q)
    x_gt = [0.0, 1.3]

    for k in range(STEPS):

        learned[k] = x[0]
        gt[k] = x_gt

        # -----------------------------------
        # Neural RK4 rollout
        # -----------------------------------
        x = RK4Step(x, DT, params)

        # -----------------------------------
        # Ground truth
        # -----------------------------------
        x_gt = VelocityVerlet(x_gt, DT)

    return learned, gt

learned, gt = main()

learnedTraj = learned
groundTruthTraj = gt

plt.figure(figsize=(12, 5))

# =========================
# Phase space
# =========================
plt.subplot(1, 2, 1)
plt.plot(learnedTraj[:, 1], learnedTraj[:, 0], label="PINN")
plt.plot(groundTruthTraj[:, 1], groundTruthTraj[:, 0],
         linestyle="dashed", label="Ground Truth")
plt.xlabel("q")
plt.ylabel("p")
plt.title("Phase Space")
plt.axis("equal")
plt.grid(True)
plt.legend()

# =========================
# Energy
# =========================
def energy(p, q):
    return 0.5 * (p**2 + q**2)

E = energy(learnedTraj[:, 0], learnedTraj[:, 1])
E_gt = energy(groundTruthTraj[:, 0], groundTruthTraj[:, 1])

plt.subplot(1, 2, 2)
plt.plot(E, label="PINN Energy")
plt.plot(E_gt, linestyle="--", label="Ground Truth")
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.title("Energy vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
