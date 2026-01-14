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
    dt = 0.05
    p0, q0 = 0.0, 1.0

    learnedTraj = np.zeros((T, 2))
    groundTruthTraj = np.zeros((T, 2))

    # Learned state (p, q)
    p, q = p0, q0

    # Ground truth state (q, p) for Verlet
    q_gt, p_gt = q0, p0

    for t in range(T):
        learnedTraj[t] = [p, q]
        groundTruthTraj[t] = [p_gt, q_gt]

        # Step both systems forward
        p, q = VanillaNet_step(p, q, params, dt)
        q_gt, p_gt = VelocityVerlet([q_gt, p_gt], dt)

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
         label='Learned Trajectory (Hidden Dim 2)')
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
plt.plot(E, label='Learned Energy (Hidden Dim 2)')
plt.plot(E_gt, linestyle='--', label='Ground Truth Energy')
plt.xlabel("Time step")
plt.ylabel("Energy")
plt.title("Energy vs Time")
plt.grid(True)
plt.legend()

# Layout + show
plt.tight_layout()
plt.show()