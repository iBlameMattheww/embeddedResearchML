import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
CSV_PATH = "Benchmarks/SHO_Results/sho_inference_results.csv"
OUTPUT_PATH = "/Users/matthewobrien/Documents/sympnet_compare.svg"

STEPS = 500
T = 2 * np.pi
DT = 0.05

# -------------------------------
# SHO INTEGRATOR (GROUND TRUTH)
# -------------------------------
def VelocityVerlet(state, dt):
    q, p = state
    p_half = p - 0.5 * dt * q
    q_new = q + dt * p_half
    p_new = p_half - 0.5 * dt * q_new
    return q_new, p_new

def GenerateSHOFromInitial(q0, p0, steps, dt):
    traj = np.zeros((steps, 2))
    q, p = q0, p0

    for i in range(steps):
        traj[i] = [q, p]
        q, p = VelocityVerlet((q, p), dt)

    return traj

# -------------------------------
# LOAD EMBEDDED DATA
# -------------------------------
df = pd.read_csv(CSV_PATH)

traj_id = np.random.choice(df["trajectory_index"].unique())
traj_df = df[df["trajectory_index"] == traj_id].sort_values("step_index")

q_emb = traj_df["q"].to_numpy()
p_emb = traj_df["p"].to_numpy()

# Initial conditions from embedded run
q0 = q_emb[0]
p0 = p_emb[0]

print(f"Trajectory {traj_id}")
print(f"Initial conditions: q0={q0:.6f}, p0={p0:.6f}")

# -------------------------------
# GENERATE ANALYTICAL TRAJECTORY
# -------------------------------
gt_traj = GenerateSHOFromInitial(q0, p0, STEPS, DT)
q_gt = gt_traj[:, 0]
p_gt = gt_traj[:, 1]

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(12, 5))

# Phase space
plt.subplot(1, 2, 1)
plt.plot(q_gt, p_gt, "--", label="Analytical", linewidth=2)
plt.plot(q_emb, p_emb, label="Embedded", alpha=0.8)
plt.axis("equal")
plt.grid(True)
plt.xlabel("q")
plt.ylabel("p")
plt.title("Phase Space")
plt.legend()

# Time series
plt.subplot(1, 2, 2)
plt.plot(q_gt, "--", label="q (Analytical)")
plt.plot(q_emb, label="q (Embedded)", alpha=0.8)
plt.plot(p_gt, "--", label="p (Analytical)")
plt.plot(p_emb, label="p (Embedded)", alpha=0.8)
plt.grid(True)
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Time Series")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PATH)
plt.show()
