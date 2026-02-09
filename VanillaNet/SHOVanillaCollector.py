import os
import pandas as pd
import numpy as np
import json


TRAINING_DATA_FILE = "SimpleHarmonicOscillator/data/sho_trajectories.npy"
TEST_IID_DATA_FILE = "SimpleHarmonicOscillator/data/sho_Test_IID_Trajectories.npy"
TEST_OOD_DATA_FILE = "SimpleHarmonicOscillator/data/sho_Test_OOD_Trajectories.npy"



RESULTS_DIR = "VanillaNet/SHO_Results"

def DataLoader(path):
    data = np.load(path)
    return data

def CollectFirstInitialValuesFromDataset(data): 
    initialConditions = []
    for trajectory in data:
        q0, p0 = trajectory[0]
        initialConditions.append((q0, p0))
    return initialConditions

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

def rollout_vanillanet_numpy(q0, p0, params, dt, steps):
    q, p = q0, p0
    traj = []

    for k in range(steps + 1):
        traj.append((q, p))
        p, q = VanillaNet_step(p, q, params, dt)

    return traj


def generate_csv_from_dataset(
    dataset_path,
    rollout_fn,
    rollout_name,
    dt,
    steps,
    output_path,
    params=None
):
    data = DataLoader(dataset_path)
    initial_conditions = CollectFirstInitialValuesFromDataset(data)

    records = []

    for traj_idx, (q0, p0) in enumerate(initial_conditions):
        if params is not None:
            traj = rollout_fn(q0, p0, params, dt, steps)
        else:
            traj = rollout_fn(q0, p0, dt, steps)

        for step_idx, (q, p) in enumerate(traj):
            records.append({
                "trajectory_index": traj_idx,
                "step_index": step_idx,
                "q": float(q),
                "p": float(p),
                "inference_time_sec": 0.0  # placeholder for FP32 / GT
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[INFO] Saved {rollout_name} results to {output_path}")

params = GetParams()
DT = 0.05
STEPS = 500

generate_csv_from_dataset(
    TRAINING_DATA_FILE,
    rollout_vanillanet_numpy,
    "train_fp32",
    DT,
    STEPS,
    f"{RESULTS_DIR}/train_fp32.csv",
    params=params
)

generate_csv_from_dataset(
    TEST_IID_DATA_FILE,
    rollout_vanillanet_numpy,
    "iid_fp32",
    DT,
    STEPS,
    f"{RESULTS_DIR}/iid_fp32.csv",
    params=params
)

generate_csv_from_dataset(
    TEST_OOD_DATA_FILE,
    rollout_vanillanet_numpy,
    "ood_fp32",
    DT,
    STEPS,
    f"{RESULTS_DIR}/ood_fp32.csv",
    params=params
)
