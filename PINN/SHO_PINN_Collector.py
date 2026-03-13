import os
import pandas as pd
import numpy as np
import json


TRAINING_DATA_FILE = "SimpleHarmonicOscillator/data/sho_trajectories.npy"
TEST_IID_DATA_FILE = "SimpleHarmonicOscillator/data/sho_Test_IID_Trajectories.npy"
TEST_OOD_DATA_FILE = "SimpleHarmonicOscillator/data/sho_Test_OOD_Trajectories.npy"



RESULTS_DIR = "PINN/SHO_Results"

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
    path = "PINN/params/h_pinn_params.json"
    if not os.path.exists(path):
        raise FileNotFoundError("Parameter file not found.")
    with open(path, "r") as f:
        return json.load(f)
    
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

def rollout_pinn_numpy(q0, p0, params, dt, steps):

    x = np.array([p0, q0])
    traj = []

    for k in range(steps + 1):
        traj.append(x.copy())

        x = RK4Step(x, dt, params)

    return np.array(traj)

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

        for step_idx, (p, q) in enumerate(traj):
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
    rollout_pinn_numpy,
    "train_fp32",
    DT,
    STEPS,
    f"{RESULTS_DIR}/train_fp32.csv",
    params=params
)

generate_csv_from_dataset(
    TEST_IID_DATA_FILE,
    rollout_pinn_numpy,
    "iid_fp32",
    DT,
    STEPS,
    f"{RESULTS_DIR}/iid_fp32.csv",
    params=params
)

generate_csv_from_dataset(
    TEST_OOD_DATA_FILE,
    rollout_pinn_numpy,
    "ood_fp32",
    DT,
    STEPS,
    f"{RESULTS_DIR}/ood_fp32.csv",
    params=params
)