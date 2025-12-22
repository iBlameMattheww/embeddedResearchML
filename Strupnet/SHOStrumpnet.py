from strupnet import SympNet
from tqdm import tqdm
import numpy as np
import torch
import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(CURRENT_DIR, 'params')
PARAMS_PATH = os.path.join(PARAMS_PATH, 'sympnet_params.json')
DATASET_PATH = os.path.join("SimpleHarmonicOscillator", "data", "sho_trajectories.npy")

DEVICE = torch.device("cpu")
print("[INFO] Using device:", DEVICE)

EPOCHS = 1000

sympnet = SympNet(
    dim = 1,
    layers = 2,
    max_degree = 2,
    method = 'P',
)

def LoadDataset(path, dt, train_frac = 0.8):
    data = np.load(path) # shape (samples, timesteps, 2)
    data = data[:, : , [1, 0]]  # swap to (p, q) ordering

    N = data.shape[0]
    split = int(N * train_frac)

    train = data[:split]
    test = data[split:]

    def MakePairs(arr):
        x0 = arr[:, :-1, :]  # shape (samples, timesteps-1, 2)
        x1 = arr[:, 1:, :]   # shape (samples, timesteps-
        return (
            torch.tensor(x0.reshape(-1, 2), dtype=torch.float64),
            torch.tensor(x1.reshape(-1, 2), dtype=torch.float64),
        )
    
    x0_train, x1_train = MakePairs(train)
    x0_test, x1_test = MakePairs(test)

    dtTensorTrain = dt * torch.ones(len(x0_train), 1)
    dtTensorTest = dt * torch.ones(len(x0_test), 1)

    return x0_train, x1_train, dtTensorTrain, x0_test, x1_test, dtTensorTest

dt = 0.05
x0, x1, dt_train, x0_test, x1_test, dt_test = LoadDataset(DATASET_PATH, dt)

sympnet.double().to(DEVICE)
x0 = x0.to(DEVICE)
x1 = x1.to(DEVICE)
dt_train = dt_train.to(DEVICE)
x0_test = x0_test.to(DEVICE)
x1_test = x1_test.to(DEVICE)
dt_test = dt_test.to(DEVICE)

optimizer = torch.optim.Adam(sympnet.parameters(), lr=0.01)
mse = torch.nn.MSELoss()
for epoch in tqdm(range(EPOCHS), desc="Training SympNet on SHO data"):
    optimizer.zero_grad()    
    x1_pred = sympnet(x=x0, dt=dt_train)
    loss = mse(x1, x1_pred)
    loss.backward()
    optimizer.step()

    N = len(x0)
    print(f"\nEpoch {epoch+1}/{EPOCHS} "
          f"Loss: {loss.item():.6f}")

params = []
for i, layer in enumerate(sympnet.layers_list):
    params.append({
        "layer": i,
        "a": layer.params["a"].detach().cpu().numpy().tolist(),
        "w": layer.params["w"].detach().cpu().numpy().tolist(),
    })  

os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)

sympnet.eval()
with torch.no_grad():
    x1_test_pred = sympnet(x=x0_test, dt=dt_test)
test_error = torch.norm(x1_test_pred - x1_test).item()

print("Final training loss:", loss.item())
print("Test set error:", test_error)