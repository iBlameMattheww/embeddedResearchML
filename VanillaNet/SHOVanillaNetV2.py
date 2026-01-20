from tqdm import tqdm
import numpy as np
import torch
import json
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(CURRENT_DIR, 'params')
PARAMS_PATH = os.path.join(PARAMS_PATH, 'vanillanet_paramsV2.json')
DATASET_PATH = os.path.join("SimpleHarmonicOscillator", "data", "sho_trajectories.npy")

DEVICE = torch.device("cpu")
print("[INFO] Using device:", DEVICE)

EPOCHS = 1000
HIDDEN_DIM = 32
DT = 0.05

class VanillaNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, dt):
        dx = self.net(x)
        return x + dt * dx

    
def LoadDataset(path, dt, train_frac = 0.8):
    data = np.load(path)
    data = data[:, : , [1, 0]]  # swap to (p, q) ordering

    N = data.shape[0]
    split = int(N * train_frac)

    train = data[:split]
    test = data[split:]

    def MakePairs(arr):
        x0 = arr[:, :-1, :]  # shape (samples, timesteps-1, 2)
        x1 = arr[:, 1:, :]   # shape (samples, timesteps-1, 2)
        return (
            torch.tensor(x0.reshape(-1, 2), dtype=torch.float64),
            torch.tensor(x1.reshape(-1, 2), dtype=torch.float64),
        )
    
    x0_train, x1_train = MakePairs(train)
    x0_test, x1_test = MakePairs(test)

    dtTensorTrain = dt * torch.ones(len(x0_train), 1)
    dtTensorTest = dt * torch.ones(len(x0_test), 1)

    return x0_train, x1_train, dtTensorTrain, x0_test, x1_test, dtTensorTest

dt = DT
x0, x1, dt_train, x0_test, x1_test, dt_test = LoadDataset(DATASET_PATH, dt)

vanillanet = VanillaNet(hidden_dim=HIDDEN_DIM).double().to(DEVICE)

x0 = x0.to(DEVICE)
x1 = x1.to(DEVICE)
dt_train = dt_train.to(DEVICE)

optimizer = torch.optim.Adam(vanillanet.parameters(), lr=1e-3)
mse = torch.nn.MSELoss()

for epoch in tqdm(range(EPOCHS), desc="Training VanillaNet on SHO data"):
    optimizer.zero_grad()    
    x1_pred = vanillanet(x=x0, dt=dt_train)
    loss = mse(x1, x1_pred)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

params = {
    "fc1": {
        "weight": vanillanet.net[0].weight.detach().cpu().numpy().tolist(),
        "bias": vanillanet.net[0].bias.detach().cpu().numpy().tolist(),
    },
    "fc2": {
        "weight": vanillanet.net[2].weight.detach().cpu().numpy().tolist(),
        "bias": vanillanet.net[2].bias.detach().cpu().numpy().tolist(),
    },
}

os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
with open(PARAMS_PATH, "w") as f:
    json.dump(params, f, indent=2)

vanillanet.eval()
x0_test = x0_test.to(DEVICE)
x1_test = x1_test.to(DEVICE)
dt_test = dt_test.to(DEVICE)

with torch.no_grad():
    x1_test_pred = vanillanet(x0_test, dt_test)

testError = torch.norm(x1_test_pred - x1_test).item()

print("Final Training Loss:", loss.item())
print(f"Test Error (Frobenius Norm): {testError:.6f}")
