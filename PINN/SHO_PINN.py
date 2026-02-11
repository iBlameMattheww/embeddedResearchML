from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

# ==================================================
# Paths
# ==================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(CURRENT_DIR, 'params')
PARAMS_PATH = os.path.join(PARAMS_PATH, 'h_pinn_params.json')
DATASET_PATH = os.path.join("SimpleHarmonicOscillator", "data", "sho_trajectories.npy")

DEVICE = torch.device("cpu")
print("[INFO] Using device:", DEVICE)

EPOCHS = 5000
DT = 0.05
DAMPING = 0
W0 = 1

class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

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

def main():
    mu, k = 2*DAMPING, W0**2

    torch.manual_seed(123)
    model = FCN(N_INPUT=2, N_OUTPUT=2, N_HIDDEN=32, N_LAYERS=3).double().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------------------------
    # Sample training states in phase space
    # --------------------------------------------
    N_SAMPLES = 5000
    q = torch.randn(N_SAMPLES, 1, dtype=torch.float64)
    p = torch.randn(N_SAMPLES, 1, dtype=torch.float64)

    x = torch.cat([p, q], dim=1).to(DEVICE)  # (p,q)

    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()

        dx_hat = model(x)

        # True derivatives for SHO
        dp_true = -k * q.to(DEVICE)
        dq_true = p.to(DEVICE)

        dp_hat = dx_hat[:, 0:1]
        dq_hat = dx_hat[:, 1:2]

        # Supervised vector field loss
        loss_data = torch.mean((dp_hat - dp_true)**2 +
                               (dq_hat - dq_true)**2)

        # Physics residual
        r1 = dq_hat - p.to(DEVICE)
        r2 = dp_hat + k * q.to(DEVICE) + mu * p.to(DEVICE)
        loss_phys = torch.mean(r1**2 + r2**2)

        loss = loss_data + 0.1 * loss_phys
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")

    # --------------------------------------------
    # Export parameters (unchanged)
    # --------------------------------------------
    params =  {
        "fc1": {
            "weight": model.fcs[0].weight.detach().cpu().numpy().tolist(),
            "bias": model.fcs[0].bias.detach().cpu().numpy().tolist()
        },
        "fc2": {
            "weight": model.fch[0][0].weight.detach().cpu().numpy().tolist(),
            "bias": model.fch[0][0].bias.detach().cpu().numpy().tolist()
        },
        "fc3": {
            "weight": model.fch[1][0].weight.detach().cpu().numpy().tolist(),
            "bias": model.fch[1][0].bias.detach().cpu().numpy().tolist()
        },
        "fc4": {
            "weight": model.fce.weight.detach().cpu().numpy().tolist(),
            "bias": model.fce.bias.detach().cpu().numpy().tolist()
        }
    }

    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

    print("Final Training Loss:", loss.item())


if __name__ == "__main__":
    main()