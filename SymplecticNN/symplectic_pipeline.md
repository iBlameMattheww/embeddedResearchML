# ⭐ SymplecticNN Full Pipeline — Beginning to End

---

## 1. Data Loading & Preprocessing

- **500 trajectories, 10,000 time-steps each**
- Each time-step: $(x_t, y_t, z_t)$
- Data shape: `(trajectories, timesteps, 3)`

```python
# dataset_loader.py
class LorenzDataset(Dataset):
    def __init__(self, npyPath: str, normalize: bool = True):
        self.data = np.load(npyPath)  # (num_trajectories, time_steps, 3)
        X = self.data[:, :-1, :].reshape(-1, 3)
        Y = self.data[:, 1:, :].reshape(-1, 3)
        # normalization
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        self.X = (X - self.mean) / self.std
        self.Y = (Y - self.mean) / self.std
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)
```

---

## 2. Encoder: $(x, y, z) \to (q_1, ..., q_4, p_1, ..., p_4)$

- **Learned nonlinear map:** $\phi: \mathbb{R}^3 \to \mathbb{R}^8$
- 8D latent space (Hamiltonian form)

```python
# encoder.py
class Encoder(nn.Module):
    def __init__(self, inputDim=3, latentDim=8, hiddenDim=128, numLayers=3):
        layers = []
        inDim = inputDim
        for _ in range(numLayers):
            layers.append(nn.Linear(inDim, hiddenDim))
            layers.append(nn.Tanh())
            inDim = hiddenDim
        layers.append(nn.Linear(inDim, latentDim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
```

---

## 3. HamiltonianNN: $H(q, p)$

- **Scalar function:** $H: \mathbb{R}^8 \to \mathbb{R}$
- Defines the vector field in latent space

```python
# hamiltonian.py
class HamiltonianNN(nn.Module):
    def __init__(self, dim, hiddenSize=64, numHiddenLayers=2):
        layers = []
        inDim = dim
        for _ in range(numHiddenLayers):
            layers.append(nn.Linear(inDim, hiddenSize))
            layers.append(nn.Tanh())
            inDim = hiddenSize
        layers.append(nn.Linear(inDim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
```

---

## 4. Symplectic Layer: Leapfrog Integration

- **Leapfrog update:**
    - $p_{t+1/2} = p_t - \frac{\Delta t}{2} \frac{\partial H}{\partial q}$
    - $q_{t+1} = q_t + \Delta t \frac{\partial H}{\partial p_{t+1/2}}$
    - $p_{t+1} = p_{t+1/2} - \frac{\Delta t}{2} \frac{\partial H}{\partial q_{t+1}}$
- Preserves symplectic structure, energy stability

```python
# symplectic_layer.py
class SymplecticLayer(nn.Module):
    def __init__(self, latentDim=8, hiddenDim=128, numHiddenLayers=2, dt=0.01, numSteps=1):
        self.hamiltonianNN = HamiltonianNN(latentDim, hiddenDim, numHiddenLayers)
        self.dt = dt
        self.numSteps = numSteps
    def GradH(self, z):
        z = z.clone().detach().requires_grad_(True)
        hVal = self.hamiltonianNN(z)
        grad = torch.autograd.grad(hVal.sum(), z, create_graph=True)[0]
        return grad
    def forward(self, z):
        # Leapfrog integration step
        # ...implementation...
        pass
```

---

## 5. Multi-step Rollout

- **20-step rollout:**
    - $x_t \to z_t \to z_{t+1} \to \cdots \to z_{t+20}$
    - Forces long-term stability and physical accuracy

---

## 6. Decoder: $(q, p) \to (x, y, z)$

- **Inverse map:** $\phi^{-1}: \mathbb{R}^8 \to \mathbb{R}^3$

```python
# decoder.py
class Decoder(nn.Module):
    def __init__(self, latentDim=8, outputDim=3, hiddenDim=128, numLayers=3):
        layers = []
        inDim = latentDim
        for _ in range(numLayers):
            layers.append(nn.Linear(inDim, hiddenDim))
            layers.append(nn.Tanh())
            inDim = hiddenDim
        layers.append(nn.Linear(inDim, outputDim))
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)
```

---

## 7. Losses

- **Reconstruction Loss:**
    - $\|\hat{x}_{t+1} - x_{t+1}\|^2$
- **Rollout Prediction Loss:**
    - $\frac{1}{20} \sum_{k=1}^{20} \|\hat{x}_{t+k} - x_{t+k}\|^2$
- **Energy Drift Loss:**
    - $|H(z_{t+k}) - H(z_t)|$
- **Total Loss:**
    - $L = L_{recon} + L_{rollout} + \lambda L_{energy}$

```python
# metrics.py
import torch.nn.functional as F

def reconstructionLoss(x_pred, x_true):
    return F.mse_loss(x_pred, x_true)

def energyDrift(hamiltonianNN, z0, zT):
    with torch.no_grad():
        H0 = hamiltonianNN(z0).squeeze()
        HT = hamiltonianNN(zT).squeeze()
    drift = torch.abs(HT - H0)
    return drift.mean()
```

---

## 8. Training Loop

- **Backprop through:**
    - Decoder
    - 20-step symplectic integrator
    - HamiltonianNN
    - Encoder
- **Optimizer:** Adam
- **Epochs:** 80

```python
# train_symp.py
for epoch in range(EPOCHS):
    for batch in dataloader:
        x, y = batch
        # Encode
        z = encoder(x)
        # Rollout
        z_rollout = z
        for step in range(ROLLOUT_STEPS):
            z_rollout = symplectic(z_rollout)
        # Decode
        x_hat = decoder(z_rollout)
        # Compute losses
        loss = ...
        # Backprop
        loss.backward()
        optimizer.step()
```

---

## 9. Evaluation & Visualization

- **Rollout prediction:** Compare predicted vs. true trajectories
- **Energy drift:** Track $H(z_{t+k}) - H(z_t)$
- **Plots:**
    - 3D Lorenz attractor
    - Loss curves
    - Energy drift

```python
# eval_sym.py
pred_traj, energies = rollout(model, x0, ROLL_STEPS)
plot_lorenz_plotly(true_traj, pred_traj)
plot_energy_drift(energies)
```

---

## 10. Utilities

- **Seed setting, device selection, checkpointing, logging**

```python
# utils.py
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

---

# End-to-End Pipeline Summary

1. Load and normalize Lorenz data
2. Encode $(x, y, z)$ to $(q, p)$
3. Predict next state using HamiltonianNN + symplectic integrator
4. Decode $(q, p)$ back to $(x, y, z)$
5. Compute losses (reconstruction, rollout, energy drift)
6. Backpropagate through all components
7. Update model parameters
8. Repeat for all epochs
9. Evaluate and visualize results

---

**This pipeline learns a physically structured, stable, and interpretable model for chaotic Lorenz dynamics using Hamiltonian neural networks and symplectic integration.**
