import sys, os

# Add project root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import numpy as np
from pathlib import Path

from SymplecticNN.model_symp import SymplecticNN
from SymplecticNN.plots import (
    plot_lorenz_plotly,
    plot_phase_portraits,
    plot_energy_drift
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configs
# ============================================================

CHECKPOINT = "checkpoints/symplecticnn_epoch80.pt"
DATA_PATH = "lorenzSystemSimulator/data/lorenz_trajectories500.npy"

ROLL_STEPS = 500


# ============================================================
# Rollout the SymplecticNN
# ============================================================

def rollout(model, x0, steps, mean, std):
    device = next(model.parameters()).device
    
    # Normalize starting point
    x0_norm = (x0 - mean) / std

    # Encode initial state → z0
    z = model.encoder(x0_norm.unsqueeze(0))   # (1,8)
    z = z.clone().detach().requires_grad_(True)

    traj = []
    energies = []

    H0 = model.symplectic.hamiltonianNN(z).detach()

    for _ in range(steps):
        # One symplectic leapfrog step
        z = model.symplectic(z).clone().detach().requires_grad_(True)   # keep (1,8)

        # Decode → normalized x̂
        x_pred_norm = model.decoder(z)           # (1,3)
        
        # Undo normalization
        x_pred = x_pred_norm * std + mean
        traj.append(x_pred.squeeze().detach().cpu().numpy())   # ensures shape is (3,)

        # Track energy drift (scalar)
        Ht = model.symplectic.hamiltonianNN(z).detach()
        energies.append(float(torch.abs(Ht - H0).cpu()))

    return np.array(traj), np.array(energies)






# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Load dataset example trajectory
    # --------------------------------------------------------
    data = np.load(DATA_PATH)
    true_traj = data[0]          # (T,3)

    x0_np = true_traj[0]
    x0 = torch.tensor(x0_np, dtype=torch.float32).to(DEVICE)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = SymplecticNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print("[✓] Loaded model checkpoint.")

    # --------------------------------------------------------
    # Recompute normalization (same as training)
    # --------------------------------------------------------
    flat = data.reshape(-1, 3)
    mean = torch.tensor(flat.mean(axis=0), dtype=torch.float32).to(DEVICE)
    std  = torch.tensor(flat.std(axis=0) + 1e-8, dtype=torch.float32).to(DEVICE)

    # --------------------------------------------------------
    # Run rollout
    # --------------------------------------------------------
    pred_traj, energies = rollout(model, x0, ROLL_STEPS, mean, std)
    print("[✓] Rollout completed.")

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------
    plot_lorenz_plotly(true_traj[:ROLL_STEPS], pred_traj, title="SymplecticNN Lorenz Rollout")
    plot_phase_portraits(true_traj[:ROLL_STEPS], pred_traj)
    plot_energy_drift(energies, savePath="checkpoints/energy_drift.png")

    print("[✓] All plots complete.")



if __name__ == "__main__":
    main()
