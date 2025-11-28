import torch
import numpy as np
from pathlib import Path

from SymplecticNN.model_symp import SymplecticNN
from SymplecticNN.plots import plot_lorenz_plotly, plot_phase_portraits, plot_energy_drift

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Configs
# ============================================================

CHECKPOINT = "checkpoints/symplecticnn_epoch80.pt"
DATA_PATH = "lorenzSystemSimulator/data/lorenz_trajectories500.npy"

ROLL_STEPS = 500

# ============================================================
# Helper: rollout the SymplecticNN
# ============================================================

def rollout(model, x0, steps):
    """
    x0: (1,3) tensor (initial condition)
    steps: number of integration steps
    """
    model.eval()
    xs = [x0.cpu().numpy().squeeze()]

    with torch.no_grad():
        z = model.encoder(x0)

        energies = []
        H0 = model.symplectic.hamiltonianNN(z).item()

        for _ in range(steps):
            z = model.symplectic(z)
            x = model.decoder(z)

            xs.append(x.cpu().numpy().squeeze())

            # energy drift tracking
            Ht = model.symplectic.hamiltonianNN(z).item()
            energies.append(abs(Ht - H0))

    return np.array(xs), np.array(energies)


# ============================================================
# Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Load dataset example trajectory
    # --------------------------------------------------------
    data = np.load(DATA_PATH)
    true_traj = data[0]          # (T,3)

    # pick starting point
    x0_np = true_traj[0]
    x0 = torch.tensor(x0_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = SymplecticNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print("[✓] Loaded model checkpoint.")

    # --------------------------------------------------------
    # Run rollout
    # --------------------------------------------------------
    pred_traj, energies = rollout(model, x0, ROLL_STEPS)
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
