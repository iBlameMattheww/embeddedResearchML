# ============================================================
# SymplecticNN Training Script (RTX 3070 Ti Machine Only)
# ------------------------------------------------------------
# Features:
#   ✓ Multi-step rollout (numRolloutSteps = 20)
#   ✓ Energy drift penalty
#   ✓ GPU-aware (CUDA)
#   ✓ Checkpoint saving
#   ✓ Loss logging
#   ✓ Compatible with Encoder / SymplecticLayer / Decoder
# ============================================================

import sys
import os

# Add project root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from SymplecticNN.model_symp import SymplecticNN
from SymplecticNN.dataset_loader import get_dataloader

# ============================================================
# Configurations
# ============================================================

DATA_PATH = "lorenzSystemSimulator/data/lorenz_trajectories500.npy"

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

LATENT_DIM = 8
HIDDEN_DIM = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
DT = 0.01

LEAPFROG_STEPS = 1         # how many inner leapfrog steps per "symplectic layer" (usually 1)
ROLLOUT_STEPS = 20         # how far we unroll during training
ENERGY_WEIGHT = 0.1        # multiplier for energy drift penalty

# ============================================================
# Energy Function
# ============================================================

def compute_energy(z, hamiltonian_nn):
    """
    Computes H(z) using the Hamiltonian network.
    Returns shape: (batch,)
    """
    H = hamiltonian_nn(z)           # (batch, 1)
    return H.squeeze(-1)            # (batch,)


# ============================================================
# Training Function
# ============================================================

def train():
    loader = get_dataloader(DATA_PATH, batchSize=BATCH_SIZE)

    model = SymplecticNN(
        inputDim=3,
        latentDim=LATENT_DIM,
        hiddenDim=HIDDEN_DIM,
        numEncoderLayers=ENC_LAYERS,
        numDecoderLayers=DEC_LAYERS,
        dt=DT,
        numLeapfrogSteps=LEAPFROG_STEPS
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    print("[INFO] Starting training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_energy_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

        for x, y in loop:
            x = x.to(DEVICE)     # (batch,3)
            y = y.to(DEVICE)     # true x(t+1)

            optimizer.zero_grad()

            # -------------------------------------------------------
            # Forward step 1: encode to latent canonical variables
            # -------------------------------------------------------
            z = model.encoder(x)                 # (batch, 8)

            # compute H0 for energy drift penalty
            H0 = compute_energy(z, model.symplectic.hamiltonianNN)

            # -------------------------------------------------------
            # Rollout loop: repeatedly apply symplectic layer
            # -------------------------------------------------------
            z_roll = z
            x_roll = None

            for _ in range(ROLLOUT_STEPS):
                z_roll = model.symplectic(z_roll)
                x_roll = model.decoder(z_roll)   # reconstruct

            # reconstruction loss using final rollout
            recon_loss = mse(x_roll, y)

            # -------------------------------------------------------
            # Energy drift penalty
            # -------------------------------------------------------
            Ht = compute_energy(z_roll, model.symplectic.hamiltonianNN)
            energy_loss = mse(Ht, H0)            # encourages H(t)=H(0)

            # -------------------------------------------------------
            # Total combined loss
            # -------------------------------------------------------
            loss = recon_loss + ENERGY_WEIGHT * energy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_energy_loss += energy_loss.item()

            loop.set_postfix({
                "loss": f"{loss.item():.5f}",
                "recon": f"{recon_loss.item():.5f}",
                "energy": f"{energy_loss.item():.5f}"
            })

        # -------------------------------
        # Epoch Summary
        # -------------------------------
        N = len(loader)
        print(f"\nEpoch {epoch}: "
              f"Total={epoch_loss/N:.6f}, "
              f"Recon={epoch_recon_loss/N:.6f}, "
              f"Energy={epoch_energy_loss/N:.6f}")

        if epoch % 10 == 0:
            save_checkpoint(model, epoch)

    save_checkpoint(model, EPOCHS)
    print("\n[✓] Training complete.\n")


# ============================================================
# Checkpoint Saving
# ============================================================

def save_checkpoint(model, epoch):
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"symplecticnn_epoch{epoch}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[✓] Saved checkpoint: {ckpt_path}")


# ============================================================
# Script Entry
# ============================================================

if __name__ == "__main__":
    train()
