# Normalalization, batching, dt utilities, printing functions
# utils.py
import os
import torch
import numpy as np
import random
from datetime import datetime


# -------------------------------------------------------
# 1. Set seeds for reproducibility
# -------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
# 2. Select training device (Mac MPS, CUDA, or CPU)
# -------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -------------------------------------------------------
# 3. Save and load model checkpoints
# -------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"model_epoch_{epoch}.pt")

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, filename)

    print(f"[✓] Saved checkpoint: {filename}")


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"[✓] Loaded checkpoint from {filepath}")
    return checkpoint["epoch"]


# -------------------------------------------------------
# 4. Format timestamp for logging
# -------------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------------
# 5. Nice training progress print
# -------------------------------------------------------
def print_epoch_stats(epoch, loss, recon, energy):
    print(
        f"[Epoch {epoch:04d}] "
        f"Loss={loss:.6f} | "
        f"Recon={recon:.6f} | "
        f"EnergyDrift={energy:.6f} | "
        f"Time={timestamp()}"
    )
