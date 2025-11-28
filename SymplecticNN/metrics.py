# MSE loss, Energy drift (H(x_t) - H(x_0)), long-term trajectory divergence, Lyapunov-like metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstructionLoss(x_pred: torch.Tensor, x_true: torch.Tensor):

    """
    Mean Squared Error between original and reconstructed data.
    Args:
        x (torch.Tensor): Original data of shape (batch_size, dim).
        x_hat (torch.Tensor): Reconstructed data of shape (batch_size, dim).
    Returns:
        torch.Tensor: Scalar tensor representing the MSE loss.
    """
    
    return F.mse_loss(x_pred, x_true)

def latentConsistencyLoss(z_next_pred, z_next_true):

    """
    Optional: If you have the true canonical coordinates.
    Usually for Lorenz, we do NOT — so this stays unused unless needed.
    """

    return F.mse_loss(z_next_pred, z_next_true)

def energyDrift(hamiltonianNN, z0: torch.Tensor, zT: torch.Tensor):

    """
    Computes |H(z_T) - H(z_0)| for a batch.
    Returns mean drift over the batch.
    """

    with torch.no_grad():
        H0 = hamiltonianNN(z0).squeeze()  
        HT = hamiltonianNN(zT).squeeze()  

    drift = torch.abs(HT - H0)
    return drift.mean()

def rolloutMSE(model, x0: torch.Tensor, steps: int = 100):

    """
    Computes average MSE over a rollout of `steps` predictions.

    Args:
        model: SymplecticNN (encoder → symplectic → decoder)
        x0: (batch, 3) initial states
        steps: number of forward predictions to evaluate

    Returns:
        mean-squared error over all steps
    """

    model.eval()
    mse_sum = 0.0
    x = x0.clone()

    for step in range(steps):
        x_pred = model(x)
        mse_sum += F.mse_loss(x_pred, x, reduction='mean')
        x = x_pred.detach()  # Detach to prevent gradient accumulation

    return mse_sum / steps

def SymplecticLoss(
        x_pred,
        x_true,
        z0,
        zT,
        hamiltonianNN,
        lambda_energy = 0.01
    ):

    """
    Total loss used in many symplectic NN papers:
    L = MSE_recon + λ * EnergyDrift

    Args:
        lambda_energy: weight of energy conservation
    """

    recon = reconstructionLoss(x_pred, x_true)
    drift = energyDrift(hamiltonianNN, z0, zT)

    total = recon + lambda_energy * drift
    return total, recon, drift