# One canonical symplectic layer implementing q += dt * dH/dp, p -= dt * dH/dq
import torch
import torch.nn as nn

from .hamiltonian import HamiltonianNN

class SymplecticLayer(nn.Module):

    """
    Implements a differentiable symplectic integrator (Leapfrog / Stormer-Verlet)
    for latent Hamiltonian dynamics.

    Input:
        z = [q1, q2, q3, q4, p1, p2, p3, p4]   (shape: (batch, 8))

    Output:
        z_next = symplectic_step(z)
    """

    def __init__(
            self,
            latentDim: int = 8,
            hiddenDim: int = 128,
            numHiddenLayers: int = 2,
            dt: float = 0.01,
            numSteps: int = 1
    ):
        super().__init__()
        self.latentDim = latentDim
        self.dt = dt
        self.numSteps = numSteps

        self.hamiltonianNN = HamiltonianNN(
            dim = latentDim,
            hiddenSize = hiddenDim,
            numHiddenLayers = numHiddenLayers
        )

    def GradH(self, z):

        """
        Computes gradient of Hamiltonian H(z) with respect to z.
        Returns tensor of shape (batch, 8).
        """

        z = z.clone().detach().requires_grad_(True)
        hVal = self.hamiltonianNN(z)
        grad = torch.autograd.grad(
            hVal.sum(),
            z,
            create_graph=True
        )[0]
        return grad
    
    def forward(self, z):

        """
        Performs one symplectic leapfrog integration step.
        z has shape (batch, 8).
        """

        q, p = torch.chunk(z, 2, dim = 1)
        dt = self.dt

        for step in range(self.numSteps):
            gradH = self.GradH(torch.cat([q, p], dim = 1))
            dHdq, dHdp = torch.chunk(gradH, 2, dim = 1)

            pHalf = p - 0.5 * dt * dHdq
            qNew = q + dt * dHdp
            
            gradHNew = self.GradH(torch.cat([qNew, pHalf], dim = 1))
            dHdqNew, _ = torch.chunk(gradHNew, 2, dim = 1)
            
            pNew = pHalf - 0.5 * dt * dHdqNew
            q, p = qNew, pNew
        
        return torch.cat([q, p], dim = 1)