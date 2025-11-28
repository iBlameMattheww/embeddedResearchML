# One canonical symplectic layer implementing q += dt * dH/dp, p -= dt * dH/dq
import torch
import torch.nn as nn

class SymplecticLayer(nn.Module):

    """
    Implements a differentiable symplectic integrator (Leapfrog / Stormer-Verlet)
    for latent Hamiltonian dynamics.

    Input:
        z = [q1, q2, q3, q4, p1, p2, p3, p4]   (shape: (batch, 8))

    Output:
        z_next = symplectic_step(z)
    """

    def __init__(self, hamiltonian_nn: nn.Module, dt : float = 0.01):
        super().__init__()
        self.hamiltonianNN = hamiltonian_nn
        self.dt = dt

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
    
    def Forward(self, z):

        """
        Performs one symplectic leapfrog integration step.
        z has shape (batch, 8).
        """

        q, p = torch.chunk(z, 2, dim = 1)
        gradH = self.GradH(z)
        dHdq, dHdp = torch.chunk(gradH, 2, dim = 1)

        pHalf = p - 0.5 * self.dt * dHdq
        qNew = q + self.dt * pHalf

        zHalf = torch.cat([qNew, pHalf], dim = 1)
        gradHNew = self.GradH(zHalf)
        dHdqNew, dHdpNew = torch.chunk(gradHNew, 2, dim = 1)

        pNew = pHalf - 0.5 * self.dt * dHdqNew
        zNew = torch.cat([qNew, pNew], dim = 1)
        return zNew