# Contains the learned map R^3 -> R^8 (x, y, z) -> (q1, q2, q3, q4, p1, p2, p3, p4)
import torch
import torch.nn as nn

class Encoder(nn.Module):

    """
    Encoder network that transforms (x, y, z) → (q1, q2, q3, q4, p1, p2, p3, p4).
    This learns a nonlinear coordinate transformation into a canonical latent space
    suitable for a Hamiltonian + Symplectic integrator.
    """

    def __init__(self, inputDim: int = 3, latentDim: int = 8, hiddenDim = 128, numLayers = 3):
        super().__init__()

        self.inputDim = inputDim
        self.latentDim = latentDim

        layers = []
        inDim = inputDim

        for layer in range(numLayers):
            layers.append(nn.Linear(inDim, hiddenDim))
            layers.append(nn.Tanh())
            inDim = hiddenDim

        layers.append(nn.Linear(inDim, latentDim))
        self.net = nn.Sequential(*layers)
        self.__init__weights()

    def __init__weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3) tensor of (x,y,z)
        Returns:
            latent: (batch, latent_dim) tensor, interpreted as (q's, p's)
        """
        return self.net(x)
