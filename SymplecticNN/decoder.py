#  Contains the learned map R^8 -> R^3 (q1, q2, q3, q4, p1, p2, p3, p4) -> (x, y, z)
import torch
import torch.nn as nn

class Decoder(nn.Module):

    """
    Decoder network that transforms (q1, q2, q3, q4, p1, p2, p3, p4) → (x, y, z).
    This learns the inverse nonlinear coordinate transformation from the canonical
    latent space back to the original space.
    """

    def __init__(self, latentDim: int = 8, outputDim: int = 3, hiddenDim = 128, numLayers = 3):
        super().__init__()

        self.latentDim = latentDim
        self.outputDim = outputDim

        layers = []
        inDim = latentDim

        for layer in range(numLayers):
            layers.append(nn.Linear(inDim, hiddenDim))
            layers.append(nn.Tanh())
            inDim = hiddenDim

        layers.append(nn.Linear(inDim, outputDim))
        self.net = nn.Sequential(*layers)
        self.__init__weights()

    def __init__weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latentDim) tensor, interpreted as (q's, p's)
        Returns:
            x: (batch, 3) tensor of (x,y,z)
        """
        return self.net(z)