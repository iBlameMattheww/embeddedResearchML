# SymplecticNN class (stacking layers, dt handling)
import torch   
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .symplectic_layer import SymplecticLayer

class SymplecticNN(nn.Module):

    """
    Full Symplectic Neural Network:

        x_t  → Encoder → z_t
        z_t  → SymplecticLayer → z_{t+dt}
        z_{t+dt} → Decoder → x_hat_{t+dt}
    """

    def __init__(
            self,
            inputDim: int = 3,
            latentDim: int = 8,
            hiddenDim: int = 128,
            numEncoderLayers: int = 3,
            numDecoderLayers: int = 3,
            dt: float = 0.01,
            numLeapfrogSteps: int = 1
    ):
        super().__init__()

        self.encoder = Encoder(
            inputDim = inputDim,
            latentDim = latentDim,
            hiddenDim = hiddenDim,
            numLayers = numEncoderLayers
        )

        self.symplectic = SymplecticLayer(
            latentDim = latentDim,
            hiddenDim = hiddenDim,
            numHiddenLayers = 2,
            dt = dt,
            numSteps = numLeapfrogSteps
        )

        self.decoder = Decoder(
            latentDim = latentDim,
            outputDim = inputDim,
            hiddenDim = hiddenDim,
            numLayers = numDecoderLayers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Args:
            x (batch, 3)
        Returns:
            x_hat (batch, 3)
        """

        z = self.encoder(x)
        zNext = self.symplectic(z)
        xHat = self.decoder(zNext)
        return xHat
    
# Rollout steps maybe later??? x(t) → x(t + K·dt)
