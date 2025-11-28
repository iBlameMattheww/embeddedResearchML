# Scalar output network H(x)
import torch
import torch.nn as nn

class HamiltonianNN(nn.Module):
    def __init__(self, dim: int, hiddenSize: int = 64, numHiddenLayers: int = 2):
        super().__init__()

        layers = []
        inDim = dim

        for i in range(numHiddenLayers):
            layers.append(nn.Linear(inDim, hiddenSize))
            layers.append(nn.Tanh())
            inDim = hiddenSize

        layers.append(nn.Linear(inDim, 1))
        self.net = nn.Sequential(*layers)

        self.__init__weights() # initialize weights (might tweak later)

    def __init__weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Forward pass through the Hamiltonian network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) representing H(x).
        """
        return self.net(x)