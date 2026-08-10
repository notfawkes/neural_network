"""
MLP Layers
==========
Simple multi-layer perceptron utility.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    """
    General-purpose MLP with optional activation on the output.

    Parameters
    ----------
    input_dim : int
    hidden_dims : list of int
    output_dim : int
    activation : nn.Module class
    output_activation : nn.Module or None
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation=nn.ReLU,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation(inplace=True) if hasattr(activation(), 'inplace') else activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

