"""
Crafter CNN Feature Extractor
=============================

Architecture:
  Input:  (B, 3, 64, 64) float32 in [0, 1]
  Conv2D: 32 filters, kernel 3, stride 2 → (B, 32, 31, 31)
  ReLU
  Conv2D: 64 filters, kernel 3, stride 2 → (B, 64, 15, 15)
  ReLU
  Conv2D: 64 filters, kernel 3, stride 2 → (B, 64, 7, 7)
  ReLU
  Flatten → (B, 64*7*7) = (B, 3136)
  Linear 512 → (B, 512)

Output: (B, 512) float32
"""

import torch
import torch.nn as nn
from typing import Tuple


def _conv_out_size(size: int, kernel: int, stride: int, padding: int = 0) -> int:
    return (size - kernel + 2 * padding) // stride + 1


class CrafterCNN(nn.Module):
    """
    Convolutional feature extractor for Crafter observations.

    Input shape: (B, 3, 64, 64)
    Output shape: (B, 512)

    Parameters
    ----------
    in_channels : int
        Number of input channels. 3 for single frame, 12 for 4-frame stack.
    out_features : int
        Output feature dimension.
    """

    def __init__(self, in_channels: int = 3, out_features: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features

        self.conv_layers = nn.Sequential(
            # Layer 1: (B, C, 64, 64) → (B, 32, 31, 31)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # Layer 2: (B, 32, 31, 31) → (B, 64, 15, 15)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            # Layer 3: (B, 64, 15, 15) → (B, 64, 7, 7)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size dynamically
        self._flat_size = self._compute_flat_size(in_channels)

        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, out_features),
            nn.ReLU(inplace=True),
        )

        # Initialize weights with orthogonal initialization
        self._init_weights()

    def _compute_flat_size(self, in_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            out = self.conv_layers(dummy)
            return int(out.view(1, -1).shape[1])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, H, W) or (B, T, C, H, W) for sequential input.

        Returns
        -------
        features : torch.Tensor
            Shape (B, out_features) or (B*T, out_features).
        """
        # Handle (B, T, C, H, W) → (B*T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self._forward_flat(x)
            return features.view(B, T, self.out_features)

        return self._forward_flat(x)

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_layers(x)
        flat = conv_out.view(conv_out.size(0), -1)
        return self.fc(flat)

    def get_activation_stats(self) -> dict:
        """Return lightweight activation statistics for visualization."""
        stats = {}
        with torch.no_grad():
            # Get conv layer output norms (cached from last forward pass)
            for i, layer in enumerate(self.conv_layers):
                if isinstance(layer, nn.Conv2d):
                    stats[f"conv{i//2 + 1}_weight_norm"] = float(
                        layer.weight.data.norm().item()
                    )
        return stats

    def get_flat_size(self) -> int:
        return self._flat_size

    def __repr__(self):
        return (
            f"CrafterCNN(in_channels={self.in_channels}, "
            f"flat_size={self._flat_size}, out_features={self.out_features})"
        )

