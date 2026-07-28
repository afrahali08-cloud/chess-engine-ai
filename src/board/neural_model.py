"""PyTorch model architecture shared by training and runtime inference."""

from __future__ import annotations

import torch
from torch import nn

from .neural_features import TOTAL_FEATURES


class EvalNet(nn.Module):
    def __init__(self, input_size: int = TOTAL_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
