"""
=============================================================================
MLP Model — standard multilayer perceptron baseline
=============================================================================
"""

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """Standard MLP baseline: 2 hidden layers with BatchNorm, GELU, Dropout.

    This is a standard baseline configuration commonly used in remote sensing
    literature — no Mixup, no SWA, no sample weighting.
    """

    def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.net(x).squeeze(-1) + self.output_bias
