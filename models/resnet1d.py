"""
=============================================================================
ResNet1D — 1D residual network baseline for LST reconstruction
=============================================================================
"""

import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """1D residual block with two linear layers, BatchNorm, GELU, Dropout."""

    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.act(self.dropout(self.block(x) + x))


class ResNet1D(nn.Module):
    """Dual-branch 1D ResNet baseline.

    Static and dynamic features are projected separately, then fused
    through residual blocks. Standard baseline — no Mixup, no SWA.
    """

    def __init__(self, input_dim, n_static, hidden_dim=256, n_blocks=4, dropout=0.15):
        super().__init__()
        self.n_static = n_static
        n_dynamic = input_dim - n_static
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
        )
        self.dynamic_proj = nn.Sequential(
            nn.Linear(n_dynamic, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResBlock1D(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_static = x[:, :self.n_static]
        x_dynamic = x[:, self.n_static:]
        h_s = self.static_proj(x_static)
        h_d = self.dynamic_proj(x_dynamic)
        h = torch.cat([h_s, h_d], dim=-1)
        h = self.blocks(h)
        return self.head(h).squeeze(-1) + self.output_bias
