"""
=============================================================================
PhyLST — Physics-Constrained Land Surface Temperature Model
=============================================================================
Our proposed method. Key innovations:

  1. Energy balance residual: predicts delta = LST - T2m (structural prior)
  2. Heteroscedastic uncertainty: dual-head (mean + log_var), NLL loss
  3. Expanded monotonicity: 5 positive + 1 negative monotonic constraints
  4. Radiation-driven diurnal constraint
  5. Spatial regularization via feature similarity

Architecture: Dual-branch 1D ResNet (static + dynamic), 8 residual blocks,
hidden_dim=512, dropout=0.12.
=============================================================================
"""

import torch
import torch.nn as nn

from .resnet1d import ResBlock1D


class PhyLST(nn.Module):
    """Physics-Constrained ResNet for LST Reconstruction.

    Predicts temperature anomaly (delta = LST - T2m) with uncertainty
    estimation via a dual-head architecture.

    Parameters
    ----------
    input_dim : int
        Total number of input features.
    n_static : int
        Number of static features (first n_static columns of input).
    hidden_dim : int
        Hidden dimension for residual blocks.
    n_blocks : int
        Number of residual blocks.
    dropout : float
        Dropout rate.
    t2m_feat_idx : int
        Column index of t2m_mean in the input tensor.
    t2m_feat_mean : float
        Mean of t2m_mean feature (from StandardScaler).
    t2m_feat_std : float
        Std of t2m_mean feature (from StandardScaler).
    y_mean : float
        Mean of target variable (for denormalization).
    y_std : float
        Std of target variable (for denormalization).
    """

    def __init__(self, input_dim, n_static, hidden_dim=512, n_blocks=8, dropout=0.12,
                 t2m_feat_idx=0, t2m_feat_mean=0.0, t2m_feat_std=1.0,
                 y_mean=0.0, y_std=1.0):
        super().__init__()
        self.n_static = n_static
        self.t2m_feat_idx = t2m_feat_idx
        self.register_buffer("t2m_feat_mean", torch.tensor(t2m_feat_mean, dtype=torch.float32))
        self.register_buffer("t2m_feat_std", torch.tensor(t2m_feat_std, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))

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
        # Mean head: outputs delta (temperature anomaly in °C)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Uncertainty head: outputs log-variance
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """Forward pass.

        Returns
        -------
        pred_norm : Tensor (batch,)
            Predicted LST in normalized space (z-score).
        log_var : Tensor (batch,)
            Log-variance for heteroscedastic uncertainty.
        """
        x_static = x[:, :self.n_static]
        x_dynamic = x[:, self.n_static:]
        h_s = self.static_proj(x_static)
        h_d = self.dynamic_proj(x_dynamic)
        h = torch.cat([h_s, h_d], dim=-1)
        h = self.blocks(h)

        # Energy balance residual: LST = T2m + delta
        t2m_standardized = x[:, self.t2m_feat_idx]
        t2m_celsius = t2m_standardized * self.t2m_feat_std + self.t2m_feat_mean
        delta = self.head(h).squeeze(-1)
        lst_celsius = t2m_celsius + delta
        pred_norm = (lst_celsius - self.y_mean) / self.y_std

        # Uncertainty
        log_var = self.uncertainty_head(h).squeeze(-1)
        log_var = torch.clamp(log_var, -6.0, 6.0)

        return pred_norm, log_var
