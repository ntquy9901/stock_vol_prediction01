"""
CryptoMamba Model for Volatility Prediction

Simplified implementation adapted from:
https://github.com/MShahabSepehri/CryptoMamba

Key changes:
- Simplified architecture (removed complex modular structure)
- Adapted for single-target volatility prediction
- Compatible with project's existing evaluation framework
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MambaBlock(nn.Module):
    """Core Mamba block with selective state space."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 1D causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.act = nn.SiLU()

        # X projection (for dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        # Delta projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # State matrix A (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_log = torch.log(A).unsqueeze(0)  # (1, d_state)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x):
        """
        x: (B, L, D) where B=batch, L=length, D=features

        Returns: (B, L, D) - same shape
        """
        batch, seqlen, dim = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Causal convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :seqlen]  # (B, d_inner, L)
        x = self.act(x)
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # X projection for dt, B, C
        x_db = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Delta projection
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = dt.transpose(1, 2)  # (B, d_inner, L)

        # State matrix
        A = -torch.exp(self.A_log.float())  # (d_state,)

        # Simplified SSM computation (selective scan would go here)
        # For simplicity, using parameterized linear transformation
        # In full implementation, this would use selective_scan_fn

        # Reshape for computation
        B = B.transpose(1, 2)  # (B, d_state, L)
        C = C.transpose(1, 2)  # (B, d_state, L)

        # Simplified SSM dynamics (y = C * state)
        y = torch.einsum('bdl,bsl->bdl', C, B)  # (B, d_inner, L)
        y = y + self.D * x.unsqueeze(-1)  # Skip connection
        y = y * torch.sigmoid(z)  # Gating

        # Output projection
        y = y.transpose(1, 2)  # (B, L, d_inner)
        out = self.out_proj(y)  # (B, L, d_model)

        return out


class CryptoMambaVolatility(nn.Module):
    """
    CryptoMamba adapted for volatility prediction.

    Architecture:
    - Input: (batch, seq_length, features)
    - Multiple Mamba blocks for temporal modeling
    - Prediction head for single volatility value

    Simplified from original CryptoMamba for clean baseline implementation.
    """

    def __init__(
        self,
        num_features: int = 5,  # OHLCV
        hidden_dim: int = 64,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        seq_length: int = 22,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_length, num_features)

        Returns:
            (batch, 1) - predicted volatility
        """
        # Input projection
        x = self.input_proj(x)  # (B, L, hidden_dim)

        # Apply Mamba blocks
        for block in self.mamba_blocks:
            x = block(x) + x  # Residual connection

        # Normalize and take last timestep
        x = self.norm(x)  # (B, L, hidden_dim)
        x = x[:, -1, :]  # (B, hidden_dim) - last timestep

        # Predict
        out = self.pred_head(x)  # (B, 1)

        return out


def create_cryptomamba_model(
    num_features: int = 5,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    seq_length: int = 22,
) -> CryptoMambaVolatility:
    """
    Factory function to create CryptoMamba model.

    Uses simplified hyperparameters based on CryptoMamba paper:
    - hidden_dim: 64 (smaller than 128 for faster training)
    - num_layers: 3 (balance between capacity and overfitting)
    - d_state: 16 (default from paper)
    - seq_length: 22 (HAR monthly window)
    """
    model = CryptoMambaVolatility(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=dropout,
        seq_length=seq_length,
    )

    return model
