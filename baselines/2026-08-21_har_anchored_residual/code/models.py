"""Residual expert net + HAR-anchored reconstruction for E5-E8 (plan sections 10, 11, 25).

``ResidualNet`` maps per-date snapshots [B, N, lookback, 3] + adjacency -> a per-node correction signal
``c`` [B, N]. The final linear layer is zero-initialized so ``c == 0`` at initialization, giving an exact
HAR fallback under both reconstructions. Branch toggles select the expert:
  use_lstm=True,  use_graph=False -> E5 (ticker-local temporal only)
  use_lstm=False, use_graph=True  -> E6 (cross-sectional graph only)
  use_lstm=True,  use_graph=True  -> E7 / E8 (both)
The GAT branch reuses the tested ``GATLayer`` from the frozen submission (masked on adjacency != 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import model as _submodel  # noqa: E402  (reuse GATLayer)


class ResidualNet(nn.Module):
    def __init__(self, price_dim: int = 3, hidden: int = 64, heads: int = 4, dropout: float = 0.2,
                 use_lstm: bool = True, use_graph: bool = True) -> None:
        super().__init__()
        if not (use_lstm or use_graph):
            raise ValueError("ResidualNet needs at least one branch")
        self.use_lstm, self.use_graph, self.hidden = use_lstm, use_graph, hidden
        feat_dim = 0
        if use_lstm:
            self.lstm = nn.LSTM(price_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
            feat_dim += hidden
        if use_graph:
            self.gat = _submodel.GATLayer(price_dim, hidden, heads)
            feat_dim += hidden * heads
        self.head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)     # zero-init final layer -> c == 0 at init (HAR fallback)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, price: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, n, seq, d = price.shape
        parts = []
        if self.use_lstm:
            out, _ = self.lstm(price.reshape(b * n, seq, d))
            parts.append(out[:, -1].reshape(b, n, self.hidden))
        if self.use_graph:
            adj = adjacency if adjacency.dim() == 3 else adjacency.unsqueeze(0).expand(b, n, n)
            parts.append(self.gat(price[:, :, -1, :], adj))
        h = torch.cat(parts, dim=-1)
        return self.head(h).squeeze(-1)          # [B, N]


def additive_pred(har_raw, c, res_scale: float, lam: float = 1.0):
    """E5-E7: y_hat = har + lam * res_scale * c. c==0 -> exact HAR fallback. Works for numpy or torch."""
    return har_raw + lam * res_scale * c


def multiplicative_pred(har_raw, c, eps: float = 1e-8, lam: float = 1.0, res_scale=1.0):
    """E8: y_hat = (har + eps) * exp(lam * res_scale * c). c==0 -> HAR+eps; always positive.

    ``res_scale`` (scalar or per-node array) rescales the network's normalized log-correction back to the
    log-residual scale, mirroring ``additive_pred``. numpy or torch."""
    exp = torch.exp if isinstance(c, torch.Tensor) else np.exp
    return (har_raw + eps) * exp(lam * res_scale * c)
