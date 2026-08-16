"""Price-only graph model for the EDA-recommended GNN ablation.

A 2-layer price LSTM encoder shared across nodes, an optional GAT-style residual message-passing
step over the (directed vol->PK) adjacency, a shared prediction head, and the pilot's per-ticker
denormalized-scale positivity floor. Trained end-to-end (no news branch, no frozen-encoder cache):
the EDA config is price-only and the graph is small, so end-to-end is simpler than the pilot's
frozen-backbone path while preserving the scaler/positivity contract.

``use_gnn=False`` (or ``apply_message_passing=False`` at call time) reads out the identical model
with the graph residual removed -- the nested "remove the graph" control.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.functional import softplus

from models import POSITIVITY_EPSILON, _ResidualMessagePassing


class PriceGraphModel(nn.Module):
    """Shared price encoder + optional residual message passing + head + positivity floor."""

    def __init__(
        self, price_dim: int, num_tickers: int, use_gnn: bool, hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_tickers < 1:
            raise ValueError("num_tickers must be positive")
        self.use_gnn = use_gnn
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.message_passing = _ResidualMessagePassing(hidden_dim) if use_gnn else None
        self.register_buffer("target_mean", torch.zeros(num_tickers))
        self.register_buffer("target_std", torch.ones(num_tickers))
        self._positivity_configured = False
        self._positivity_epsilon = POSITIVITY_EPSILON

    def configure_positivity(self, store: Any, epsilon: float = POSITIVITY_EPSILON) -> "PriceGraphModel":
        """Install each ticker's train-fitted target mean/std (same values the store inverts with)."""

        if epsilon <= 0:
            raise ValueError("positivity epsilon must be positive")
        mean = self.target_mean.clone()
        std = self.target_std.clone()
        for ticker_id in range(mean.numel()):
            preprocessor = store.preprocessors.get(ticker_id)
            if preprocessor is None:
                raise ValueError(f"positivity config missing preprocessor for ticker_id {ticker_id}")
            scaler = preprocessor.target_scaler
            mean[ticker_id] = float(scaler.mean[0])
            std[ticker_id] = float(scaler.std[0])
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)
        self._positivity_epsilon = float(epsilon)
        self._positivity_configured = True
        return self

    def _apply_positivity(self, output: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        mean = self.target_mean[ticker_ids].reshape(output.shape)
        std = self.target_std[ticker_ids].reshape(output.shape)
        raw = output * std + mean
        epsilon = self._positivity_epsilon
        raw_positive = epsilon * softplus(raw / epsilon) + epsilon
        return (raw_positive - mean) / std

    def encode(self, x_price: torch.Tensor) -> torch.Tensor:
        """Encode ``[batch, nodes, time, features]`` price windows to ``[batch, nodes, hidden]``."""

        if x_price.ndim != 4:
            raise ValueError("x_price must be [batch, nodes, time, features]")
        batch_size, node_count = x_price.shape[:2]
        flat = x_price.reshape(batch_size * node_count, *x_price.shape[2:])
        _, (hidden, _) = self.price_lstm(flat)
        return hidden[-1].reshape(batch_size, node_count, -1)

    def forward(
        self, x_price: torch.Tensor, adjacency: torch.Tensor, ticker_ids: torch.Tensor,
        presence_mask: torch.Tensor | None = None, apply_message_passing: bool = True,
    ) -> torch.Tensor:
        base = self.encode(x_price)
        if self.message_passing is not None and apply_message_passing:
            base = base + self.message_passing(base, adjacency, presence_mask)
        output = self.head(base).squeeze(-1)
        if self._positivity_configured:
            output = self._apply_positivity(output, ticker_ids.reshape(output.shape).to(torch.long))
        return output
