"""Shared sequence models for pooled P1-P3 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PooledPriceLSTM(nn.Module):
    """P1: one price encoder and prediction head shared by pooled samples."""

    def __init__(self, price_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_price: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.price_lstm(x_price)
        return self.head(hidden[-1]).squeeze(-1)


class PooledPriceNewsLSTM(nn.Module):
    """P2/P3: shared price/news encoders with an optional ticker-indexed news gate."""

    def __init__(
        self,
        price_dim: int,
        news_dim: int,
        num_tickers: int,
        use_gate: bool,
        hidden_dim: int = 64,
        news_hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.price_lstm = nn.LSTM(price_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.news_lstm = nn.LSTM(news_dim, news_hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + news_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if use_gate:
            self.gate_logits = nn.Parameter(torch.zeros(num_tickers))
        else:
            self.register_parameter("gate_logits", None)

    def _encode_news(self, x_news: torch.Tensor, news_mask: torch.Tensor) -> torch.Tensor:
        representations = x_news.new_zeros((x_news.shape[0], self.news_lstm.hidden_size))
        valid_indices = news_mask.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return representations

        sequences = [x_news[index, news_mask[index]] for index in valid_indices]
        lengths = torch.tensor([sequence.shape[0] for sequence in sequences], device=x_news.device)
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.news_lstm(packed)
        representations[valid_indices] = hidden[-1]
        return representations

    def forward(
        self,
        x_price: torch.Tensor,
        x_news: torch.Tensor,
        news_mask: torch.Tensor,
        ticker_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, (price_hidden, _) = self.price_lstm(x_price)
        news_hidden = self._encode_news(x_news, news_mask)
        if self.use_gate:
            news_hidden = torch.sigmoid(self.gate_logits[ticker_ids])[:, None] * news_hidden
        return self.head(torch.cat((price_hidden[-1], news_hidden), dim=1)).squeeze(-1)
