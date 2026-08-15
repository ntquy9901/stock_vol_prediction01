# code/model.py
"""Track-A-style GNN: LSTM temporal + real multi-head GAT (concat branch) + news + per-ticker gate."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat import GATLayer  # noqa: E402

POSITIVITY_EPSILON = 1e-6


class TrackAGatModel(nn.Module):
    def __init__(self, price_dim: int, news_dim: int, num_tickers: int,
                 hidden: int = 64, heads: int = 4, dropout: float = 0.2,
                 use_news: bool = True, use_gate: bool = True):
        super().__init__()
        self.use_news, self.use_gate, self.hidden = use_news, use_gate, hidden
        self.price_lstm = nn.LSTM(price_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.news_proj = nn.Linear(news_dim, hidden)
        self.news_lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.gate_logits = nn.Parameter(torch.zeros(num_tickers))
        self.gat1 = GATLayer(hidden, hidden, heads)          # 64 -> 256
        self.gat2 = GATLayer(hidden * heads, hidden, heads)  # 256 -> 256
        gnn_dim = hidden * heads
        self.head = nn.Sequential(
            nn.Linear(hidden + gnn_dim + hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("scaler_mean", torch.zeros(num_tickers))
        self.register_buffer("scaler_std", torch.ones(num_tickers))

    def configure_positivity(self, mean: torch.Tensor, std: torch.Tensor) -> "TrackAGatModel":
        self.scaler_mean.copy_(mean)
        self.scaler_std.copy_(std)
        return self

    def _encode_seq(self, lstm: nn.LSTM, x, proj=None):
        b, n, seq, d = x.shape
        flat = x.reshape(b * n, seq, d)
        if proj is not None:
            flat = torch.relu(proj(flat))          # news branch: Linear -> ReLU (per ARCHITECTURE)
        out, _ = lstm(flat)
        return out[:, -1].reshape(b, n, -1)                 # last hidden [B,N,hidden]

    def forward(self, price, news, news_mask, ticker_ids, adjacency, apply_graph: bool = True):
        h_lstm = self._encode_seq(self.price_lstm, price)                          # [B,N,64]
        b, n, _ = h_lstm.shape
        if self.use_news:
            # zero out news on no-news timesteps (causal mask) before the news encoder
            news_masked = news * news_mask.unsqueeze(-1)
            news_hidden = self._encode_seq(self.news_lstm, news_masked, proj=self.news_proj)
            gate = (torch.sigmoid(self.gate_logits[ticker_ids]).unsqueeze(-1)
                    if self.use_gate else 1.0)
            gated_news = gate * news_hidden                                        # [B,N,64]
        else:
            gated_news = torch.zeros(b, n, self.hidden, device=h_lstm.device)
        adj = adjacency if apply_graph else torch.eye(n, device=h_lstm.device).unsqueeze(0).expand(b, n, n)
        h_gnn = self.gat2(self.gat1(h_lstm, adj), adj)                             # [B,N,256]
        h = torch.cat([h_lstm, h_gnn, gated_news], dim=-1)                        # [B,N,384]
        raw = self.head(h).squeeze(-1)
        mean = self.scaler_mean[ticker_ids]
        std = self.scaler_std[ticker_ids]
        denorm = raw * std + mean
        eps = POSITIVITY_EPSILON
        floored = eps * torch.nn.functional.softplus(denorm / eps) + eps
        return (floored - mean) / std
