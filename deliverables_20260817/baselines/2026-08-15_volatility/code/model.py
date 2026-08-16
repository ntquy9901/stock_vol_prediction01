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


class VolatilityModel(nn.Module):
    def __init__(self, price_dim: int, news_dim: int, num_tickers: int,
                 hidden: int = 64, heads: int = 4, dropout: float = 0.2,
                 use_news: bool = True, use_gate: bool = True, use_graph: bool = True,
                 gat_layers: int = 2):
        super().__init__()
        self.use_news, self.use_gate, self.use_graph, self.hidden = use_news, use_gate, use_graph, hidden
        if gat_layers not in (1, 2):
            raise ValueError(f"gat_layers must be 1 or 2, got {gat_layers}")
        self.gat_layers = gat_layers
        self.price_lstm = nn.LSTM(price_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.news_proj = nn.Linear(news_dim, hidden)
        self.news_lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.gate_logits = nn.Parameter(torch.zeros(num_tickers))
        # GAT consumes RAW node features at t (Track-A parallel branch), NOT h_lstm: matches the
        # vol->PK edge semantics (volume_shock_i(t) -> sqrt(PK_j)(t+1)) and keeps the graph branch
        # an independent cross-sectional view of the LSTM branch. use_graph=False removes the WHOLE
        # graph subsystem (no node/edge/GAT built) -> a clean leave-one-out "no graph" variant.
        # gat_layers=1 keeps ONLY gat1 (1-hop; paper: usually enough) at the same output dim so the
        # head is unchanged; gat_layers=2 stacks gat2 (2-hop, the current default).
        gnn_dim = hidden * heads if use_graph else 0
        if use_graph:
            self.gat1 = GATLayer(price_dim, hidden, heads)       # price_dim -> 256 (1-hop)
            if gat_layers == 2:
                self.gat2 = GATLayer(hidden * heads, hidden, heads)  # 256 -> 256 (2-hop)
        self.head = nn.Sequential(
            nn.Linear(hidden + gnn_dim + hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("scaler_mean", torch.zeros(num_tickers))
        self.register_buffer("scaler_std", torch.ones(num_tickers))

    def configure_positivity(self, mean: torch.Tensor, std: torch.Tensor) -> "VolatilityModel":
        self.scaler_mean.copy_(mean)
        self.scaler_std.copy_(std)
        return self

    def _gat_outputs(self, node_raw, adj):
        """Per-layer GAT embeddings: [gat1_out] for 1-hop, [gat1_out, gat2_out] for 2-hop."""
        outs = [self.gat1(node_raw, adj)]
        if self.gat_layers == 2:
            outs.append(self.gat2(outs[-1], adj))
        return outs

    def gat_layer_outputs(self, price, adjacency):
        """Public graph-branch embeddings per layer (for MAD over-smoothing analysis). Raises if the
        model has no graph branch (use_graph=False)."""
        if not self.use_graph:
            raise ValueError("gat_layer_outputs requires use_graph=True")
        node_raw = price[:, :, -1, :]
        return self._gat_outputs(node_raw, adjacency)

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
        parts = [h_lstm]
        if self.use_graph:
            adj = adjacency if apply_graph else torch.eye(n, device=h_lstm.device).unsqueeze(0).expand(b, n, n)
            node_raw = price[:, :, -1, :]                                          # raw feats at t [B,N,price_dim]
            parts.append(self._gat_outputs(node_raw, adj)[-1])                     # h_gnn [B,N,256]
        parts.append(gated_news)                                                   # [B,N,64] (zeros if no news)
        h = torch.cat(parts, dim=-1)                                               # [B,N,384] or 128 if no graph
        raw = self.head(h).squeeze(-1)
        mean = self.scaler_mean[ticker_ids]
        std = self.scaler_std[ticker_ids]
        denorm = raw * std + mean
        eps = POSITIVITY_EPSILON
        floored = eps * torch.nn.functional.softplus(denorm / eps) + eps
        return (floored - mean) / std
