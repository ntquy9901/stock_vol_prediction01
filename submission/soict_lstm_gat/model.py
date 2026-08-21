"""HAR-LSTM-GAT: proposed volatility model (temporal LSTM + spatial GAT over a graphical-lasso graph).

Parallel two-branch design (Track-A style):
  * LSTM branch reads the [lookback, 3] HAR-feature sequence per node.
  * GAT branch reads the RAW 3-feature node vector at the last day t and attends over the frozen
    graphical-lasso adjacency (spatial cross-sectional view). Disabled by ``use_graph=False`` -> the
    "LSTM (w/o GAT)" ablation variant.
The two branch outputs are concatenated and passed to a small head. Output is on the normalized scale
(per-ticker inverse-transform is applied at evaluation).
"""
from __future__ import annotations

import torch
from torch import nn


class GATLayer(nn.Module):
    """Multi-head Graph Attention layer (Velickovic-style), masked by adjacency."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        wh = self.W(h).view(b, n, self.heads, self.out_dim)
        e_src = (wh * self.a_src).sum(-1)
        e_dst = (wh * self.a_dst).sum(-1)
        e = self.leaky(e_dst.unsqueeze(2) + e_src.unsqueeze(1))
        # edge EXISTS wherever the (signed) partial-correlation adjacency is nonzero — a negative
        # partial correlation is still a conditional-dependence edge; masking on ">0" (the prior bug)
        # silently dropped every negative edge. Connectivity is sign-agnostic; attention learns weights.
        mask = (adjacency != 0).unsqueeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = torch.einsum("bijh,bjho->biho", alpha, wh)
        return torch.nn.functional.elu(out.reshape(b, n, self.heads * self.out_dim))


class HARLSTMGAT(nn.Module):
    """LSTM temporal branch (+ optional GAT spatial branch) over 3 HAR node features."""

    def __init__(self, price_dim: int = 3, hidden: int = 64, heads: int = 4,
                 dropout: float = 0.2, use_graph: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.hidden = hidden
        self.price_lstm = nn.LSTM(price_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        gnn_dim = hidden * heads if use_graph else 0
        if use_graph:
            self.gat = GATLayer(price_dim, hidden, heads)
        self.head = nn.Sequential(
            nn.Linear(hidden + gnn_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, price: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # price [B,N,seq,3]; adjacency [N,N] or [B,N,N]
        b, n, seq, d = price.shape
        flat = price.reshape(b * n, seq, d)
        out, _ = self.price_lstm(flat)
        h_lstm = out[:, -1].reshape(b, n, self.hidden)        # [B,N,hidden]
        parts = [h_lstm]
        if self.use_graph:
            adj = adjacency if adjacency.dim() == 3 else adjacency.unsqueeze(0).expand(b, n, n)
            node_raw = price[:, :, -1, :]                     # raw 3-feat at day t
            parts.append(self.gat(node_raw, adj))             # [B,N,hidden*heads]
        h = torch.cat(parts, dim=-1)
        return self.head(h).squeeze(-1)                       # [B,N]
