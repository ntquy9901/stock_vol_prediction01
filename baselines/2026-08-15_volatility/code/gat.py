# code/gat.py
"""Self-written multi-head Graph Attention layer (Velickovic-style), masked by adjacency."""
from __future__ import annotations
import torch
from torch import nn


class GATLayer(nn.Module):
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
        # h [B,N,in], adjacency [B,N,N] (>0 where edge j->i allowed; diagonal kept by caller)
        b, n, _ = h.shape
        wh = self.W(h).view(b, n, self.heads, self.out_dim)          # [B,N,H,O]
        e_src = (wh * self.a_src).sum(-1)                            # [B,N,H]
        e_dst = (wh * self.a_dst).sum(-1)                            # [B,N,H]
        e = self.leaky(e_dst.unsqueeze(2) + e_src.unsqueeze(1))      # [B,N(dst i),N(src j),H]
        mask = (adjacency > 0).unsqueeze(-1)                         # [B,N,N,1]
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)                             # over source j
        alpha = torch.nan_to_num(alpha, nan=0.0)                    # isolated node -> all -inf row
        out = torch.einsum("bijh,bjho->biho", alpha, wh)           # [B,N,H,O]
        return torch.nn.functional.elu(out.reshape(b, n, self.heads * self.out_dim))
