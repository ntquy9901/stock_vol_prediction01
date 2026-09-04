"""From-scratch PatchTST temporal encoder + PatchTSTRichNet (PatchTST temporal branch in PARALLEL
with the existing vol->PK weighted-GAT spatial branch).

PatchTST reference: Nie, Nguyen, Sinthong, Kalagnanam (2023), "A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers", ICLR 2023 (arXiv:2211.14730). Core ideas used here:
(1) **patching** the lookback series into (possibly overlapping) subseries patches; (2)
**channel-independence** — the same patch-embed + transformer weights process every feature channel
and every node, and channels do not attend to each other in the backbone; (3) a vanilla
**TransformerEncoder** over the patch tokens.

Adaptation (documented, design.md §3): vanilla PatchTST keeps channels independent through the head;
here the temporal branch must emit ONE per-node embedding fusing all 5 features (the role the LSTM
branch played), so the FINAL linear projection mixes the 5 channel embeddings. The backbone stays
channel-independent; only the projection mixes channels.

This module edits NO shared file. It reuses ``WeightedGATLayer`` (the exact VolGA GAT) by import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "submission" / "soict_lstm_gat",
           _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           _REPO / "scripts" / "quality_gate"):
    sys.path.insert(0, str(_p))

import masked_rich as MR       # noqa: E402  (N_FEAT, read-only)
from run_masked_rich import WeightedGATLayer  # noqa: E402  (the exact VolGA GAT, reused unchanged)

from patchtst_config import PatchTSTHParams  # noqa: E402


class PatchTSTEncoder(nn.Module):
    """Channel-independent PatchTST encoder: [B, N, seq, D] -> [B, N, out_dim].

    Weight sharing across nodes AND channels (they are folded into the batch dim), so the transformer
    sees ``B*N*D`` patch-token sequences per forward. ``num_patches = floor((seq-patch_len)/stride)+1``.
    """

    def __init__(self, seq_len: int, n_feat: int, out_dim: int, hp: PatchTSTHParams,
                 dropout: float = 0.2):
        super().__init__()
        if seq_len < hp.patch_len:
            raise ValueError(f"seq_len ({seq_len}) must be >= patch_len ({hp.patch_len})")
        self.n_feat = n_feat
        self.patch_len = hp.patch_len
        self.stride = hp.stride
        self.pool = hp.pool
        self.num_patches = (seq_len - hp.patch_len) // hp.stride + 1
        self.patch_embed = nn.Linear(hp.patch_len, hp.d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, hp.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model, nhead=hp.n_heads, dim_feedforward=hp.ff_dim,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=hp.depth)
        pool_dim = (self.num_patches * hp.d_model) if hp.pool == "flatten" else hp.d_model
        self.proj = nn.Linear(n_feat * pool_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [B,N,seq,D]
        b, n, seq, d = x.shape
        if d != self.n_feat:
            raise ValueError(f"expected {self.n_feat} channels, got {d}")
        # one univariate series per (node, channel); channels folded into batch -> channel-independent
        z = x.permute(0, 1, 3, 2).reshape(b * n * d, seq)              # [B*N*D, seq]
        patches = z.unfold(-1, self.patch_len, self.stride)           # [B*N*D, P, patch_len]
        h = self.patch_embed(patches) + self.pos                      # [B*N*D, P, d_model]
        h = self.dropout(h)
        h = self.encoder(h)                                          # [B*N*D, P, d_model]
        if self.pool == "mean":
            h = h.mean(dim=1)                                        # [B*N*D, d_model]
        else:
            h = h.reshape(h.shape[0], -1)                            # [B*N*D, P*d_model]
        h = h.reshape(b * n, self.n_feat * h.shape[-1])              # [B*N, D*pool_dim] (channels fused)
        return self.proj(h).reshape(b, n, -1)                        # [B,N,out_dim]


class PatchTSTRichNet(nn.Module):
    """PatchTST temporal branch + optional 2-hop weighted-GAT spatial branch (mirror of MaskedRichNet).

    Identical to ``run_masked_rich.MaskedRichNet`` except the LSTM temporal submodule is replaced by a
    ``PatchTSTEncoder``. The GAT branch, its 2-hop stacking, and the concat head are unchanged (the GAT
    reads RAW node features at day t via ``x[:, :, -1, :]``). ``use_graph=False`` == PatchTST only;
    ``use_graph=True`` == PatchTST + vol->PK wGAT (leave-one-out graph contrast, like VolGA).
    """

    def __init__(self, seq_len: int, hidden: int = 64, heads: int = 4, dropout: float = 0.2,
                 use_graph: bool = True, in_dim: int = MR.N_FEAT, gat_layers: int = 2,
                 hp: PatchTSTHParams | None = None):
        super().__init__()
        hp = hp or PatchTSTHParams()
        self.use_graph, self.hidden, self.gat_layers = use_graph, hidden, gat_layers
        self.patchtst = PatchTSTEncoder(seq_len, in_dim, hidden, hp, dropout=dropout)
        gdim = hidden * heads if use_graph else 0
        if use_graph:
            self.gat1 = WeightedGATLayer(in_dim, hidden, heads)               # in_dim -> hidden*heads
            if gat_layers == 2:
                self.gat2 = WeightedGATLayer(hidden * heads, hidden, heads)   # hidden*heads -> hidden*heads
        self.head = nn.Sequential(nn.Linear(hidden + gdim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _gat(self, node_raw, adj_b):
        out = self.gat1(node_raw, adj_b)
        if self.gat_layers == 2:
            out = self.gat2(out, adj_b)
        return out

    def forward(self, x, adj_b):                    # x [B,N,seq,5]; adj_b [B,N,N] (invalid cols zeroed)
        parts = [self.patchtst(x)]                  # [B,N,hidden]
        if self.use_graph:
            parts.append(self._gat(x[:, :, -1, :], adj_b))
        return self.head(torch.cat(parts, -1)).squeeze(-1)
