"""MASTER (Market-Guided Stock Transformer) nn.Module -- VENDORED + ADAPTED.

Source: Li, Wang, Yang, Luo, Qu, Wang, Zhang, Zhou, "MASTER: Market-Guided Stock Transformer for
Stock Price Forecasting", AAAI 2024, arXiv:2312.15235. Reference implementation:
https://github.com/SJTU-DMTai/MASTER (MIT License, (c) 2025 Data Management Technology and AI;
see ``_master_ref/LICENSE``). The five sub-modules (PositionalEncoding, TAttention, SAttention, Gate,
TemporalAttention) and the MASTER forward are taken from ``_master_ref/master.py``.

ADAPTATION (documented for honesty): the reference modules process ONE cross-section ``[N, T, D]`` per
forward (N stocks as the effective batch for the attention). To use the RTX 4060 well we add a leading
anchor-batch dimension ``B`` and rewrite the three attention modules with explicit batched axes. The
math is IDENTICAL per anchor -- no cross-anchor mixing:
  * TAttention  batches over ``(B, N)`` and attends over T (intra-stock temporal);
  * SAttention  permutes to ``[B, T, N, dim]`` and attends over N within each ``(B, T)`` (dense
    inter-stock, softmax over ALL stocks -- the contrast to VolGA's sparse Top-5 edge);
  * TemporalAttention aggregates over T per ``(B, N)``.
A ``B=1`` batched forward equals the per-anchor reference (pinned by test_batched_b1_matches_reference).
The reference qlib ``SequenceModel`` training loop is NOT vendored; training mirrors the project's proven
``train_masked_rich`` loop (per-node z-score target, linear output, inverse-transform, positivity floor).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added over the time axis (reference master.py)."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [B, N, T, d_model]
        t = x.shape[-2]
        return x + self.pe[:t]                               # [T, d_model] broadcasts over (B, N)


class TAttention(nn.Module):
    """Intra-stock temporal multi-head self-attention over T (reference master.py, batched over (B,N))."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)]) if dropout > 0 else None
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(p=dropout),
                                 nn.Linear(d_model, d_model), nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x [B, N, T, d_model]
        x = self.norm1(x)
        q, k, v = self.qtrans(x), self.ktrans(x), self.vtrans(x)
        dim = self.d_model // self.nhead
        outs = []
        for i in range(self.nhead):
            sl = slice(i * dim, None) if i == self.nhead - 1 else slice(i * dim, (i + 1) * dim)
            qh, kh, vh = q[..., sl], k[..., sl], v[..., sl]   # [B, N, T, dim]
            att = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)), dim=-1)   # [B, N, T, T]
            if self.attn_dropout is not None:
                att = self.attn_dropout[i](att)
            outs.append(torch.matmul(att, vh))               # [B, N, T, dim]
        att_output = torch.cat(outs, dim=-1)
        xt = self.norm2(x + att_output)
        return xt + self.ffn(xt)


class SAttention(nn.Module):
    """Dense inter-stock multi-head self-attention over N at each timestep (reference master.py).

    Batched over ``(B, T)``: every stock attends to EVERY other stock in the same anchor's cross-section
    (softmax over all N sources) -- the dense-attention contrast to VolGA's sparse Top-5 vol->PK edge.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.temperature = math.sqrt(d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(p=dropout),
                                 nn.Linear(d_model, d_model), nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x [B, N, T, d_model]; key_mask [B, N] (1=valid stock, 0=invalid/non-trading -> excluded as a key).
        x = self.norm1(x)
        # move stocks to the second-to-last axis so attention runs over N within each (B, T)
        q = self.qtrans(x).permute(0, 2, 1, 3)               # [B, T, N, d_model]
        k = self.ktrans(x).permute(0, 2, 1, 3)
        v = self.vtrans(x).permute(0, 2, 1, 3)
        dim = self.d_model // self.nhead
        # invalid KEY stocks get -inf logits so valid stocks never attend to zero-padded non-trading stocks
        neg = None if key_mask is None else (key_mask < 0.5).view(key_mask.shape[0], 1, 1, key_mask.shape[1])
        outs = []
        for i in range(self.nhead):
            sl = slice(i * dim, None) if i == self.nhead - 1 else slice(i * dim, (i + 1) * dim)
            qh, kh, vh = q[..., sl], k[..., sl], v[..., sl]   # [B, T, N, dim]
            logit = torch.matmul(qh, kh.transpose(-1, -2)) / self.temperature   # [B, T, N, N]
            if neg is not None:
                logit = logit.masked_fill(neg, float("-inf"))
            att = torch.softmax(logit, dim=-1)
            att = torch.nan_to_num(att, nan=0.0)             # invalid TARGET rows (all-masked) -> 0 (discarded)
            att = self.attn_dropout[i](att)
            outs.append(torch.matmul(att, vh))               # [B, T, N, dim]
        att_output = torch.cat(outs, dim=-1).permute(0, 2, 1, 3)   # back to [B, N, T, d_model]
        xt = self.norm2(x + att_output)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    """Market-guided feature gate: market features -> soft per-feature weight (reference master.py)."""

    def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        output = torch.softmax(self.trans(gate_input) / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    """Aggregate the T timesteps into one vector per stock, query = the last step (reference master.py)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:      # z [B, N, T, d_model]
        h = self.trans(z)
        query = h[:, :, -1, :].unsqueeze(2)                  # [B, N, 1, d_model]
        lam = (h * query).sum(-1)                            # [B, N, T]
        lam = torch.softmax(lam, dim=-1).unsqueeze(-1)       # [B, N, T, 1]
        return (lam * z).sum(2)                              # [B, N, d_model]


class MASTER(nn.Module):
    """Batched MASTER: input ``[B, N, T, d_feat + d_gate]`` -> ``[B, N]`` per-stock forecast.

    ``gate_input_start_index`` splits stock features (``[..., :start]``) from the appended market
    features (``[..., start:end]``); the gate reads the market features at the LAST timestep.
    """

    def __init__(self, d_feat: int, d_model: int, t_nhead: int, s_nhead: int, t_dropout: float,
                 s_dropout: float, gate_input_start_index: int, gate_input_end_index: int, beta: float,
                 max_len: int = 100):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.feature_gate = Gate(gate_input_end_index - gate_input_start_index, d_feat, beta=beta)
        # named submodules (the reference wraps these in nn.Sequential; kept explicit so the dense
        # inter-stock SAttention can receive the per-anchor stock-validity key mask)
        self.proj = nn.Linear(d_feat, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.tattn = TAttention(d_model=d_model, nhead=t_nhead, dropout=t_dropout)
        self.sattn = SAttention(d_model=d_model, nhead=s_nhead, dropout=s_dropout)
        self.temporal = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x [B, N, T, d_feat + d_gate]; key_mask [B, N] (1=valid stock)
        src = x[:, :, :, :self.gate_input_start_index]                       # [B, N, T, d_feat]
        gate_input = x[:, :, -1, self.gate_input_start_index:self.gate_input_end_index]   # [B, N, d_gate]
        src = src * self.feature_gate(gate_input).unsqueeze(2)               # gate broadcasts over T
        h = self.pos(self.proj(src))
        h = self.tattn(h)
        h = self.sattn(h, key_mask)
        h = self.temporal(h)
        return self.decoder(h).squeeze(-1)                                   # [B, N]


def build_master(cfg) -> MASTER:
    """Construct a MASTER from a MasterConfig (single-source hyperparameters)."""
    return MASTER(d_feat=cfg.n_stock_feat, d_model=cfg.d_model, t_nhead=cfg.t_nhead, s_nhead=cfg.s_nhead,
                  t_dropout=cfg.dropout, s_dropout=cfg.dropout, gate_input_start_index=cfg.n_stock_feat,
                  gate_input_end_index=cfg.n_stock_feat + cfg.n_market_feat, beta=cfg.beta,
                  max_len=cfg.max_len)
