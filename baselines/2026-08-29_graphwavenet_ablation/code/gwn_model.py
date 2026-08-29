"""Faithful Graph WaveNet (Wu, Pan, Long, Jiang, Zhang, IJCAI 2019, arXiv:1906.00121) for the masked
volatility panel, plus its own "w/o adaptive graph" ablation.

Reference implementation: github.com/nnzhan/Graph-WaveNet, ``model.py`` (classes ``nconv``, ``linear``,
``gcn``, ``gwnet``). Verified against the official source, 2026-08-29.

PAPER / OFFICIAL CODE  ->  THIS MODULE
-------------------------------------
* Self-adaptive adjacency (paper Eq. for adaptive adj; official
  ``adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)``):
      A_adp = SoftMax(ReLU(E1 . E2^T))            ->  ``GraphWaveNet.adaptive_adjacency()``
  with learnable node embeddings E1 = ``nodevec1`` in R^{N x c} and E2^T = ``nodevec2`` in R^{c x N}
  (c = ``node_dim``, paper default 10). SoftMax over ``dim=1`` (per source row) exactly as the official code.
* Graph propagation ``nconv`` (official ``einsum('ncvl,vw->ncwl', x, A)``)  ->  ``NConv`` (also accepts a
  BATCHED [B,N,N] A via ``einsum('ncvl,nvw->ncwl')`` so the per-sample node-validity mask can zero invalid
  SOURCE nodes -- the same ``base * nmask`` masked-union-panel convention the sibling ablations use).
      out[..., w, :] = sum_v x[..., v, :] * A[v, w]
* Order-K diffusion ``gcn`` (official concat of [x, nconv(x,a), nconv^2(x,a), ...] then 1x1 ``linear``)
  ->  ``GCN``. ``c_in = (order*support_len + 1) * c_in``; order=2, support_len=1 (adaptive graph only).
* Dilated causal conv + gated activation (official ``filter=tanh(FilterConv(x)); gate=sigmoid(GateConv(x));
  x = filter * gate``), residual + skip connections, WaveNet stacked dilations (1,2,1,2,... reset each
  block)  ->  ``GraphWaveNet.forward``. receptive_field = 1 + blocks*(kernel-1)*(2^layers - 1) = 13 for
  blocks=4, layers=2, kernel=2; when seq < receptive_field the input is LEFT-padded (official behaviour).
* Head: relu(skip) -> relu(EndConv1) -> EndConv2 (official ``end_conv_1``/``end_conv_2``).

DELIBERATE, DOCUMENTED DEVIATIONS (do not change the faithful part):
* out_dim = 1 (single horizon-1 target per node) instead of the traffic 12-step horizon.
* in_dim = 5 (the delivered node-feature vector) instead of the traffic (speed, time-of-day) = 2.
* channel widths reduced from the paper's traffic defaults (skip 256->64, end 512->128) for 8 GB VRAM and
  the smaller daily-vol panel; blocks/layers/dilation/kernel/gating/gcn/adaptive are UNCHANGED.
* The predefined supports (road-network + its transpose) are dropped: on this panel there is no physical
  graph, so ``GWN_adaptive`` uses ONLY the self-adaptive adjacency (the paper's "only adaptive adj"
  configuration). ``GWN_no_adaptive`` then removes that adaptive graph, leaving NO graph conv (a pure TCN).
  NOTE: this is the ablation RELATIVE TO the adaptive-only model, not a bit-identical reproduction of the
  paper's Table-4 "w/o adaptive adj" row (that row keeps the predefined road-graph supports; this panel has
  none). The variable isolated -- the self-adaptive graph -- is identical.
* MASKED-PANEL / BatchNorm caveat: the official ``gwnet`` uses ``BatchNorm2d``, whose statistics pool over
  ``[B, N, T]``. On the masked-union panel invalid nodes are zero-filled (``masked_rich`` zeroes them), so
  BN normalizes the valid nodes using batch statistics that include zero-padded slots. This is inherent to
  applying the paper's BN to a masked panel (kept for fidelity, NOT changed to a mask-aware norm). It is a
  COMMON-MODE effect: both GWN variants carry the same BN, so it cancels in the headline in-family adaptive
  ablation (GWN_adaptive vs GWN_no_adaptive); it does NOT cancel versus the LSTM/HAR baselines (no BN), so
  the cross-family GWN-vs-LSTM/HAR comparison is a backbone comparison and the valid-node fraction is
  reported alongside it to bound the effect.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NConv(nn.Module):
    """Graph propagation x . A. ``A`` may be a fixed [N,N] (official) or a batched [B,N,N] (masked)."""

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:  # x [B,C,N,T]
        if a.dim() == 3:                                   # batched adjacency (per-sample source mask)
            x = torch.einsum("ncvl,nvw->ncwl", x, a)
        else:                                              # fixed [N,N] adjacency (official convention)
            x = torch.einsum("ncvl,vw->ncwl", x, a)
        return x.contiguous()


class Linear1x1(nn.Module):
    """1x1 convolution mixing channels (official ``linear``)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GCN(nn.Module):
    """Order-K diffusion graph conv over a list of supports (official ``gcn``)."""

    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 1, order: int = 2):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear1x1((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    """Graph WaveNet backbone for the masked panel; ``adaptive`` toggles the self-adaptive graph.

    ``adaptive=True``  -> the only spatial graph is the learned self-adaptive adjacency (paper "only
    adaptive adj"): every temporal layer is followed by an order-2 diffusion ``GCN`` on that graph.
    ``adaptive=False`` -> NO graph conv (adaptive graph removed from the adaptive-only model): every layer
    is followed by a 1x1 ``ResidualConv`` instead, so the model is a pure dilated-causal TCN with the
    identical temporal stack.
    Everything else (start conv, gated dilated convs, skip/residual, batchnorm, head) is identical, so the
    adaptive graph is the ONLY variable between the two.
    """

    def __init__(self, n_nodes: int, in_dim: int = 5, out_dim: int = 1, residual_channels: int = 32,
                 dilation_channels: int = 32, skip_channels: int = 64, end_channels: int = 128,
                 kernel_size: int = 2, blocks: int = 4, layers: int = 2, dropout: float = 0.2,
                 adaptive: bool = True, node_dim: int = 10):
        super().__init__()
        self.n_nodes, self.adaptive, self.dropout = n_nodes, adaptive, dropout
        self.blocks, self.layers = blocks, layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        if adaptive:                                        # learnable node embeddings E1, E2 (paper adaptive adj)
            self.nodevec1 = nn.Parameter(torch.randn(n_nodes, node_dim))
            self.nodevec2 = nn.Parameter(torch.randn(node_dim, n_nodes))
        receptive_field = 1
        for _b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _i in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                   (1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                 (1, kernel_size), dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, (1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                if adaptive:
                    self.gconv.append(GCN(dilation_channels, residual_channels, dropout,
                                          support_len=1, order=2))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def adaptive_adjacency(self) -> torch.Tensor:
        """A_adp = SoftMax(ReLU(E1 . E2^T)) [N,N] (official ``softmax(relu(mm(nodevec1, nodevec2)), dim=1)``)."""
        if not self.adaptive:
            raise RuntimeError("adaptive_adjacency() called on a GraphWaveNet built with adaptive=False")
        return F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

    def _masked_adp(self, nmask: torch.Tensor) -> torch.Tensor:
        """[B,N,N] = A_adp with invalid SOURCE nodes (nmask==0) zeroed per sample (masked-union convention)."""
        adp = self.adaptive_adjacency()
        return adp.unsqueeze(0) * nmask.unsqueeze(-1)       # zero rows v where nmask[b,v]==0

    def forward(self, x: torch.Tensor, nmask: torch.Tensor) -> torch.Tensor:  # x [B,N,seq,5]; nmask [B,N]
        x = x.permute(0, 3, 1, 2)                           # -> [B, in_dim, N, seq]  (official layout)
        in_len = x.size(3)
        if in_len < self.receptive_field:                  # left-pad time so the deepest dilation is causal
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        x = self.start_conv(x)
        supports = [self._masked_adp(nmask.float())] if self.adaptive else None
        skip = torch.zeros(1, device=x.device, dtype=x.dtype)
        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate                                 # gated activation unit
            s = self.skip_convs[i](x)
            skip = skip[..., -s.size(3):] if skip.dim() == 4 else skip
            skip = s + skip
            if self.adaptive:
                x = self.gconv[i](x, supports)              # diffusion conv on the self-adaptive graph
            else:
                x = self.residual_convs[i](x)              # no graph: 1x1 residual conv (paper "w/o adaptive")
            x = x + residual[..., -x.size(3):]              # residual connection (temporal-aligned)
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)                              # [B, out_dim, N, 1]
        return x[:, 0, :, -1]                               # [B, N]  (out_dim=1, last time step)
