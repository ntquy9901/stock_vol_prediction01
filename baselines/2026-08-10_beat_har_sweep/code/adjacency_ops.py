"""Adjacency construction helpers for the beat-HAR sweep (static spillover masking, learned graph).

All helpers preserve the pilot message-passing invariant "each PRESENT node has a self-loop or a
neighbour": static masks keep a present-node self-loop (or a directed-top-k fallback), and the learned
graph adds a present-node self-loop diagonal.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mask_static_adjacency(
    static: np.ndarray, presence: np.ndarray, omit_self: bool = False, top_k: int | None = None
) -> np.ndarray:
    """Mask a single frozen [N,N] matrix to one snapshot's present nodes.

    Args:
        static: [N,N] directed weights (e.g. spillover connectedness); rows = receiver i.
        presence: [N] binary present mask.
        omit_self: if True, drop the self-loop and instead keep directed top-k present neighbours,
            with a self-loop fallback for any present node left isolated.
        top_k: number of directed out-neighbours to keep per present row when ``omit_self``.

    Returns:
        [N,N] float32 masked adjacency; absent rows/cols zero; every present node non-isolated.
    """

    n = static.shape[0]
    present = np.flatnonzero(np.asarray(presence).astype(bool))
    adjacency = np.zeros((n, n), dtype=np.float32)
    if present.size == 0:
        raise ValueError("static adjacency masking requires at least one present node")
    sub = static[np.ix_(present, present)].astype(np.float32).copy()
    if not omit_self:
        adjacency[np.ix_(present, present)] = sub
        for idx in present:
            adjacency[idx, idx] = 1.0
        return adjacency
    # omit self-loops: keep directed top-k present neighbours per row (exclude self)
    magnitude = np.abs(sub).copy()
    np.fill_diagonal(magnitude, -np.inf)
    kept = np.zeros_like(sub, dtype=bool)
    k = min(top_k if top_k is not None else sub.shape[0] - 1, sub.shape[0] - 1)
    if k >= 1:
        order = np.argsort(-magnitude, axis=1, kind="stable")[:, :k]
        np.put_along_axis(kept, order, True, axis=1)
        finite = np.isfinite(magnitude)
        kept &= finite  # never keep a -inf (self) slot when the row has < k real neighbours
    masked_sub = np.where(kept, sub, 0.0).astype(np.float32)
    np.fill_diagonal(masked_sub, 0.0)
    # isolated-node fallback: a present node with no kept out-edge gets a self-loop
    isolated = ~masked_sub.astype(bool).any(axis=1)
    for local in np.flatnonzero(isolated):
        masked_sub[local, local] = 1.0
    adjacency[np.ix_(present, present)] = masked_sub
    return adjacency


class LearnedAdjacency(nn.Module):
    """MTGNN-style directed learned adjacency from input-independent per-ticker embeddings.

    ``A = ReLU(tanh(alpha * (E1 E2^T - E2 E1^T)))``, top-k sparsified per row. The embeddings are
    plain parameters (no dependence on inputs / targets), so the learned graph cannot leak future
    observations. A present-node self-loop is added at masking time to satisfy the MP invariant.
    """

    def __init__(self, num_nodes: int, dim: int = 16, top_k: int = 8, alpha: float = 3.0) -> None:
        super().__init__()
        self.embed_source = nn.Parameter(torch.randn(num_nodes, dim) * 0.1)
        self.embed_target = nn.Parameter(torch.randn(num_nodes, dim) * 0.1)
        self.top_k = top_k
        self.alpha = alpha

    def forward(self) -> torch.Tensor:
        product = self.embed_source @ self.embed_target.t() - self.embed_target @ self.embed_source.t()
        adjacency = torch.relu(torch.tanh(self.alpha * product))
        n = adjacency.shape[0]
        k = min(self.top_k, n - 1)
        if k < n - 1:
            # keep the top-k entries per row, zero the rest (differentiable through kept entries)
            threshold = torch.topk(adjacency, k=k, dim=1).values[:, -1:].detach()
            adjacency = torch.where(adjacency >= threshold, adjacency, torch.zeros_like(adjacency))
        return adjacency


def mask_learned_adjacency(adjacency: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """Expand a shared [N,N] learned graph to [B,N,N], zero absent nodes, add present self-loops."""

    if presence.ndim != 2:
        raise ValueError("presence must be [batch, nodes]")
    present = presence.to(dtype=adjacency.dtype)
    batch = present.shape[0]
    expanded = adjacency.unsqueeze(0).expand(batch, -1, -1)
    expanded = expanded * present.unsqueeze(1) * present.unsqueeze(2)
    eye = torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
    self_loops = eye.unsqueeze(0) * present.unsqueeze(2)
    # add self-loops for present nodes (overwrites diagonal to a positive value)
    return expanded * (1.0 - eye.unsqueeze(0)) + self_loops
