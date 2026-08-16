# code/mad.py
"""MAD (Mean Average Distance) — over-smoothing diagnostic for GNN node embeddings.

MAD(emb) = mean over present ordered node pairs (i != j) of (1 - cosine_similarity(emb_i, emb_j)).
Lower MAD => embeddings more similar => more over-smoothed. Stacking GNN layers drives MAD down
(Zhang et al., arXiv:2308.01419): use it to decide whether a 2nd hop is worth the smoothing cost.
"""
from __future__ import annotations

import torch


def mad(emb: torch.Tensor, presence: torch.Tensor | None = None) -> torch.Tensor:
    """`emb` [N, d]; optional `presence` [N] (1=present). Scalar tensor; 0 if <2 present nodes."""
    n = emb.shape[0]
    if presence is None:
        presence = torch.ones(n, dtype=emb.dtype, device=emb.device)
    e = emb[presence > 0]
    m = e.shape[0]
    if m < 2:
        return torch.zeros((), dtype=emb.dtype, device=emb.device)
    normed = torch.nn.functional.normalize(e, dim=-1)
    dist = 1.0 - normed @ normed.t()                                  # [m, m] cosine distance
    off_diag = ~torch.eye(m, dtype=torch.bool, device=emb.device)
    return dist[off_diag].mean()
