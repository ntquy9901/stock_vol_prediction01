"""Faithful MTGNN adaptive graph-learning layer + a drop-in learned-adjacency model wrapper.

Paper: Wu, Pan, Long, Jiang, Chang, Zhang, "Connecting the Dots: Multivariate Time Series
Forecasting with Graph Neural Networks", KDD 2020 (arXiv:2005.11650). Section 3.1 "Graph Learning
Layer". Reference implementation: github.com/nnzhan/MTGNN, ``layer.py::graph_constructor``.

PAPER EQUATIONS -> THIS CODE (verified against ar5iv.labs.arxiv.org/abs/2005.11650 and the official
code, 2026-08-29):

    Eq. (1)   M1 = tanh(alpha * E1 * Theta1)          -> ``m1 = tanh(alpha * self.theta1(self.emb1(idx)))``
    Eq. (2)   M2 = tanh(alpha * E2 * Theta2)          -> ``m2 = tanh(alpha * self.theta2(self.emb2(idx)))``
    Eq. (3)   A = ReLU(tanh(alpha*(M1 M2^T - M2 M1^T))) -> ``adj = relu(tanh(alpha*(m1@m2.T - m2@m1.T)))``
    Eq. (5-6) idx = argtopk(A[i,:]); A[i,-idx] = 0    -> per-row top-k mask (``adj.topk(k,1)`` + scatter)

  * ``E1, E2`` in R^{N x d} are learnable node-embedding matrices (``nn.Embedding``), ``Theta1, Theta2``
    are linear layers (``nn.Linear(d, d)``), ``alpha`` is the saturation rate (paper default alpha=3).
  * The subtraction ``M1 M2^T - M2 M1^T`` makes ``A`` DIRECTED / ASYMMETRIC: if ``A_uv > 0`` then
    ``A_vu = 0`` (paper). ReLU zeroes the negative half.
  * Top-k keeps only each node's k largest OUTGOING weights (connectivity / sparsity); the official code
    breaks ties with ``+ rand_like(adj)*0.01`` before ``topk`` -- replicated here for faithfulness.
  * NO self-loop is added inside ``graph_constructor`` in the official code (the self-loop / identity is
    added by the downstream propagation). This module therefore exposes ``add_self_loop`` on the wrapper,
    matching the ``self-loop = 1.0`` convention of the statistical adjacencies fed to ``WeightedGATLayer``.

The wrapper (``LearnedGraphNet``) subclasses the delivered ``MaskedRichNet`` so the LSTM temporal branch,
the 2-hop ``WeightedGATLayer`` spatial branch, the 5 node features, the masked panel, the HAR-X anchor,
the per-ticker scalers and the QLIKE evaluation are all IDENTICAL to the fixed-edge variants -- ONLY the
edge source differs (learned, built each forward) so the ablation isolates the edge mechanism. MTGNN
originally pairs the learned graph with mix-hop propagation; a controlled EDGE-only ablation instead keeps
the same GAT propagation across every edge choice (documented in design.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
# read-only import of the delivered training-path model (never edited)
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))
import run_masked_rich as RMR  # noqa: E402


class GraphConstructor(nn.Module):
    """MTGNN graph-learning layer (Eqs. 1-3 + top-k sparsification, Eqs. 5-6).

    Produces a learnable, directed, top-k-sparse [N, N] adjacency. ``A[i, j]`` is an edge kept among
    node ``i``'s top-k outgoing weights. No self-loop (added downstream). Faithful to nnzhan/MTGNN
    ``graph_constructor``.
    """

    def __init__(self, n_nodes: int, subgraph_size: int, node_dim: int, alpha: float = 3.0):
        super().__init__()
        if subgraph_size < 1:
            raise ValueError(f"subgraph_size (top-k) must be >= 1, got {subgraph_size}")
        self.n_nodes = n_nodes
        self.k = min(subgraph_size, n_nodes)   # cannot keep more neighbours than nodes
        self.alpha = alpha
        self.emb1 = nn.Embedding(n_nodes, node_dim)   # E1 in R^{N x d}
        self.emb2 = nn.Embedding(n_nodes, node_dim)   # E2 in R^{N x d}
        self.theta1 = nn.Linear(node_dim, node_dim)   # Theta1
        self.theta2 = nn.Linear(node_dim, node_dim)   # Theta2

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        m1 = torch.tanh(self.alpha * self.theta1(self.emb1(idx)))          # Eq. (1)
        m2 = torch.tanh(self.alpha * self.theta2(self.emb2(idx)))          # Eq. (2)
        a = torch.mm(m1, m2.transpose(1, 0)) - torch.mm(m2, m1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))                          # Eq. (3): directed, asymmetric
        # top-k per row (Eqs. 5-6): keep the k largest outgoing weights, zero the rest. Tie-break with a
        # tiny random perturbation exactly as the official code (nnzhan/MTGNN layer.py).
        mask = torch.zeros(idx.size(0), idx.size(0), device=adj.device)
        _, top_idx = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, top_idx, 1.0)
        return adj * mask


class LearnedGraphNet(RMR.MaskedRichNet):
    """``MaskedRichNet`` whose GAT adjacency is BUILT each forward by an MTGNN ``GraphConstructor``.

    Everything except the edge source is inherited unchanged: the 2-layer LSTM temporal branch, the 2-hop
    ``WeightedGATLayer`` spatial branch, and the fusion head. ``forward(x, nmask)`` mirrors the parent's
    ``forward(x, adj_b)`` but constructs ``adj_b`` internally from the learnable graph (self-loop added,
    then masked by valid source nodes -- the same ``base * nmask.unsqueeze(1)`` batching the training loop
    applies to the fixed adjacencies).
    """

    def __init__(self, n_nodes: int, hidden: int = 64, heads: int = 4, dropout: float = 0.2,
                 subgraph_size: int = 20, node_dim: int = 40, alpha: float = 3.0, gat_layers: int = 2):
        super().__init__(hidden, heads, dropout, use_graph=True, gat_layers=gat_layers)
        self.gc = GraphConstructor(n_nodes, subgraph_size, node_dim, alpha)
        self.register_buffer("_node_idx", torch.arange(n_nodes))

    def learned_adjacency(self) -> torch.Tensor:
        """The [N, N] learned adjacency with a self-loop (diagonal = 1.0), pre-masking."""
        adj = self.gc(self._node_idx)
        eye = torch.eye(adj.size(0), device=adj.device)
        # self-loop weight 1.0 (matching adj_vol2pk/adj_corr): overwrite the diagonal rather than add, so a
        # non-zero learned diagonal cannot exceed 1.0.
        return adj * (1.0 - eye) + eye

    def build_adj(self, nmask: torch.Tensor) -> torch.Tensor:
        """[B, N, N] batched adjacency = learned base * valid-source-node mask (training-loop convention)."""
        base = self.learned_adjacency()
        return base.unsqueeze(0) * nmask.unsqueeze(1)

    def forward(self, x: torch.Tensor, nmask: torch.Tensor) -> torch.Tensor:  # x [B,N,seq,5]; nmask [B,N]
        adj_b = self.build_adj(nmask)
        b, n, seq, d = x.shape
        out, _ = self.lstm(x.reshape(b * n, seq, d))
        h = out[:, -1].reshape(b, n, self.hidden)
        gnn = self._gat(x[:, :, -1, :], adj_b)
        return self.head(torch.cat([h, gnn], -1)).squeeze(-1)
