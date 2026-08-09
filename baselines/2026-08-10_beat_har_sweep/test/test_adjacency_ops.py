"""Tests for static-adjacency masking and the learned adjacency module (C3/C5/C6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from adjacency_ops import LearnedAdjacency, mask_learned_adjacency, mask_static_adjacency  # noqa: E402


def _present_have_edge(adjacency, presence):
    present = np.flatnonzero(presence)
    return all(adjacency[i].astype(bool).any() for i in present)


def test_static_keeps_self_loop_and_masks_absent():
    static = np.array([[1.0, 0.4, 0.2], [0.3, 1.0, 0.5], [0.1, 0.6, 1.0]], dtype=np.float32)
    presence = np.array([1, 0, 1])
    adjacency = mask_static_adjacency(static, presence)
    assert adjacency[1].sum() == 0 and adjacency[:, 1].sum() == 0  # absent node isolated
    assert adjacency[0, 0] == 1.0 and adjacency[2, 2] == 1.0  # self-loops kept
    assert adjacency[0, 2] == pytest.approx(0.2)
    assert _present_have_edge(adjacency, presence)


def test_static_omit_self_directed_topk():
    static = np.array([[1.0, 0.9, 0.1, 0.2], [0.3, 1.0, 0.5, 0.05],
                       [0.1, 0.6, 1.0, 0.8], [0.7, 0.2, 0.3, 1.0]], dtype=np.float32)
    presence = np.array([1, 1, 1, 1])
    adjacency = mask_static_adjacency(static, presence, omit_self=True, top_k=1)
    assert np.diag(adjacency).sum() == 0.0  # no self-loops
    # each row keeps exactly its single strongest off-diagonal neighbour
    assert (adjacency.astype(bool).sum(axis=1) == 1).all()
    assert adjacency[0, 1] > 0  # 0's strongest neighbour is 1 (0.9)
    assert _present_have_edge(adjacency, presence)


def test_static_omit_self_isolated_fallback():
    """A present node whose only nonzero out-weights are to absent nodes falls back to a self-loop."""

    static = np.array([[1.0, 0.0, 0.9], [0.5, 1.0, 0.0], [0.0, 0.8, 1.0]], dtype=np.float32)
    presence = np.array([1, 1, 0])  # node 2 absent; node 0's only neighbour (2) is gone
    adjacency = mask_static_adjacency(static, presence, omit_self=True, top_k=1)
    assert _present_have_edge(adjacency, presence)
    assert adjacency[0, 0] == 1.0  # isolated -> self-loop fallback


def test_learned_adjacency_grad_and_shape():
    module = LearnedAdjacency(num_nodes=6, dim=8, top_k=3)
    adjacency = module()
    assert adjacency.shape == (6, 6)
    assert (adjacency >= 0).all()
    assert (adjacency.detach().bool().sum(dim=1) <= 3).all()  # top-k sparsity per row
    adjacency.sum().backward()
    assert module.embed_source.grad is not None
    assert module.embed_source.grad.abs().sum() > 0


def test_mask_learned_adds_self_loop_and_masks_absent():
    module = LearnedAdjacency(num_nodes=4, dim=4, top_k=2)
    adjacency = module()
    presence = torch.tensor([[True, True, False, True]])
    masked = mask_learned_adjacency(adjacency, presence)
    assert masked.shape == (1, 4, 4)
    assert masked[0, 2].sum() == 0 and masked[0, :, 2].sum() == 0  # absent node 2 isolated
    assert masked[0, 0, 0] > 0 and masked[0, 1, 1] > 0 and masked[0, 3, 3] > 0  # self-loops
