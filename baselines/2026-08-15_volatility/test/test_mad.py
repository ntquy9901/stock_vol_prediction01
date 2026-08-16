"""P2: MAD (Mean Average Distance) over-smoothing diagnostic (Zhang et al. 2308.01419).

MAD(emb) = mean over present ordered node pairs (i != j) of (1 - cosine_similarity(emb_i, emb_j)).
Lower MAD => node embeddings more similar => more over-smoothed (what stacking GNN layers causes).
"""
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from mad import mad  # noqa: E402


def test_mad_identical_embeddings_is_zero():
    emb = torch.ones(4, 8)
    assert abs(mad(emb).item()) < 1e-6


def test_mad_orthogonal_embeddings_is_one():
    emb = torch.eye(3)               # rows pairwise orthogonal -> cosine 0 -> distance 1
    assert abs(mad(emb).item() - 1.0) < 1e-6


def test_mad_presence_excludes_absent_nodes():
    # node 2 is a wild outlier but absent -> must not enter the pairwise mean
    emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    presence = torch.tensor([1.0, 1.0, 0.0])
    # only nodes 0,1 present and identical -> MAD 0
    assert abs(mad(emb, presence).item()) < 1e-6


def test_mad_fewer_than_two_present_is_zero():
    emb = torch.randn(3, 5)
    presence = torch.tensor([1.0, 0.0, 0.0])
    assert mad(emb, presence).item() == 0.0
