"""Integration test: exercises train_alignment.py's REAL train_epoch (not a duplicate) — closes
the coverage gap flagged in code review 2026-07-18.
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_alignment import AlignmentLossBaseline
from train_alignment import train_epoch

pytestmark = pytest.mark.smoke


def _tiny_loader(B, T, S, A, D):
    x_har = torch.randn(B * 2, T, S, 3)
    adj = torch.rand(B * 2, S, S)
    x_emb = torch.randn(B * 2, T, S, A, D)
    mask = torch.ones(B * 2, T, S, A)
    y = torch.randn(B * 2, S)
    ds = TensorDataset(x_har, adj, x_emb, mask, y)
    return DataLoader(ds, batch_size=B, shuffle=False)


def test_train_epoch_runs_with_alignment_loss():
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = AlignmentLossBaseline(config, emb_dim=8, d_news=8, d_align=4, dropout=0.0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = _tiny_loader(B=2, T=3, S=2, A=4, D=8)

    loss = train_epoch(model, loader, criterion, optimizer, "cpu", lambda_align=0.1)
    assert loss == loss and loss < 100, f"train_epoch produced bad loss: {loss}"


if __name__ == "__main__":
    test_train_epoch_runs_with_alignment_loss()
    print("PASS")
