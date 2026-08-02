"""Test-first: validate() in train_sp500_lstm_gnn.py must compute
directional_accuracy per-STOCK (chronological order within a stock), not by
flattening [batch, num_stocks] -> 1-D (which interleaves different stocks on
the same day, and different days across the batch boundary -- both spurious).

Found while smoke-testing the new LSTM-GNN baseline: the flatten-then-diff
pattern was copied faithfully from src/lstm_gat_hybrid/train_parallel.py
(VN30's own proven code), which has the same bug.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = baseline_dir
for _ in range(2):
    project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(baseline_dir, "code"))


class _StubModel(nn.Module):
    """Returns a fixed prediction tensor supplied per-call via `x`'s 2nd
    channel (see _make_batch) -- avoids needing a real trained LSTM/GAT."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj_matrix):
        return x[:, 1, :, 0] + self.dummy * 0  # channel 1 = pred values


class _IdentityNormalizer:
    """Stub normalizer: inverse_transform is the identity (no-op)."""

    def inverse_transform(self, data):
        return data


class _LinearNormalizer:
    """Stub normalizer matching VolatilityNormalizer's inverse_transform:
    data * std + mean."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def _make_batch(true_values, pred_values):
    """true_values/pred_values: [batch, num_stocks] arrays."""
    y = torch.tensor(true_values, dtype=torch.float32)
    batch, num_stocks = y.shape
    x = torch.zeros(batch, 2, num_stocks, 1)
    x[:, 0, :, 0] = y
    x[:, 1, :, 0] = torch.tensor(pred_values, dtype=torch.float32)
    adj = torch.zeros(num_stocks, num_stocks)
    return x, adj, y, {}


class TestValidateGroupedDirAcc:
    def test_per_stock_grouping_not_flattened_across_stocks(self):
        from train_sp500_lstm_gnn import validate

        # 2 stocks, 3 chronological days (1 batch each).
        # True:  A = 1 -> 2 -> 1   (up, down)     B = 100 -> 50 -> 200  (down, up)
        # Pred:  A = 10 -> 20 -> 10 (up, down, SAME sign as true A)
        #        B = 1 -> 0.5 -> 2  (down, up, SAME sign as true B)
        # Every WITHIN-stock direction is predicted correctly -> grouped
        # accuracy must be 100%. But flattening [A_day, B_day] per day and
        # taking np.diff() over the pooled sequence compares DIFFERENT
        # stocks on the SAME day (and across the day boundary) -- with these
        # magnitudes, every single one of those spurious comparisons has the
        # OPPOSITE sign between true and pred, so the naive (buggy) flatten
        # approach scores 0%. This gap is what proves the grouping matters,
        # not just an off-by-a-few-percent artifact.
        batches = [
            _make_batch([[1.0, 100.0]], [[10.0, 1.0]]),
            _make_batch([[2.0, 50.0]], [[20.0, 0.5]]),
            _make_batch([[1.0, 200.0]], [[10.0, 2.0]]),
        ]

        model = _StubModel()
        criterion = nn.MSELoss()
        stock_names = ["A", "B"]
        normalizers = {"A": _IdentityNormalizer(), "B": _IdentityNormalizer()}
        _, metrics = validate(model, batches, criterion, torch.device("cpu"), stock_names, normalizers)

        assert metrics["directional_accuracy"] == 100.0


class TestValidateInverseTransform:
    def test_qlike_computed_on_original_scale_not_normalized_scale(self):
        """MultiStockDataset returns z-score NORMALIZED y (mean 0, std 1,
        can be negative) -- QLIKE is only meaningful on the original positive
        volatility scale. Found via smoke test: QLIKE=19620 (should be O(1))
        because normalized values near/below 0 get clipped to epsilon=1e-8
        inside qlike_loss(), blowing up the ratio term. validate() must
        inverse_transform each stock's true/pred back to original scale
        BEFORE computing metrics.
        """
        from train_sp500_lstm_gnn import validate

        # Stock A: original-scale volatility around mean=0.002, std=0.001.
        # In normalized (z-score) space, e.g. 0.001 original = -1.0 normalized.
        mean_a, std_a = 0.002, 0.001
        # Perfect predictions in NORMALIZED space (pred == true exactly).
        normalized_true = [[-1.0], [0.0], [1.0]]  # -> original: 0.001, 0.002, 0.003
        batches = [_make_batch([normalized_true[i]], [normalized_true[i]]) for i in range(3)]

        model = _StubModel()
        criterion = nn.MSELoss()
        stock_names = ["A"]
        normalizers = {"A": _LinearNormalizer(mean=mean_a, std=std_a)}
        _, metrics = validate(model, batches, criterion, torch.device("cpu"), stock_names, normalizers)

        # Perfect prediction (even after inverse-transform) -> QLIKE ~ 0, not
        # a blown-up value in the thousands.
        assert metrics["qlike"] < 1e-6
        assert metrics["rmse"] < 1e-6
