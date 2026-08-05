"""Tests for the per-ticker DirAcc fix applied to
src/lstm_har_gat_hybrid/train_hybrid.py's HybridModelTrainer.validate().

Background: src/common/evaluation.py's evaluate_predictions() accepts an optional
n_stocks parameter. When given, it reshapes a flattened multi-stock array as
(num_windows, n_stocks) — day-major, ticker-interleaved order, i.e. index i
belongs to ticker (i % n_stocks) on window (i // n_stocks) — and reports the
correct per-ticker directional_accuracy as the headline value, keeping the old
(biased) flattened calculation under directional_accuracy_flat_biased. That fix
was already applied across ~22 news-fusion baseline callers and to
src/lstm_gat_hybrid/{train,train_parallel,train_parallel_enhanced}.py but NOT to
this hybrid (LSTM-HAR + GAT) pipeline's validate() — this test suite covers that
gap.

See docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md for the full write-up of why the
flattened (day-major) np.diff mostly compares different tickers' same-day values
instead of the same ticker across time, inflating the metric.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.evaluation import evaluate_predictions
from src.lstm_har_gat_hybrid.train_hybrid import HybridModelTrainer

pytestmark = pytest.mark.smoke


def _make_known_diverging_arrays():
    """2 tickers x 5 windows, day-major flattened.

    Per-ticker DirAcc is 0% (predictions move OPPOSITE to the true small
    oscillation for every ticker, every window) while the flattened (day-major,
    cross-ticker) DirAcc is 100%, because ticker levels differ by two orders of
    magnitude (100 vs 1) so np.diff on the interleaved array is dominated by
    WHICH TICKER comes next, not the true temporal change — the exact failure
    mode in docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md.
    """
    n_stocks = 2
    t2 = np.array([
        [100.0, 1.00],
        [101.0, 1.01],
        [100.0, 1.00],
        [101.0, 1.01],
        [100.0, 1.00],
    ])
    p2 = np.array([
        [100.0, 1.00],
        [99.0, 0.99],
        [100.0, 1.00],
        [99.0, 0.99],
        [100.0, 1.00],
    ])
    y_true = t2.reshape(-1)
    y_pred = p2.reshape(-1)
    return y_true, y_pred, n_stocks, t2, p2


class TestEvaluatePredictionsKnownDivergence:
    """Pure-helper sanity check that the known scenario really does diverge
    (guards the rest of this file's assumptions before testing the wiring)."""

    def test_flat_diracc_is_inflated_to_100_percent(self):
        y_true, y_pred, n_stocks, _, _ = _make_known_diverging_arrays()
        flat_only = evaluate_predictions(y_true, y_pred)
        assert flat_only['directional_accuracy'] == pytest.approx(100.0)

    def test_per_ticker_diracc_is_correctly_0_percent(self):
        y_true, y_pred, n_stocks, _, _ = _make_known_diverging_arrays()
        per_ticker = evaluate_predictions(y_true, y_pred, n_stocks=n_stocks)
        assert per_ticker['directional_accuracy'] == pytest.approx(0.0)
        assert per_ticker['directional_accuracy_flat_biased'] == pytest.approx(100.0)


class _QueueModel(nn.Module):
    """Dummy model matching LSTMHAR_GAT_Hybrid's forward signature
    (x, edge_index, edge_weight) that ignores its inputs and returns pre-baked
    predictions in call order (one queued tensor per forward() call). Lets the
    test drive validate()'s I/O path (dataloader -> model -> flatten ->
    evaluate_predictions) with a KNOWN, controlled prediction sequence instead
    of testing evaluate_predictions in isolation."""

    def __init__(self, batched_predictions):
        super().__init__()
        self._queue = list(batched_predictions)
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_weight=None):
        return self._queue.pop(0)


def _make_dummy_val_loader(t2, p2, n_stocks):
    """One 'batch' per window (batch_size=1), y from t2 shaped (1, n_stocks, 1)
    as produced by custom_collate_fn, and queued model predictions from p2 with
    the same shape — matches train_hybrid.py's HybridVolatilityDataset +
    custom_collate_fn layout (day-major, batch-major/stock-minor within a row)."""
    n_windows = t2.shape[0]
    batches = []
    predictions = []
    for w in range(n_windows):
        y_tensor = torch.tensor(t2[w:w + 1], dtype=torch.float32).unsqueeze(-1)  # (1, n_stocks, 1)
        x_dummy = torch.zeros(1, n_stocks, 22, 3)
        batches.append((x_dummy, y_tensor))
        predictions.append(torch.tensor(p2[w:w + 1], dtype=torch.float32).unsqueeze(-1))  # (1, n_stocks, 1)
    return batches, predictions


class TestHybridModelTrainerValidateIntegration:
    """I/O-level test of HybridModelTrainer.validate(), per CLAUDE.md's rule
    that run_*()-style functions need an integration test, not just a pure
    helper test of evaluate_predictions."""

    def test_validate_reports_per_ticker_diracc_as_headline(self):
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        val_loader, predictions = _make_dummy_val_loader(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        trainer = HybridModelTrainer(model, device='cpu')

        _, metrics = trainer.validate(
            val_loader, edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_weight=None, criterion=criterion
        )

        assert metrics['directional_accuracy'] == pytest.approx(0.0), (
            "validate() must report the per-ticker DirAcc as the headline value, "
            "not the flattened/inflated one"
        )
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)
