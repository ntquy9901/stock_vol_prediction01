"""Tests for the per-ticker DirAcc fix applied to the original HAR-only pipeline's
train.py / train_parallel.py / train_parallel_enhanced.py.

Background: src/common/evaluation.py's evaluate_predictions() was fixed (see git log
commit "Fix reproducibility gaps ... add per-ticker DirAcc as headline metric") to
accept an optional n_stocks parameter. When given, it reshapes a flattened multi-stock
array as (num_windows, n_stocks) — day-major, ticker-interleaved order, i.e. index i
belongs to ticker (i % n_stocks) on window (i // n_stocks) — and reports the correct
per-ticker directional_accuracy as the headline value, keeping the old (biased)
flattened calculation under directional_accuracy_flat_biased. That fix was applied
across ~22 news-fusion baseline callers but NOT to these 3 original, non-news
"HAR-only" pipeline files (src/lstm_gat_hybrid/{train,train_parallel,
train_parallel_enhanced}.py) — this test suite covers that gap.

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
from src.lstm_gat_hybrid import train as train_module
from src.lstm_gat_hybrid import train_parallel as train_parallel_module
from src.lstm_gat_hybrid import train_parallel_enhanced as train_parallel_enhanced_module

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
    """Dummy model that ignores its inputs and returns pre-baked predictions in
    call order (one queued tensor per forward() call). Lets the test drive
    validate()'s I/O path (dataloader -> model -> flatten -> evaluate_predictions)
    with a KNOWN, controlled prediction sequence instead of testing evaluate_predictions
    in isolation."""

    def __init__(self, batched_predictions):
        super().__init__()
        self._queue = list(batched_predictions)
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj_matrix):
        return self._queue.pop(0)


def _make_dummy_batches(t2, p2, n_stocks):
    """One 'batch' per window (batch_size=1), y from t2, model prediction from p2 —
    matches the day-major ordering _create_sequences()/MultiStockDataset produce
    (see dataset.py / dataset_presplit.py), since shuffle=False everywhere and
    extend() concatenates batches in dataloader iteration order."""
    n_windows = t2.shape[0]
    batches = []
    predictions = []
    for w in range(n_windows):
        y_tensor = torch.tensor(t2[w:w + 1], dtype=torch.float32)  # [1, n_stocks]
        x_dummy = torch.zeros(1, 1, n_stocks, 1)
        adj_dummy = torch.zeros(1, n_stocks, n_stocks)
        batches.append((x_dummy, adj_dummy, y_tensor, None))
        predictions.append(torch.tensor(p2[w:w + 1], dtype=torch.float32))  # [1, n_stocks]
    return batches, predictions


class _IdentityNormalizer:
    """Stand-in for VolatilityNormalizer that leaves values unchanged, so the
    denormalized array equals the known-diverging array we constructed."""

    def inverse_transform(self, arr):
        return np.asarray(arr)


class _FakeDatasetNoSubset:
    def __init__(self, stock_names):
        self.stock_names = stock_names
        self.target_normalizers = {name: _IdentityNormalizer() for name in stock_names}


class TestTrainPyValidateIntegration:
    """I/O-level test of src/lstm_gat_hybrid/train.py's validate(), per CLAUDE.md's
    rule that run_*()-style functions need an integration test, not just a pure
    helper test of evaluate_predictions."""

    def test_validate_reports_per_ticker_diracc_as_headline(self):
        """No dataset= given: exercises the no-normalizer path (metrics computed
        directly on the normalized-scale arrays coming out of the dataloader)."""
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        batches, predictions = _make_dummy_batches(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        device = torch.device('cpu')

        _, metrics = train_module.validate(model, batches, criterion, device)

        assert metrics['directional_accuracy'] == pytest.approx(0.0), (
            "validate() must report the per-ticker DirAcc as the headline value, "
            "not the flattened/inflated one"
        )
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)

    def test_validate_with_dataset_denormalizes_before_diracc(self):
        """dataset= given (the real-run path via create_multi_stock_dataloaders'
        post-P1.1-fix normalized targets): exercises the per-stock inverse_transform
        loop (train.py ~line 195-219) that must map i % n_stocks back to the same
        stock_names order used when sequences were built, BEFORE computing DirAcc."""
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        batches, predictions = _make_dummy_batches(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        device = torch.device('cpu')
        fake_dataset = _FakeDatasetNoSubset(['A', 'B'])

        _, metrics = train_module.validate(model, batches, criterion, device, dataset=fake_dataset)

        assert metrics['directional_accuracy'] == pytest.approx(0.0), (
            "the denormalization branch must report per-ticker DirAcc as the "
            "headline value on the denormalized (identity here) scale"
        )
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)


class TestTrainParallelPyValidateIntegration:
    """I/O-level test of src/lstm_gat_hybrid/train_parallel.py's validate()."""

    def test_validate_reports_per_ticker_diracc_as_headline(self):
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        batches, predictions = _make_dummy_batches(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        device = torch.device('cpu')

        _, metrics = train_parallel_module.validate(model, batches, criterion, device)

        assert metrics['directional_accuracy'] == pytest.approx(0.0)
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)

    def test_validate_with_dataset_denormalizes_before_diracc(self):
        """Same denorm-branch coverage as train.py's equivalent test above."""
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        batches, predictions = _make_dummy_batches(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        device = torch.device('cpu')
        fake_dataset = _FakeDatasetNoSubset(['A', 'B'])

        _, metrics = train_parallel_module.validate(
            model, batches, criterion, device, dataset=fake_dataset
        )

        assert metrics['directional_accuracy'] == pytest.approx(0.0)
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)


class TestTrainParallelEnhancedPyValidateIntegration:
    """I/O-level test of src/lstm_gat_hybrid/train_parallel_enhanced.py's validate(),
    covering both code branches touched by the fix: the normalizer branch (the one
    actually used by every real training run in this file, dataset always has
    target_normalizers) and the no-normalizer fallback branch."""

    def test_validate_with_dataset_normalizers_reports_per_ticker_diracc(self):
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        batches, predictions = _make_dummy_batches(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        device = torch.device('cpu')
        fake_dataset = _FakeDatasetNoSubset(['A', 'B'])

        _, metrics = train_parallel_enhanced_module.validate(
            model, batches, criterion, device, dataset=fake_dataset
        )

        assert metrics['directional_accuracy'] == pytest.approx(0.0), (
            "the normalizer branch (used by every real run of this file) must "
            "report per-ticker DirAcc as the headline value"
        )
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)

    def test_validate_fallback_without_dataset_reports_per_ticker_diracc(self):
        y_true, y_pred, n_stocks, t2, p2 = _make_known_diverging_arrays()
        batches, predictions = _make_dummy_batches(t2, p2, n_stocks)
        model = _QueueModel(predictions)
        criterion = nn.MSELoss()
        device = torch.device('cpu')

        _, metrics = train_parallel_enhanced_module.validate(
            model, batches, criterion, device, dataset=None
        )

        assert metrics['directional_accuracy'] == pytest.approx(0.0), (
            "the no-normalizer fallback branch must also report per-ticker DirAcc "
            "as the headline value (num_stocks captured from the dataloader loop)"
        )
        assert metrics['directional_accuracy_flat_biased'] == pytest.approx(100.0)
