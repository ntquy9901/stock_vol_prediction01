"""Tests for the AUD-016 non-finite metric guard in src/common/evaluation.py.

Background: json.dump()/json.dumps() default to allow_nan=True, so a NaN or
Infinity metric value would otherwise be silently written into a results.json
file as an invalid-JSON token instead of surfacing the bug that produced it
(e.g. QLIKE or MSE overflowing to Infinity for extreme-magnitude inputs).
assert_finite_metrics() / evaluate_predictions() must fail loudly instead.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.evaluation import assert_finite_metrics, evaluate_predictions

pytestmark = pytest.mark.smoke


class TestAssertFiniteMetricsUnit:
    def test_all_finite_metrics_pass_without_raising(self):
        metrics = {"mse": 0.01, "rmse": 0.1, "mae": 0.05, "r2": 0.8,
                   "qlike": 0.02, "directional_accuracy": 60.0}
        assert_finite_metrics(metrics)  # must not raise

    def test_nan_value_raises_value_error_naming_key(self):
        metrics = {"mse": 0.01, "qlike": float("nan"), "directional_accuracy": 60.0}
        with pytest.raises(ValueError, match="qlike"):
            assert_finite_metrics(metrics)

    def test_positive_infinity_raises_value_error_naming_key(self):
        metrics = {"mse": float("inf"), "rmse": 0.1}
        with pytest.raises(ValueError, match="mse"):
            assert_finite_metrics(metrics)

    def test_negative_infinity_raises_value_error(self):
        metrics = {"r2": float("-inf")}
        with pytest.raises(ValueError, match="r2"):
            assert_finite_metrics(metrics)


class TestEvaluatePredictionsFiniteGuard:
    def test_normal_finite_inputs_return_all_six_metrics(self):
        rng = np.random.default_rng(0)
        y_true = np.abs(rng.normal(0.01, 0.002, size=50))
        y_pred = y_true + rng.normal(0, 0.0005, size=50)
        y_pred = np.maximum(y_pred, 1e-6)

        metrics = evaluate_predictions(y_true, y_pred)

        for key in ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"):
            assert key in metrics
            assert np.isfinite(metrics[key]), f"{key} should be finite for normal inputs"

    def test_normal_inputs_with_n_stocks_still_finite(self):
        rng = np.random.default_rng(1)
        n_stocks = 3
        num_windows = 20
        y_true = np.abs(rng.normal(0.01, 0.002, size=num_windows * n_stocks))
        y_pred = y_true + rng.normal(0, 0.0005, size=num_windows * n_stocks)
        y_pred = np.maximum(y_pred, 1e-6)

        metrics = evaluate_predictions(y_true, y_pred, n_stocks=n_stocks)

        assert np.isfinite(metrics["directional_accuracy"])
        assert np.isfinite(metrics["directional_accuracy_flat_biased"])

    def test_extreme_magnitude_inputs_raise_instead_of_returning_non_finite(self):
        """Reproduces AUD-016: an extreme-magnitude (but individually finite,
        non-NaN) true/pred pair overflows MSE/R^2 to Infinity/NaN even though
        QLIKE's own epsilon guard clips the near-zero side. Without the new
        guard this dict would silently be handed to json.dump() as-is.
        """
        y_true = np.array([0.01, 0.02, 0.015, 1e300])
        y_pred = np.array([0.01, 0.02, 0.015, 0.0])

        with np.errstate(over="ignore", invalid="ignore"):
            with pytest.raises(ValueError, match="non-finite"):
                evaluate_predictions(y_true, y_pred)
