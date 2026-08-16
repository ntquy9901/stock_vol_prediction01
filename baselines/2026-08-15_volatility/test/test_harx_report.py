"""Fairness baseline: HAR-X = linear regression on ALL 5 node features (HAR + market_pk +
volume_zscore), isolating extra-feature contribution from LSTM nonlinearity. Core linear predictor
must be an ordinary least-squares fit (pure, testable)."""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import harx_report as hx  # noqa: E402


def test_linear_norm_predictions_recovers_linear_map():
    # y = 3*x0 - 2*x1 + 1 ; OLS must recover it on test points
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(200, 2))
    y_train = 3 * x_train[:, 0] - 2 * x_train[:, 1] + 1.0
    x_test = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    pred = hx._linear_norm_predictions(x_train, y_train, x_test)
    expected = 3 * x_test[:, 0] - 2 * x_test[:, 1] + 1.0
    assert np.allclose(pred, expected, atol=1e-6)


def test_linear_norm_predictions_uses_all_columns():
    # a model that ignored column 1 would mispredict when only column 1 varies
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(300, 5))
    y_train = x_train[:, 4] * 5.0                  # depends ONLY on the 5th feature
    x_test = np.zeros((2, 5))
    x_test[1, 4] = 1.0
    pred = hx._linear_norm_predictions(x_train, y_train, x_test)
    assert pred[1] - pred[0] > 4.0                 # must respond to the 5th feature
