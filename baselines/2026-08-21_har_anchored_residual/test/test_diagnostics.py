"""TDD: pre-gating diagnostics (plan section 14)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "code"))

import diagnostics  # noqa: E402


# --------------------------------------------------------------------------- #
# error_complementarity                                                       #
# --------------------------------------------------------------------------- #
def test_error_complementarity_perfectly_correlated_errors():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    # Both experts share the SAME error vector -> HAR err == NN err -> corr ~ 1.
    err = rng.normal(size=100)
    har_pred = y - err
    nn_pred = y - err
    out = diagnostics.error_complementarity(y, har_pred, nn_pred)
    assert out["pearson"] == pytest.approx(1.0, abs=1e-9)
    assert out["spearman"] == pytest.approx(1.0, abs=1e-9)


def test_error_complementarity_anticorrelated_errors():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    err = rng.normal(size=100)
    har_pred = y - err
    nn_pred = y + err                                    # NN err = -HAR err
    out = diagnostics.error_complementarity(y, har_pred, nn_pred)
    assert out["pearson"] == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# disagreement_winrate                                                        #
# --------------------------------------------------------------------------- #
def test_disagreement_winrate_bins_partition_all_rows():
    rng = np.random.default_rng(2)
    n = 200
    y = rng.normal(size=n)
    har_pred = y + rng.normal(0.0, 0.5, size=n)
    nn_pred = y + rng.normal(0.0, 0.5, size=n)
    bins = diagnostics.disagreement_winrate(y, har_pred, nn_pred, n_bins=5)
    assert len(bins) == 5
    assert sum(b["count"] for b in bins) == n
    for b in bins:
        assert set(b) == {"bin_lo", "bin_hi", "count", "p_nn_better"}
        if b["count"] > 0:
            assert 0.0 <= b["p_nn_better"] <= 1.0


def test_disagreement_winrate_detects_nn_advantage_on_big_disagreement():
    # Where NN and HAR disagree a lot, make NN essentially correct and HAR wrong.
    n = 300
    rng = np.random.default_rng(4)
    y = rng.normal(size=n)
    d = np.linspace(-3, 3, n)                             # disagreement magnitude spread
    nn_pred = y + rng.normal(0.0, 0.01, size=n)           # NN near-perfect
    har_pred = nn_pred - d                                # HAR off by d
    bins = diagnostics.disagreement_winrate(y, har_pred, nn_pred, n_bins=4)
    # top |d| bin: NN should win almost always
    assert bins[-1]["p_nn_better"] > 0.9


# --------------------------------------------------------------------------- #
# residual_r2_oos                                                             #
# --------------------------------------------------------------------------- #
def test_residual_r2_oos_perfect_prediction():
    rng = np.random.default_rng(5)
    y = rng.normal(size=50)
    har_pred = rng.normal(size=50)
    r = y - har_pred
    assert diagnostics.residual_r2_oos(y, har_pred, resid_pred=r) == pytest.approx(1.0)


def test_residual_r2_oos_zero_prediction_is_zero():
    rng = np.random.default_rng(6)
    y = rng.normal(size=50)
    har_pred = rng.normal(size=50)
    zero = np.zeros(50)
    assert diagnostics.residual_r2_oos(y, har_pred, resid_pred=zero) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# relative_r2_vs_har                                                          #
# --------------------------------------------------------------------------- #
def test_relative_r2_vs_har_model_equals_har_is_zero():
    rng = np.random.default_rng(7)
    y = rng.normal(size=60)
    har_pred = rng.normal(size=60)
    assert diagnostics.relative_r2_vs_har(y, har_pred.copy(), har_pred) == pytest.approx(0.0)


def test_relative_r2_vs_har_perfect_model_is_one():
    rng = np.random.default_rng(8)
    y = rng.normal(size=60)
    har_pred = rng.normal(size=60)
    assert diagnostics.relative_r2_vs_har(y, y.copy(), har_pred) == pytest.approx(1.0)
