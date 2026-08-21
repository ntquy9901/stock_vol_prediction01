"""TDD: panel-aware statistical comparison (plan section 18).

Covers date-clustered DM (cross-sectional dedup), circular block-bootstrap CI on the
date-aggregated loss differential, and the elimination Model Confidence Set.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "code"))

import stats  # noqa: E402


# --------------------------------------------------------------------------- #
# date_clustered_dm                                                           #
# --------------------------------------------------------------------------- #
def test_date_clustered_dm_dedups_cross_section():
    """Two tickers per date must collapse to one differential per unique date:
    n_dates == number of unique dates, not the inflated row count."""
    rng = np.random.default_rng(0)
    n_dates = 40
    dates = np.repeat(np.arange(n_dates), 2)          # 2 tickers/date -> 80 rows
    loss_a = rng.normal(1.0, 0.1, size=dates.size)
    loss_b = loss_a + 0.05                              # b uniformly worse
    res = stats.date_clustered_dm(loss_a, loss_b, dates, h=1)
    assert res["n_dates"] == n_dates
    assert res["mean_diff"] < 0                         # A has smaller loss -> favors A
    assert set(res) == {"dm_hln", "p_value", "mean_diff", "n_dates"}


def test_date_clustered_dm_matches_manual_aggregation():
    """date_clustered_dm on the panel equals a plain DM on hand-aggregated dates."""
    sys.path.insert(0, str(HERE.parents[2] / "submission" / "soict_lstm_gat"))
    import metrics  # noqa: E402

    dates = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    loss_a = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.0])
    loss_b = np.array([1.4, 1.5, 1.2, 1.0, 1.1, 1.3, 1.6, 1.4])
    a_by = np.array([loss_a[dates == d].mean() for d in np.unique(dates)])
    b_by = np.array([loss_b[dates == d].mean() for d in np.unique(dates)])
    manual = metrics.diebold_mariano(a_by, b_by, h=1)
    res = stats.date_clustered_dm(loss_a, loss_b, dates, h=1)
    assert res["n_dates"] == 4
    assert res["dm_hln"] == pytest.approx(manual.dm_hln)
    assert res["p_value"] == pytest.approx(manual.p_value)
    assert res["mean_diff"] == pytest.approx(manual.mean_diff)


# --------------------------------------------------------------------------- #
# block_bootstrap_ci                                                          #
# --------------------------------------------------------------------------- #
def test_block_bootstrap_ci_positive_constant_is_significant():
    n = 200
    dates = np.arange(n)
    loss_a = np.full(n, 5.0)
    loss_b = np.full(n, 3.0)                            # d = +2 everywhere
    out = stats.block_bootstrap_ci(loss_a, loss_b, dates, seed=1)
    assert out["mean_diff"] == pytest.approx(2.0)
    assert out["ci_low"] > 0.0
    assert out["significant"] is True


def test_block_bootstrap_ci_identical_losses_not_significant():
    n = 120
    dates = np.arange(n)
    loss = np.random.default_rng(3).normal(size=n)
    out = stats.block_bootstrap_ci(loss, loss.copy(), dates, seed=1)
    assert out["mean_diff"] == pytest.approx(0.0)
    assert out["ci_low"] <= 0.0 <= out["ci_high"]
    assert out["significant"] is False


def test_block_bootstrap_ci_is_deterministic():
    rng = np.random.default_rng(7)
    n = 150
    dates = np.arange(n)
    loss_a = rng.normal(1.0, 0.5, size=n)
    loss_b = rng.normal(0.9, 0.5, size=n)
    a = stats.block_bootstrap_ci(loss_a, loss_b, dates, seed=42)
    b = stats.block_bootstrap_ci(loss_a, loss_b, dates, seed=42)
    assert a["ci_low"] == b["ci_low"]
    assert a["ci_high"] == b["ci_high"]
    assert a["block"] == b["block"]


def test_block_bootstrap_ci_default_block_is_cube_root():
    n = 27
    dates = np.arange(n)
    loss = np.random.default_rng(0).normal(size=n)
    out = stats.block_bootstrap_ci(loss, loss + 0.1, dates, seed=0)
    assert out["block"] == 3                            # ceil(27 ** (1/3)) == 3


# --------------------------------------------------------------------------- #
# model_confidence_set                                                        #
# --------------------------------------------------------------------------- #
def test_mcs_keeps_dominant_model():
    rng = np.random.default_rng(11)
    n = 80
    dates = np.arange(n)
    losses = {
        "good": np.abs(rng.normal(0.0, 0.02, size=n)),   # tiny loss
        "mid": 0.5 + rng.normal(0.0, 0.02, size=n),
        "bad": 1.0 + rng.normal(0.0, 0.02, size=n),
    }
    out = stats.model_confidence_set(losses, dates, alpha=0.10, n_boot=500, seed=0)
    assert "good" in out["mcs_set"]
    assert "bad" not in out["mcs_set"]
    assert set(out["p_values"]) == {"good", "mid", "bad"}


def test_mcs_keeps_all_identical_models():
    rng = np.random.default_rng(5)
    n = 90
    dates = np.arange(n)
    shared = np.abs(rng.normal(1.0, 0.1, size=n))
    losses = {"m1": shared, "m2": shared.copy(), "m3": shared.copy()}
    out = stats.model_confidence_set(losses, dates, alpha=0.10, n_boot=400, seed=0)
    assert set(out["mcs_set"]) == {"m1", "m2", "m3"}


def test_mcs_is_deterministic():
    rng = np.random.default_rng(9)
    n = 70
    dates = np.arange(n)
    losses = {
        "a": np.abs(rng.normal(0.1, 0.05, size=n)),
        "b": np.abs(rng.normal(0.3, 0.05, size=n)),
    }
    o1 = stats.model_confidence_set(losses, dates, n_boot=300, seed=1)
    o2 = stats.model_confidence_set(losses, dates, n_boot=300, seed=1)
    assert o1["mcs_set"] == o2["mcs_set"]
    assert o1["p_values"] == o2["p_values"]
