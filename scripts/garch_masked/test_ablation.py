"""Tests for the output-parameterization ablation helpers (scripts/garch_masked/ablation_vn_5seed.py).

Covers the bias-match math (the fair-init constants), the metric aggregator, and the flatten/date
alignment used for the date-clustered DM inputs.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))
import ablation_vn_5seed as A  # noqa: E402


def _softplus(x):
    return np.logaddexp(0.0, x)


def test_bias_match_constants_start_at_mean_ratio():
    # exp link: bias 0 -> exp(0) == 1 (mean ratio)
    assert abs(np.exp(0.0) - 1.0) < 1e-12
    # softplus link: bias log(e-1) -> softplus(bias) == 1
    bias = float(np.log(np.expm1(1.0)))
    assert abs(_softplus(bias) - 1.0) < 1e-6
    # the un-matched softplus(0) is ~0.693, i.e. ~31% below the mean ratio (the handicap we correct)
    assert abs(_softplus(0.0) - 0.6931) < 1e-3


def test_metrics_vec_exact_and_consistent():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, 4.0])
    m = A._metrics_vec(y, p)
    assert abs(m["mse"]) < 1e-12 and abs(m["mae"]) < 1e-12
    assert abs(m["r2"] - 1.0) < 1e-12          # perfect fit
    # r2 consistent with 1 - sse/sst on a non-perfect prediction
    p2 = np.array([1.0, 2.0, 3.0, 5.0])
    m2 = A._metrics_vec(y, p2)
    sse = float(np.sum((y - p2) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    assert abs(m2["r2"] - (1.0 - sse / sst)) < 1e-12


def test_dm_pairs_guards_missing_configs():
    """_dm_pairs must not KeyError when a config subset omits C or D (Blind-Hunter #4 regression guard)."""
    def fake_dm(a, b):
        return {"p_value": 0.5, "mean_diff": 0.0, "favors": "B"}
    har = "HAR"
    assert set(A._dm_pairs({"D_ratio_softplus": "D"}, har, fake_dm)) == {"D_vs_HAR"}       # C_vs_D needs C too
    assert A._dm_pairs({"A_zscore_linear_floor": "A"}, har, fake_dm) == {}                  # neither -> no crash
    assert set(A._dm_pairs({"C_ratio_exp": "C", "D_ratio_softplus": "D"}, har, fake_dm)) == {"D_vs_HAR", "C_vs_D"}


def test_flat_aligns_dates_with_targets():
    # 2 anchor rows x 3 nodes; target mask picks specific cells; dates must line up with y flatten order
    tmask = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    y_te = np.array([[10.0, 99.0, 11.0], [99.0, 20.0, 21.0]])
    D = SimpleNamespace(tmask_te=tmask, y_te=y_te, d_te=["d1", "d2"])
    m, y, dates = A._flat(D)
    # valid cells row-major: (0,0),(0,2),(1,1),(1,2) -> y = 10,11,20,21 ; dates = d1,d1,d2,d2
    assert list(y) == [10.0, 11.0, 20.0, 21.0]
    assert list(dates) == ["d1", "d1", "d2", "d2"]
