"""Tests for the regime-conditional blend: pure-function correctness, causality, per-fold isolation,
and a real-data-slice smoke run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import regime_blend as RB  # noqa: E402


def _synth_panel(seed: int = 0) -> pd.DataFrame:
    """Two-fold panel with val+test where HAR-X is better on a high-vol regime and deep on calm."""
    rng = np.random.default_rng(seed)
    rows = []
    for fold in (0, 1):
        for split in ("val", "test"):
            n = 400
            y = np.exp(rng.normal(-8, 1.2, n))                      # positive realized variance
            harx = y * np.exp(rng.normal(0.0, 0.3, n))              # HAR-X: unbiased-ish
            deep = y * np.exp(rng.normal(0.15, 0.3, n))             # deep: mild over-forecast
            rows.append(pd.DataFrame({"split": split, "fold": fold,
                                      "ticker": "T", "date": pd.RangeIndex(n).astype(str),
                                      "y": y, "harx": harx, "deep": deep}))
    return pd.concat(rows, ignore_index=True)


def test_sigmoid_bounds_and_monotone():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    s = RB.sigmoid(x)
    assert np.all((s > 0) & (s < 1))
    assert np.all(np.diff(s) > 0)
    assert abs(RB.sigmoid(np.array([0.0]))[0] - 0.5) < 1e-9


def test_blend_recovers_endpoints():
    deep = np.array([2.0, 3.0]); harx = np.array([1.0, 1.0])
    assert np.allclose(RB.blend_forecast(deep, harx, np.ones(2)), deep)     # w=1 -> deep
    assert np.allclose(RB.blend_forecast(deep, harx, np.zeros(2)), harx)    # w=0 -> harx
    assert np.allclose(RB.blend_forecast(deep, harx, np.full(2, 0.5)), (deep + harx) / 2)


def test_features_are_causal_columns_only():
    """Gate features depend ONLY on harx/deep forecasts, never on realized y (no target leakage)."""
    df = pd.DataFrame({"harx": [1e-3, 2e-3], "deep": [2e-3, 1e-3], "y": [9.0, 9.0]})
    X = RB.features(df)
    df2 = df.copy(); df2["y"] = [0.0, 100.0]                                # change y only
    assert np.allclose(X, RB.features(df2))                                 # features unchanged
    assert X.shape == (2, 2)


def test_fit_fold_lowers_val_qlike_vs_deep_only():
    panel = _synth_panel()
    val = panel[(panel["fold"] == 0) & (panel["split"] == "val")]
    params = RB.fit_fold(val)
    bl, w = RB.apply_fold(val, params)
    y = val["y"].to_numpy()
    q_blend = RB.M.qlike(y, bl, floor=RB.FLOOR)
    q_deep = RB.M.qlike(y, val["deep"].to_numpy(), floor=RB.FLOOR)
    assert q_blend <= q_deep + 1e-9                                         # fit cannot do worse on val
    assert np.all((w >= 0) & (w <= 1))
    assert len(params["theta"]) == 3 and len(params["mu"]) == 2


def test_apply_fold_uses_given_standardization():
    panel = _synth_panel()
    val = panel[(panel["fold"] == 0) & (panel["split"] == "val")]
    params = RB.fit_fold(val)
    bl_a, _ = RB.apply_fold(val, params)
    params2 = dict(params, mu=[0.0, 0.0], sd=[1.0, 1.0])
    bl_b, _ = RB.apply_fold(val, params2)
    assert not np.allclose(bl_a, bl_b)                                      # standardization matters


def test_walkforward_fits_per_fold():
    panel = _synth_panel()
    scored = RB.blend_walkforward(panel)
    assert set(scored["split"]) == {"val", "test"}
    assert "blend" in scored and "w" in scored
    thetas = scored.groupby("fold")["theta"].first()
    assert thetas.nunique() == 2                                            # two folds -> two independent fits
    assert scored["w"].between(0, 1).all()


def test_walkforward_thin_val_falls_back_to_deep():
    panel = _synth_panel()
    thin = panel[(panel["fold"] == 0)].copy()
    thin = thin[~((thin["fold"] == 0) & (thin["split"] == "val"))].head(5)  # <10 val rows in this fold
    thin["split"] = "val"; thin["fold"] = 9
    scored = RB.blend_walkforward(thin)
    assert np.allclose(scored["w"], 1.0)                                    # fallback keeps deep
    assert np.allclose(scored["blend"].to_numpy(), scored["deep"].to_numpy())


def test_load_folds_missing_returns_none():
    assert RB.load_folds("no_such_market", 1) is None


def test_evaluate_missing_returns_none():
    assert RB.evaluate("no_such_market", 1) is None


def test_walkforward_skips_empty_split():
    """A fold with validation rows but no test rows exercises the empty-split continue."""
    panel = _synth_panel()
    only_val = panel[(panel["fold"] == 0) & (panel["split"] == "val")].copy()
    scored = RB.blend_walkforward(only_val)
    assert set(scored["split"]) == {"val"} and not scored.empty


def test_verdict_win_tie_loss():
    from run_regime_blend import verdict
    assert verdict(0.40, 0.50, 0.01) == "WIN*"               # lower + significant
    assert verdict(0.40, 0.50, 0.20) == "win "               # lower, not significant
    assert verdict(0.5010, 0.5000, 0.9) == "tie "            # higher but within 0.5%
    assert verdict(0.60, 0.50, 0.9) == "LOSS"                # higher, beyond 0.5%


@pytest.mark.smoke
def test_evaluate_real_data_slice():
    """Real-data smoke: if VN30 cells are present, evaluate() returns finite metrics and sane structure."""
    if RB.load_folds("vn30", 1, "LSTM") is None:
        pytest.skip("vn30 cells not present")
    r = RB.evaluate("vn30", 1, "LSTM")
    for who in ("deep", "harx", "blend"):
        m = r["test_metrics"][who]
        assert all(np.isfinite(v) for v in m.values())
        assert m["qlike"] > 0
    assert 0.0 <= r["mean_w"] <= 1.0
    assert 0.0 <= r["dm_blend_vs_harx"]["p"] <= 1.0
    # NB: the directional "blend beats HAR-X" outcome is an empirical result, verified in the report/driver,
    # NOT asserted here -- a data-dependent number must not turn the code gate red on a finding.
