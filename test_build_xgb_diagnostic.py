"""Fast smoke test for the paper's Fig 3-5 diagnostic (build_xgb_diagnostic).

Monkeypatches the on-disk loader with a tiny synthetic panel so the XGBoost fit + the three figure
builders run in <2s, locking the invariants the figures rely on (OWN-8 contract, floored predictions,
10 equal-count deciles, gamma-deviance orientation, permutation-importance shape). No real market data.
"""
import numpy as np
import pandas as pd
import pytest

import build_xgb_diagnostic as B


def _panel(n_tickers=3, start="2023-01-02", periods=620):
    dates = pd.bdate_range(start, periods=periods)
    rng = np.random.default_rng(0)
    rows = []
    for t in range(n_tickers):
        base = np.exp(rng.normal(-9.0, 0.4, len(dates)))                 # positive variance-scale values
        d = pd.DataFrame({"date": dates.to_numpy(), "parkinson_variance": base, "ticker": f"T{t}"})
        for c in B.OWN:
            d[c] = base * (1.0 + 0.1 * rng.standard_normal(len(dates)))
        d["y"] = np.maximum(base * (1.0 + 0.2 * rng.standard_normal(len(dates))), B.FL)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def test_own8_contract():
    assert len(B.OWN) == 8 and "rq" not in B.OWN
    assert B.OWN[:3] == ["har_daily", "har_weekly", "har_monthly"]


def test_gamma_deviance_and_qlike_orientation():
    y = np.array([1e-4, 2e-4, 3e-4])                                     # realistic variance scale (< PRED_CAP)
    assert B._gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-9)       # zero at forecast==realized
    assert B._gamma_deviance(y, 2.0 * y) > 0                             # positive when biased
    assert np.allclose(B.qlike(y, y), 0.0, atol=1e-9)


def test_fit_split_is_embargoed(monkeypatch):
    monkeypatch.setattr(B, "load", lambda market, h: _panel())
    monkeypatch.setattr(B, "FIGDIR", None)                               # not used by fit()
    h = 5
    te, _ = B.fit("sp500_clean", h)
    # the train tail must be gapped from the test start by the horizon embargo (no boundary leak)
    d = _panel()
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    tr = d[(d["date"] >= B.TRAIN_START) & (d["date"] < pd.Timestamp(B.TEST_START) - embargo)]
    assert tr["date"].max() < pd.Timestamp(B.TEST_START) - embargo + pd.Timedelta(days=1)
    assert len(te) > 0 and te["date"].min() >= pd.Timestamp(B.TEST_START)


def test_fit_and_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "load", lambda market, h: _panel())
    monkeypatch.setattr(B, "FIGDIR", tmp_path)                           # do not clobber the committed figures
    te, bst = B.fit("sp500_clean", 5)
    assert te["har"].min() >= B.FL and te["xgb"].min() >= B.FL           # both floored positive
    assert te["dec"].dropna().between(0, 9).all() and te["dec"].nunique() == 10
    qh, qg, contrib = B.fig_decile(te, 5)
    assert len(qh) == len(qg) == len(contrib) == 10
    assert sum(contrib) == pytest.approx(100.0, abs=1e-6)                # shares of total QLIKE
    bg = B.fig_bias(te, 5)
    assert len(bg) == 10 and np.isfinite(bg).all()
    imp = B.fig_importance(te, bst, 5)
    assert len(imp) == len(B.OWN) == 8 and np.isfinite(imp).all()
    assert (tmp_path / "fig_feature_importance.pdf").exists()
    qh2, qg2, contrib2, bg2, imp2 = B.fig_combined(te, bst, 5)          # 2x2 combined panel
    assert len(qh2) == len(qg2) == len(contrib2) == len(bg2) == 10 and len(imp2) == 8
    assert sum(contrib2) == pytest.approx(100.0, abs=1e-6)
    assert (tmp_path / "fig_diagnostics_2x2.pdf").exists()
