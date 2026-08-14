"""Nonlinear edge discovery: Gradient-Boosting ΔQLIKE (catches tail/interaction edges) + MI-residual.

The linear (ridge) test found STOP. This checks whether a NONLINEAR relationship — where only the
tail of source i drives target j's future volatility — is being missed. The synthetic plants exactly
such a tail edge that a linear model underrates but gradient boosting should recover.
"""

import numpy as np
import pandas as pd
import pytest

from graph_eda import edge_discovery as ed
from graph_eda import edge_nonlinear as en
from graph_eda import run_edge_nonlinear as rn


def _tail_panels(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2014-01-01", periods=n, freq="B")
    market = np.abs(rng.normal(1.0, 0.3, n)) * 1e-3 + 1e-4
    s = np.abs(rng.normal(0, 1, n)) * 6e-4 + market
    thr = np.quantile(s, 0.8)
    tail = (s > thr).astype(float)                          # only the top-20% of S matters
    noise = rng.normal(0, 1, n) * 1.2e-4
    t = np.full(n, np.nan)
    t[5:] = market[5:] + 1.6 * tail[:-5] * (s[:-5] - market[:-5]) + noise[5:]   # S tail -> T(t+5)
    t[:5] = market[:5] + noise[:5]
    nrel = np.abs(rng.normal(0, 1, n)) * 6e-4 + market
    pk = pd.DataFrame({"S": s, "T": np.abs(t), "N": nrel}, index=idx)
    mkt = pd.Series(market, index=idx)
    volz = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=["S", "T", "N"])
    return pk, mkt, volz


def _splits(n):
    a = np.arange(n)
    return a < int(n * 0.7), (a >= int(n * 0.7)) & (a < int(n * 0.85)), a >= int(n * 0.85)


def test_gb_recovers_nonlinear_tail_edge_that_ridge_underrates():
    pk, market, volz = _tail_panels()
    train, val, test = _splits(len(pk))
    gb = en.predictive_dqlike_edges_gb(pk, market, volz, train, val, test, horizon=5)
    st = gb[(gb.source == "S") & (gb.target == "T")].iloc[0]
    nt = gb[(gb.source == "N") & (gb.target == "T")].iloc[0]
    # GB recovers the planted tail edge S->T out of sample; the noise source does not
    assert st["dqlike_test"] > 0 and bool(st["stable"])
    assert st["dqlike_test"] > nt["dqlike_test"]
    # and GB beats the linear ridge on this nonlinear edge (sanity: nonlinearity is the point)
    lin = ed.predictive_dqlike_edges(pk, market, volz, train, val, test, horizon=5)
    lin_st = lin[(lin.source == "S") & (lin.target == "T")].iloc[0]
    assert st["dqlike_test"] >= lin_st["dqlike_test"]


def test_mi_residual_flags_dependence_above_null():
    pk, market, volz = _tail_panels()
    train, _, _ = _splits(len(pk))
    mi = en.mi_residual(pk, market, volz, train, horizon=5, n_shuffle=30, seed=0)
    st = mi[(mi.source == "S") & (mi.target == "T")].iloc[0]
    nt = mi[(mi.source == "N") & (mi.target == "T")].iloc[0]
    assert st["mi"] > st["mi_null_p95"]          # real tail edge exceeds its shuffle null
    assert st["mi"] > nt["mi"]                    # and exceeds the noise source


def test_mi_summary_empty_and_nonempty():
    assert rn._mi_summary(pd.DataFrame())["n_mi_pairs"] == 0
    df = pd.DataFrame({"mi": [0.10, 0.02], "mi_null_p95": [0.05, 0.05]})
    s = rn._mi_summary(df)
    assert s["n_mi_pairs"] == 2 and s["n_mi_above_null"] == 1


@pytest.mark.smoke
def test_run_edge_nonlinear_smoke(tmp_path, monkeypatch):
    # swap GB for the fast ridge edge fn and stub MI so the runner smoke stays quick
    monkeypatch.setattr(rn.en, "predictive_dqlike_edges_gb",
                        lambda *a, **k: ed.predictive_dqlike_edges(*a, **k))
    monkeypatch.setattr(rn.en, "mi_residual", lambda *a, **k: pd.DataFrame(
        {"source": ["S"], "target": ["T"], "mi": [0.1], "n": [100],
         "mi_null_mean": [0.05], "mi_null_p95": [0.08]}))
    out = rn.main(out_base=tmp_path)
    assert out["gb_summary"]["verdict"] in {"BUILD", "MAYBE", "STOP"}
    assert out["mi_summary"]["n_mi_pairs"] == 1
    for p in ("tables/edge_gb_dqlike.csv", "reports/EDGE_NONLINEAR_REPORT.md",
              "figures/edge_gb_dqlike_matrix.png"):
        assert (tmp_path / p).exists()


def test_mi_residual_empty_on_short_series():
    pk, market, volz = _tail_panels(n=60)      # every pair fails the min-rows guard
    train, _, _ = _splits(len(pk))
    mi = en.mi_residual(pk, market, volz, train, horizon=5, n_shuffle=2)
    assert mi.empty
