"""Edge-discovery EDA: supervised OOS ΔQLIKE directed edges + residual lead-lag + regime."""

import numpy as np
import pandas as pd
import pytest

from graph_eda import edge_discovery as ed
from graph_eda import run_edge_discovery


def _panels(n=900, seed=0):
    """Synthetic panel where source S genuinely drives target T's FUTURE pk, N does not.

    Returns pk_var (date x ticker), market, volz on the same index. h=5.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    market = np.abs(rng.normal(1.0, 0.3, n)) * 1e-3 + 1e-4          # market pk level
    s = np.abs(rng.normal(0, 1, n)) * 5e-4 + market                 # source pk
    noise_t = rng.normal(0, 1, n) * 1.5e-4
    t = np.full(n, np.nan)
    # T at k is driven by S at k-5 (beyond market) -> S(t) predicts T(t+5): a real S->T edge
    t[5:] = market[5:] + 0.8 * (s[:-5] - market[:-5]) + noise_t[5:]
    t[:5] = market[:5] + noise_t[:5]
    nrel = np.abs(rng.normal(0, 1, n)) * 5e-4 + market              # irrelevant source
    pk = pd.DataFrame({"S": s, "T": np.abs(t), "N": nrel}, index=idx)
    mkt = pd.Series(market, index=idx)
    volz = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=["S", "T", "N"])
    return pk, mkt, volz


def test_predictive_dqlike_finds_true_edge_not_noise():
    pk, market, volz = _panels()
    n = len(pk)
    train = np.arange(n) < int(n * 0.7)
    val = (np.arange(n) >= int(n * 0.7)) & (np.arange(n) < int(n * 0.85))
    test = np.arange(n) >= int(n * 0.85)
    edges = ed.predictive_dqlike_edges(pk, market, volz, train, val, test, horizon=5)
    st = edges[(edges.source == "S") & (edges.target == "T")].iloc[0]
    nt = edges[(edges.source == "N") & (edges.target == "T")].iloc[0]
    # the real source helps target T out of sample and is sign-stable; the noise source is not
    assert st["dqlike_test"] > 0 and bool(st["stable"])
    assert not (nt["dqlike_test"] > 0 and bool(nt["stable"])) or nt["dqlike_test"] < st["dqlike_test"]
    # no self edges, all ordered pairs present
    assert (edges.source != edges.target).all()


def test_residual_leadlag_removes_market_then_measures_direction():
    pk, market, volz = _panels()
    n = len(pk)
    train = np.arange(n) < int(n * 0.7)
    ll = ed.residual_leadlag(np.sqrt(pk), market, train, horizon=5)
    # directed matrix: S leads T (S residual today predicts T residual in +5) stronger than reverse
    assert ll.loc["S", "T"] > ll.loc["T", "S"]
    assert np.isclose(np.diag(ll.values), 1.0).sum() == 0  # diagonal is the own-lag, not forced to 1


def test_dqlike_fdr_and_regime_columns_present():
    pk, market, volz = _panels()
    n = len(pk)
    train = np.arange(n) < int(n * 0.7)
    val = (np.arange(n) >= int(n * 0.7)) & (np.arange(n) < int(n * 0.85))
    test = np.arange(n) >= int(n * 0.85)
    edges = ed.predictive_dqlike_edges(pk, market, volz, train, val, test, horizon=5, with_regime=True)
    for col in ("dqlike_val", "dqlike_test", "p_test", "fdr_q", "stable",
                "dqlike_test_lo_regime", "dqlike_test_hi_regime"):
        assert col in edges.columns


def test_predictive_dqlike_skips_short_series_returns_empty():
    # a <100-row panel makes every pair fail the min-train guard -> empty typed frame, no crash
    pk, market, volz = _panels(n=60)
    n = len(pk)
    train = np.arange(n) < int(n * 0.7)
    val = (np.arange(n) >= int(n * 0.7)) & (np.arange(n) < int(n * 0.85))
    test = np.arange(n) >= int(n * 0.85)
    edges = ed.predictive_dqlike_edges(pk, market, volz, train, val, test, horizon=5)
    assert edges.empty


def _fake_edges(n_valpos, n_testpos, v=0.02):
    """Synthetic edge table (all val-positive) with n_testpos of them test-positive, for _summarise."""
    dq_test = np.concatenate([np.full(n_testpos, v), np.full(n_valpos - n_testpos, -v)])
    return pd.DataFrame({
        "source": [f"S{i}" for i in range(n_valpos)], "target": ["T"] * n_valpos,
        "dqlike_val": np.full(n_valpos, 0.01),        # all val-positive (selected set)
        "dqlike_test": dq_test, "p_test": np.full(n_valpos, 0.5), "n_test": 100,
        "fdr_q": np.full(n_valpos, 0.5), "stable": dq_test > 0,
        "dqlike_test_hi_regime": dq_test, "dqlike_test_lo_regime": dq_test,
    })


def test_summarise_verdicts_from_leakage_safe_generalization():
    ll = pd.DataFrame([[1.0, 0.2], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
    assert run_edge_discovery._summarise(_fake_edges(30, 26), ll)["verdict"] == "BUILD"   # 87%, med>0
    assert run_edge_discovery._summarise(_fake_edges(30, 17), ll)["verdict"] == "MAYBE"   # 57%, med>0
    assert run_edge_discovery._summarise(_fake_edges(30, 12), ll)["verdict"] == "STOP"    # 40%, med<0


def test_summarise_handles_empty_edges():
    ll = pd.DataFrame([[1.0, 0.2], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
    s = run_edge_discovery._summarise(pd.DataFrame(), ll)   # empty table must not crash
    assert s["verdict"] == "STOP" and s["n_val_selected"] == 0


@pytest.mark.smoke
def test_run_edge_discovery_end_to_end(tmp_path):
    out = run_edge_discovery.main(out_base=tmp_path)
    s = out["summary"]
    assert s["verdict"] in {"BUILD", "MAYBE", "STOP"}
    assert (tmp_path / "tables" / "edge_predictive_dqlike.csv").exists()
    assert (tmp_path / "tables" / "residual_leadlag_h5.csv").exists()
    assert (tmp_path / "figures" / "edge_dqlike_matrix.png").exists()
    assert (tmp_path / "reports" / "EDGE_DISCOVERY_REPORT.md").exists()
    # verdict must be driven by the leakage-safe val->test generalization, not test-peeking counts
    assert 0.0 <= s["val_selected_test_positive_rate"] <= 1.0
    assert "generalization_binom_p" in s
