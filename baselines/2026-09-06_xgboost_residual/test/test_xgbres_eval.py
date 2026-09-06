"""Pooled evaluation helpers: subset, lock share, top-1% exclusion, win rates, counts, shock trace, fit."""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))

import xgb_eval as E  # noqa: E402

_FL = 1e-8


def _pred():
    # keys (node, date); y_true, y_pred
    return {(0, "2025-04-10"): (0.02, 0.02), (1, "2025-04-10"): (1e-9, 0.5),   # lock cell (tiny target)
            (0, "2025-05-01"): (0.03, 0.031), (1, "2025-05-01"): (0.04, 0.038)}


def test_subset_inside_and_outside():
    p = _pred()
    keys = {(1, "2025-04-10")}
    assert set(E.subset(p, keys, True)) == keys
    assert (1, "2025-04-10") not in E.subset(p, keys, False)


def test_qlike_share_of_lock_cell_dominates():
    p = _pred()
    share = E.qlike_share(p, {(1, "2025-04-10")}, _FL)
    assert 0.9 < share <= 1.0                     # the tiny-target lock cell blows up QLIKE


def test_exclude_top_pct_dates_lowers_qlike():
    p = _pred()
    full = np.mean([__import__("metrics").per_obs_qlike(np.array([v[0]]), np.array([v[1]]), _FL)[0]
                    for v in p.values()])
    q_excl, n_excl, n_dates = E.exclude_top_pct_dates(p, _FL, 0.5)
    assert n_dates == 2 and n_excl == 1
    assert q_excl < full                           # dropping the worst date reduces mean QLIKE


def test_win_rate_vs():
    a = {(0, "d1"): (1.0, 1.0), (1, "d1"): (1.0, 1.0), (0, "d2"): (1.0, 1.0)}
    b = {(0, "d1"): (1.0, 2.0), (1, "d1"): (1.0, 2.0), (0, "d2"): (1.0, 2.0)}
    tw, dw = E.win_rate_vs(a, b, _FL)              # A exact, B off -> A wins everywhere
    assert tw == 1.0 and dw == 1.0


def test_count_summary_distinguishes_ticker_date_from_dates():
    ntd, nud, ntk = E.count_summary(_pred())
    assert ntd == 4 and nud == 2 and ntk == 2


def test_fit_metrics_keys():
    m = E.fit_metrics([0.02, 0.03], [0.02, 0.03], _FL)
    assert set(m) == {"mse", "qlike", "r2", "n"} and m["n"] == 2


def test_shock_month_diagnostics_flags_lock_fraction():
    p = _pred()
    d = E.shock_month_diagnostics(p, p, {(1, "2025-04-10")}, _FL, "2025-04")
    assert d["n_ticker_date"] == 2 and d["n_dates"] == 1
    assert d["n_lock_cells"] == 1 and abs(d["lock_fraction"] - 0.5) < 1e-9


def test_shock_month_empty_returns_zero_counts():
    d = E.shock_month_diagnostics(_pred(), _pred(), set(), _FL, "1999-01")
    assert d["n_ticker_date"] == 0


def test_empty_dict_guards():
    assert E.qlike_share({}, set(), _FL) is None
    assert E.exclude_top_pct_dates({}, _FL, 0.5) == (None, 0, 0)
    assert E.win_rate_vs({}, {}, _FL) == (None, None)


def test_qlike_share_zero_when_forecast_exact():
    exact = {(0, "d1"): (0.02, 0.02), (1, "d1"): (0.03, 0.03)}   # y==p -> total QLIKE 0
    assert E.qlike_share(exact, {(0, "d1")}, _FL) == 0.0


def test_exclude_all_dates_returns_none():
    p = {(0, "d1"): (0.02, 0.03)}                                 # 1 unique date, pct=1.0 removes it
    assert E.exclude_top_pct_dates(p, _FL, 1.0)[0] is None
