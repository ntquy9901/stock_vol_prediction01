"""Causality + formula + robustness tests for the graph-spillover block builder.

Proves the spillover features are leakage-safe (feature at t unchanged when future rows are perturbed),
match the neighbour-aggregation formula, degrade gracefully for isolated tickers, and never mutate the
caller's frame.
"""
import numpy as np
import pandas as pd

import spillover_features as SF


def _fold(pk_by_date):
    """Build a fold DataFrame from ``pk_by_date`` = {date_str: {ticker: parkinson_variance}}.

    Other neighbour-aggregated columns (mr_change, daily_return, volume_zscore_22) are set equal to
    parkinson_variance so the aggregation is easy to reason about; date/ticker index is preserved.
    """
    rows = []
    for d, tk_pk in pk_by_date.items():
        for tk, pk in tk_pk.items():
            rows.append({"date": pd.Timestamp(d), "ticker": tk, "parkinson_variance": pk,
                         "mr_change": pk, "daily_return": pk, "volume_zscore_22": pk})
    return pd.DataFrame(rows)


# tickers T0,T1,T2; each node's neighbours are the other two, equal weight (row-normalised)
_TICKERS = ["T0", "T1", "T2"]
_W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def test_spillover_aggregation_formula():
    """g_nb_vol == sum_j W[i,j]*pk_j(t); g_corr matches; g_node_minus_nb == pk - g_nb_vol."""
    fold = _fold({"2020-01-01": {"T0": 1.0, "T1": 2.0, "T2": 3.0}})
    out = SF.add_spillover_features(fold, _TICKERS, _W)
    by_tk = out.set_index("ticker")
    # T0 neighbours {T1=2,T2=3} @ 0.5 -> 2.5 ; T1 {1,3} -> 2.0 ; T2 {1,2} -> 1.5
    assert by_tk.loc["T0", "g_nb_vol"] == 2.5
    assert by_tk.loc["T1", "g_nb_vol"] == 2.0
    assert by_tk.loc["T2", "g_nb_vol"] == 1.5
    assert by_tk.loc["T0", "g_corr"] == 2.5                       # FM.nb agrees with graph_feats' nb_vol
    assert by_tk.loc["T0", "g_node_minus_nb"] == 1.0 - 2.5        # own minus neighbourhood
    assert by_tk.loc["T0", "g_nb_max"] == 3.0                     # max over {T1,T2}
    assert by_tk.loc["T0", "g_nb_disp"] == np.std([2.0, 3.0])     # population std over neighbours


def test_spillover_causal_future_invariant():
    """Perturbing a FUTURE date's values must not change any spillover feature at an earlier date."""
    base = {"2020-01-01": {"T0": 1.0, "T1": 2.0, "T2": 3.0},
            "2020-01-02": {"T0": 1.5, "T1": 2.5, "T2": 3.5},
            "2020-01-03": {"T0": 9.0, "T1": 8.0, "T2": 7.0}}
    out0 = SF.add_spillover_features(_fold(base), _TICKERS, _W)
    bumped = {k: dict(v) for k, v in base.items()}
    bumped["2020-01-03"] = {"T0": 500.0, "T1": 400.0, "T2": 300.0}   # perturb only the last date
    out1 = SF.add_spillover_features(_fold(bumped), _TICKERS, _W)
    cols = SF.SPILL_COLS + [SF.CORR_COL]
    early0 = out0[out0.date < pd.Timestamp("2020-01-03")].set_index(["date", "ticker"])[cols]
    early1 = out1[out1.date < pd.Timestamp("2020-01-03")].set_index(["date", "ticker"])[cols]
    pd.testing.assert_frame_equal(early0.sort_index(), early1.sort_index())
    # sanity: the perturbed (future) date's features DID change -> the test is not vacuous
    late0 = out0[out0.date == pd.Timestamp("2020-01-03")].set_index("ticker")["g_nb_vol"]
    late1 = out1[out1.date == pd.Timestamp("2020-01-03")].set_index("ticker")["g_nb_vol"]
    assert not np.allclose(late0.to_numpy(), late1.to_numpy())


def test_spillover_isolated_ticker_filled_zero():
    """A node with no neighbours (all-zero adjacency row) gets 0.0 spillover features (bounded fillna)."""
    W = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])   # T0 isolated
    fold = _fold({"2020-01-01": {"T0": 1.0, "T1": 2.0, "T2": 3.0}})
    out = SF.add_spillover_features(fold, _TICKERS, W).set_index("ticker")
    for c in ("g_nb_vol", "g_nb_shock", "g_nb_max", "g_nb_disp", "g_nb_ret", "g_nb_volshock"):
        assert out.loc["T0", c] == 0.0                            # no neighbours -> zero aggregate
    assert out.loc["T0", SF.CORR_COL] == 0.0
    assert out.loc["T0", "g_node_minus_nb"] == 1.0               # pk(1.0) - nb_vol(0.0), not NaN-filled


def test_spillover_does_not_mutate_input():
    """The builder returns a copy; the caller's frame gains no graph columns and keeps its values."""
    fold = _fold({"2020-01-01": {"T0": 1.0, "T1": 2.0, "T2": 3.0}})
    before = fold.copy(deep=True)
    _ = SF.add_spillover_features(fold, _TICKERS, _W)
    assert list(fold.columns) == list(before.columns)            # no g_* columns leaked back
    pd.testing.assert_frame_equal(fold, before)


def test_spill_cols_are_the_seven_graph_features():
    assert SF.SPILL_COLS == ["g_nb_vol", "g_nb_shock", "g_nb_max", "g_nb_disp",
                             "g_node_minus_nb", "g_nb_ret", "g_nb_volshock"]
