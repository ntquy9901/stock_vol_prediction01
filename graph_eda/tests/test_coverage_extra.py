"""Direct unit tests for the remaining pure helpers (data quality, sectors, graph/rel).

Covers functions used by the orchestrator so the analysis library has full line
coverage independent of the end-to-end run.
"""

import numpy as np
import pandas as pd

from graph_eda import graphs
from graph_eda import relationships as rel
from graph_eda.data_quality import ticker_quality, universe_quality
from graph_eda.io_data import load_ticker
from graph_eda.parkinson import build_features
from graph_eda.sectors import SECTOR_MAP, same_sector_matrix, sector_of


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame(
        rng.standard_normal((n, 4)), index=idx, columns=list("ABCD")
    )


def test_ticker_quality_flags_bad_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 9.0, 13.0],   # row 2: high < low (bad)
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 10.5, 12.5],
            "volume": [100, 0, 50],       # one zero-volume row
            "ticker": ["X", "X", "X"],
        }
    )
    q = ticker_quality(df)
    assert q["ticker"] == "X"
    assert q["number_of_rows"] == 3
    assert q["high_below_low_count"] == 1
    assert q["zero_volume_count"] == 1
    assert q["duplicate_date_count"] == 0


def test_universe_quality_real_slice():
    frames = {"ACB": load_ticker("data/raw/prices/ACB_ohlcv.csv").head(200)}
    out = universe_quality(frames)
    assert list(out["ticker"]) == ["ACB"]
    assert out["nonpositive_price_count"].iloc[0] == 0


def test_build_features_columns():
    d = load_ticker("data/raw/prices/ACB_ohlcv.csv").head(60)
    f = build_features(d)
    for col in ["pk_var", "pk_vol", "log_return_1d", "log_volume",
                "volume_zscore_20", "pk_mean_5", "pk_mean_22"]:
        assert col in f.columns
    assert (f["pk_var"].dropna() >= 0).all()


def test_sectors_map_and_matrix():
    assert sector_of("VCB") == "Banks"
    assert sector_of("ZZZ") == "Unknown"
    assert len(SECTOR_MAP) == 33
    m = same_sector_matrix(["VCB", "ACB", "FPT"])
    assert m.loc["VCB", "ACB"] == 1     # both banks
    assert m.loc["VCB", "FPT"] == 0     # bank vs IT
    assert m.loc["VCB", "VCB"] == 0     # diagonal zeroed


def test_pair_long_table_fdr_and_shape():
    w = _toy()
    t = rel.pair_long_table(w, "pearson")
    assert len(t) == 4 * 3 // 2  # unique unordered pairs
    assert {"corr", "raw_p", "n_obs", "fdr_q", "significant_fdr_0.05"} <= set(t.columns)
    assert t["fdr_q"].dropna().between(0, 1).all()


def test_pair_long_table_spearman_and_short_series():
    w = _toy(n=8)  # < 10 obs -> corr/p NaN branch
    t = rel.pair_long_table(w, "spearman")
    assert t["corr"].isna().all()


def test_benjamini_hochberg_all_nan():
    q = rel.benjamini_hochberg(np.array([np.nan, np.nan]))
    assert np.isnan(q).all()


def test_market_factor_mean_and_median():
    w = _toy().abs()
    assert np.allclose(rel.market_factor(w, "mean").values, w.mean(axis=1).values)
    assert np.allclose(rel.market_factor(w, "median").values, w.median(axis=1).values)


def test_cross_lead_lag_matrix_shape():
    a = _toy(seed=1)
    b = _toy(seed=2)
    m = rel.cross_lead_lag_matrix(a, b, 1)
    assert m.shape == (4, 4)
    assert list(m.index) == list(a.columns)


def test_edge_stability_and_density_and_neighbor_stability():
    w = _toy(n=120)
    snaps = graphs.rolling_snapshots(w, window=30, step=10)
    assert len(snaps) >= 2
    stab = graphs.edge_stability(snaps)
    assert {"mean_corr", "std_corr", "sign_consistency"} <= set(stab.columns)
    assert any(c.startswith("persistence_") for c in stab.columns)
    dens = graphs.graph_density_series(snaps, tau=0.2)
    assert set(["n_edges", "density", "avg_degree"]) <= set(dens.columns)
    assert (dens["density"] >= 0).all() and (dens["density"] <= 1).all()
    ns = graphs.neighbor_stability(snaps, k=2)
    assert set(ns["node"]) == set(w.columns)


def test_neighbor_stability_single_snapshot_empty():
    w = _toy(n=30)
    snaps = graphs.rolling_snapshots(w, window=30, step=10)  # only 1 snapshot
    ns = graphs.neighbor_stability(snaps, k=2)
    assert ns.empty


def test_pair_long_table_spearman_full():
    t = rel.pair_long_table(_toy(n=40), "spearman")  # spearman path with enough obs
    assert t["corr"].notna().all()


def test_lead_lag_spearman_path():
    m = rel.lead_lag_matrix(_toy(n=60), k=1, method="spearman")
    assert m.shape == (4, 4)


def test_cross_corr_matrix_short_returns_nan():
    x = np.random.default_rng(0).standard_normal((5, 3))  # < 10 rows
    c = rel.cross_corr_matrix(x, x)
    assert np.isnan(c).all()


def test_metrics_single_point_diracc_nan():
    from graph_eda.predictive import _metrics

    m = _metrics(np.array([1.0]), np.array([1.1]))
    assert np.isnan(m["dir_acc"])


def test_edge_stability_empty_snaps():
    assert graphs.edge_stability([]).empty


def test_corr_matrix_bad_diagonal_raises():
    import pytest

    from graph_eda import leakage

    bad = pd.DataFrame([[0.9, 0.5], [0.5, 0.9]], index=["a", "b"], columns=["a", "b"])
    with pytest.raises(AssertionError):
        leakage.assert_corr_matrix_valid(bad)


def test_chrono_split_unknown_part_raises():
    import pytest

    from graph_eda.io_data import chrono_split

    idx = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=10, freq="D"))
    s = chrono_split(idx)
    with pytest.raises(ValueError):
        s.mask(idx, "holdout")


def test_run_baselines_insufficient_data_returns_empty():
    from graph_eda.predictive import run_baselines

    w = _toy(n=40).abs() + 0.1   # too few rows -> every baseline skipped (continue)
    n = len(w)
    tr = np.arange(n) < 30
    te = np.arange(n) >= 30
    res = run_baselines(
        w, market=w.median(axis=1), train_mask=tr, test_mask=te,
        train_corr=w.corr().abs(), horizon=1, k=1,
    )
    assert res.empty


def test_edge_stability_all_nan_pair_skipped():
    a = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]], index=["A", "B"], columns=["A", "B"]
    )
    snaps = [(pd.Timestamp("2021-01-01"), a), (pd.Timestamp("2021-02-01"), a)]
    stab = graphs.edge_stability(snaps)  # only A-B pair, all NaN -> skipped
    assert stab.empty
