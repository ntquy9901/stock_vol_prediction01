"""Tests for market_index.future_targets (strictly-future index targets, causal)."""
import numpy as np
import pandas as pd

import market_index


def _synth_index(n=30, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.standard_normal(n))
    df = pd.DataFrame({"date": dates, "close": close, "volume": rng.integers(1_000, 5_000, n)})
    df["lr"] = np.log(df["close"] / df["close"].shift(1))
    return df


def test_future_targets_matches_handcomputed_std():
    idx = _synth_index()
    L = 5
    d0 = idx["date"].iloc[10]                     # a window-end date that exists in the index
    T = market_index.future_targets(idx, [d0], L)
    p = int(np.searchsorted(idx["date"].to_numpy(), np.datetime64(d0), side="left"))
    block = idx["lr"].to_numpy()[p:p + L]
    assert T.loc[d0, "idx_vol"] == float(np.std(block, ddof=1))
    assert T.loc[d0, "idx_ret"] == float(np.mean(block))
    # target_end is the date of row p+L-1 (L-1 index days ahead of d0)
    assert T.loc[d0, "target_end"] == idx["date"].to_numpy()[p + L - 1]


def test_future_targets_lnvol_positive_only():
    idx = _synth_index()
    idx.loc[12, "volume"] = 0                     # a non-positive volume must be excluded from log-mean
    L = 4
    d0 = idx["date"].iloc[10]
    T = market_index.future_targets(idx, [d0], L)
    p = int(np.searchsorted(idx["date"].to_numpy(), np.datetime64(d0), side="left"))
    vol = idx["volume"].to_numpy(float)[p:p + L]
    pos = vol[vol > 0]
    assert T.loc[d0, "idx_lnvol"] == float(np.mean(np.log(pos)))


def test_future_targets_all_volume_nan_gives_nan_lnvol():
    idx = _synth_index()
    idx["volume"] = np.nan                        # no positive volume in the block -> NaN lnvol
    d0 = idx["date"].iloc[5]
    T = market_index.future_targets(idx, [d0], 4)
    assert np.isnan(T.loc[d0, "idx_lnvol"])
    assert np.isfinite(T.loc[d0, "idx_vol"])      # vol target still computed from returns


def test_future_targets_nan_when_insufficient_future():
    idx = _synth_index(n=30)
    d0 = idx["date"].iloc[28]                      # only 2 rows remain, need L=5 -> NaN
    T = market_index.future_targets(idx, [d0], 5)
    assert np.isnan(T.loc[d0, "idx_vol"])
    assert pd.isna(T.loc[d0, "target_end"])


def test_load_index_reads_real_hose():
    # smoke over the real VNINDEX file shape (date/close/volume + lr)
    df = market_index.load_index("hose")
    assert list(df.columns) == ["date", "close", "volume", "lr"]
    assert df["date"].is_monotonic_increasing
    assert np.isnan(df["lr"].iloc[0])             # first log-return undefined
