"""Market-index target series for Experiment A (predict the FUTURE index volatility).

Loads the real VNINDEX (HOSE) or ^GSPC (S&P 500) close series and, for each network window-end date ``d0``,
builds the strictly-future L-day targets from the index (paper Section 2.4, eqs. 2-4): average log return,
average log volume, and the standard deviation of log returns (the headline volatility target).
"""
import numpy as np
import pandas as pd

import config

_INDEX_FILE = {"hose": "vnindex.csv", "sp500": "gspc.csv"}


def load_index(market):
    """Return the index series ``date, close, volume`` sorted ascending, with daily log return ``lr``."""
    fn = _INDEX_FILE[market]
    path = config.REPO / "data" / "raw" / "prices" / "_market_index" / fn
    df = pd.read_csv(path, parse_dates=["date"])[["date", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)
    df["lr"] = np.log(df["close"] / df["close"].shift(1))
    return df


def future_targets(idx, d0_index, L):
    """Strictly-future L-day index targets for each window-end date ``d0``.

    For each ``d0`` in ``d0_index``: ``p`` = first index row with ``date >= d0`` (searchsorted); the future
    block is rows ``p .. p+L-1``. Emits NaN when fewer than ``L`` future rows remain (dropped downstream).
    ``idx_vol`` = std(lr, ddof=1) over the block (paper eq. 4, headline); ``idx_ret`` = mean(lr) (eq. 2);
    ``idx_lnvol`` = mean(log(volume)) over positive volumes (eq. 3, NaN if none). ``target_end`` = date of
    the last block row (used as the walk-forward embargo boundary).
    """
    dts = idx["date"].to_numpy()
    lr = idx["lr"].to_numpy(float)
    vol = idx["volume"].to_numpy(float)
    rows = {}
    for d0 in d0_index:
        p = int(np.searchsorted(dts, np.datetime64(d0), side="left"))
        if p + L > len(dts):
            rows[d0] = [np.nan, np.nan, np.nan, np.datetime64("NaT")]
            continue
        block_lr = lr[p:p + L]
        block_vol = vol[p:p + L]
        idx_vol = float(np.std(block_lr, ddof=1))
        idx_ret = float(np.mean(block_lr))
        pos = block_vol[np.isfinite(block_vol) & (block_vol > 0)]
        idx_lnvol = float(np.mean(np.log(pos))) if pos.size else np.nan
        rows[d0] = [idx_vol, idx_ret, idx_lnvol, dts[p + L - 1]]
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["idx_vol", "idx_ret", "idx_lnvol", "target_end"]
    )
