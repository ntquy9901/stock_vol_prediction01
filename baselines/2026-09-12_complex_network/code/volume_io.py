"""Real ``ln(volume)`` panels from the raw OHLCV (Refinement 1: replaces ``volume_zscore_22``).

The paper's volume correlation matrix uses ``ln(V_i(t))``, its exact transform. The enriched frames dropped
raw volume (keeping only ``volume_zscore_22``), so the raw daily volume is loaded from the market's raw OHLCV
directory and log-transformed. Non-positive volume becomes NaN (``ln`` undefined), handled pairwise by the
downstream window correlation.
"""
import numpy as np
import pandas as pd

import config


def load_log_volume(market, tickers):
    """DataFrame indexed by date, columns = the given ``tickers``, values = ``ln(volume)`` from raw OHLCV.

    Reads ``<REPO>/<RAW_VOL_DIR[market]>/<TK>_ohlcv.csv`` (columns ``date,open,high,low,close,volume``) for
    each ticker that has a file; ``volume <= 0`` becomes NaN so the log is defined. Tickers without a raw
    file are skipped; an empty result (no files at all) returns an empty DataFrame.
    """
    base = config.REPO / config.RAW_VOL_DIR[market]
    cols = {}
    for tk in tickers:
        p = base / f"{tk}_ohlcv.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, parse_dates=["date"])[["date", "volume"]].sort_values("date")
        v = d["volume"].to_numpy(float)
        v = np.where(v > 0, v, np.nan)                 # non-positive -> NaN so ln(volume) is defined
        cols[tk] = pd.Series(np.log(v), index=pd.DatetimeIndex(d["date"]))
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()
