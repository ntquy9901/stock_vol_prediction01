"""Per-ticker data-quality checks (plan section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ticker_quality(df: pd.DataFrame) -> dict:
    """Return the plan-section-5 quality summary for one ticker frame."""
    d = df.sort_values("date")
    dates = d["date"]
    hi, lo, op, cl, vol = d["high"], d["low"], d["open"], d["close"], d["volume"]
    row_max = np.maximum.reduce([op.values, cl.values, lo.values])
    row_min = np.minimum.reduce([op.values, cl.values, hi.values])
    return {
        "ticker": d["ticker"].iloc[0],
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
        "number_of_rows": int(len(d)),
        "duplicate_date_count": int(dates.duplicated().sum()),
        "missing_open": int(op.isna().sum()),
        "missing_high": int(hi.isna().sum()),
        "missing_low": int(lo.isna().sum()),
        "missing_close": int(cl.isna().sum()),
        "missing_volume": int(vol.isna().sum()),
        "zero_volume_count": int((vol == 0).sum()),
        "high_below_low_count": int((hi < lo).sum()),
        "high_below_max_oc_count": int((hi.values < row_max).sum()),
        "low_above_min_oc_count": int((lo.values > row_min).sum()),
        "nonpositive_price_count": int(((hi <= 0) | (lo <= 0) | (cl <= 0)).sum()),
        "negative_volume_count": int((vol < 0).sum()),
        "hl_equal_count": int((hi == lo).sum()),
    }


def universe_quality(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quality summary table over all tickers."""
    return pd.DataFrame([ticker_quality(d) for d in frames.values()])
