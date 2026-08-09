"""HAR-RV-X range / overnight variance estimators from daily OHLC (C4 node features).

All estimators are expressed in **variance (sigma^2)** units, matching the Parkinson-variance target
and the existing 3 HAR features (the plan's units trap). Sources: Garman & Klass (1980),
Rogers & Satchell (1991), and the close-to-open overnight variance term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_2LN2_MINUS_1 = 2.0 * np.log(2.0) - 1.0


def garman_klass_variance(open_, high, low, close):
    """GK daily variance: 0.5*(ln(H/L))^2 - (2ln2-1)*(ln(C/O))^2 (sigma^2 units)."""

    hl = np.log(np.asarray(high, dtype=float) / np.asarray(low, dtype=float))
    co = np.log(np.asarray(close, dtype=float) / np.asarray(open_, dtype=float))
    return 0.5 * hl ** 2 - _2LN2_MINUS_1 * co ** 2


def rogers_satchell_variance(open_, high, low, close):
    """RS daily variance: ln(H/C)ln(H/O) + ln(L/C)ln(L/O) (drift-independent, sigma^2 units)."""

    o = np.log(np.asarray(open_, dtype=float))
    h = np.log(np.asarray(high, dtype=float))
    low_log = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    return (h - c) * (h - o) + (low_log - c) * (low_log - o)


def overnight_variance(open_, prev_close):
    """Close-to-open overnight variance: (ln(O_t / C_{t-1}))^2 (sigma^2 units, >= 0)."""

    return np.log(np.asarray(open_, dtype=float) / np.asarray(prev_close, dtype=float)) ** 2


def compute_rvx_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Return date + gk_variance + rs_variance + overnight_variance for a daily OHLC frame.

    The first row's overnight term has no prior close and is dropped (NaN), so callers align the
    RV-X columns to the same trading dates as the Parkinson series before windowing.
    """

    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(ohlc.columns):
        raise ValueError(f"OHLC frame must contain {sorted(required)}")
    frame = ohlc.sort_values("date").reset_index(drop=True)
    gk = garman_klass_variance(frame["open"], frame["high"], frame["low"], frame["close"])
    rs = rogers_satchell_variance(frame["open"], frame["high"], frame["low"], frame["close"])
    overnight = overnight_variance(frame["open"], frame["close"].shift(1))
    out = pd.DataFrame({"date": frame["date"], "gk_variance": gk, "rs_variance": rs,
                        "overnight_variance": overnight})
    return out.dropna().reset_index(drop=True)
