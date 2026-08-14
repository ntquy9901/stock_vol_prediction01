"""Daily Parkinson volatility (plan section 3) and per-ticker feature construction.

Explicit target semantics:
- ``pk_var`` = (ln(High/Low))^2 / (4 ln 2)  -> Parkinson VARIANCE (project target).
- ``pk_vol`` = sqrt(pk_var)                  -> Parkinson VOLATILITY (plan default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FOUR_LN2 = 4.0 * np.log(2.0)


def parkinson_var(high: pd.Series | np.ndarray, low: pd.Series | np.ndarray):
    """Parkinson variance sigma^2 = (ln(H/L))^2 / (4 ln 2). Non-negative by construction."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):  # nonpositive low -> inf, sanitised by caller
        return (np.log(high / low) ** 2) / _FOUR_LN2


def parkinson_vol(high, low):
    """Parkinson volatility = sqrt(parkinson_var)."""
    return np.sqrt(parkinson_var(high, low))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker leakage-safe daily features (returns, Parkinson, volume transforms).

    All rolling/lagged features use information available at or before date t only.
    Input must be one ticker, sorted ascending by date.
    """
    d = df.sort_values("date").copy()
    # sanitise inf from any nonpositive-price row (mirrors _pk_var_wide) so the serialised
    # feature panel never carries inf into a downstream model
    d["pk_var"] = pd.Series(parkinson_var(d["high"], d["low"]), index=d.index).replace(
        [np.inf, -np.inf], np.nan
    )
    d["pk_vol"] = np.sqrt(d["pk_var"])
    d["log_return_1d"] = np.log(d["close"] / d["close"].shift(1))
    d["log_volume"] = np.log1p(d["volume"])
    d["log_volume_change"] = d["log_volume"].diff()
    mu20 = d["log_volume"].rolling(20).mean()
    sd20 = d["log_volume"].rolling(20).std()
    d["volume_zscore_20"] = (d["log_volume"] - mu20) / sd20.replace(0.0, np.nan)
    # trailing (past-only) HAR-style aggregates of Parkinson variance
    d["pk_lag1"] = d["pk_var"].shift(1)
    d["pk_mean_5"] = d["pk_var"].rolling(5).mean()
    d["pk_mean_22"] = d["pk_var"].rolling(22).mean()
    return d
