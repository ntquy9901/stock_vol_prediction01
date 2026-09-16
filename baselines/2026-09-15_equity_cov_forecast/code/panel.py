"""Build a clean, aligned daily-return panel R[T x N] for a market, winsorized against split artifacts.

Reads the per-ticker enriched CSVs (`data/processed_enriched/<market>/<TICKER>.csv`, `daily_return` column),
selects the N_BASKET most-covered full-history tickers, and returns a wide date x ticker return matrix on the
intersection of trading days where every selected ticker has data (listwise-complete → a valid covariance
block). Winsorizes |return| to the per-market band to remove rare unadjusted-split jumps.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import cov_config as C

REPO = Path(__file__).resolve().parents[3]
_DIRS = {"hose": "hose", "sp500": "sp500_clean", "hnx": "hnx", "vn100": "vn100", "vn30": "vn30"}


def _market_dir(market: str) -> Path:
    return REPO / "data" / "processed_enriched" / _DIRS.get(market, market)


def _load_returns(market: str) -> dict[str, pd.Series]:
    """Load {ticker -> daily_return Series indexed by date} from the market's enriched CSVs, skipping the
    per-ticker `*_rejections.csv` audit files."""
    d = _market_dir(market)
    out = {}
    for f in sorted(d.glob("*.csv")):
        if f.name.endswith("_rejections.csv"):
            continue
        df = pd.read_csv(f, usecols=["date", "daily_return"], parse_dates=["date"])
        s = df.dropna(subset=["daily_return"]).set_index("date")["daily_return"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if len(s):
            out[f.stem] = s
    return out


def winsorize(R: pd.DataFrame, band: float) -> pd.DataFrame:
    """Clip every return to [-band, +band] (removes unadjusted-split / corporate-action jumps)."""
    return R.clip(lower=-band, upper=band)


def build_panel(market: str, n_basket: int | None = None, load_fn=None) -> pd.DataFrame:
    """Return a winsorized, listwise-complete return matrix R (index=date, columns=ticker) for the N most
    -covered full-history tickers over [TRAIN_START, end]. Columns sorted; no NaN."""
    n_basket = n_basket or C.N_BASKET
    series = (load_fn or _load_returns)(market)
    if not series:
        raise ValueError(f"no return series found for market {market!r} in {_market_dir(market)}")
    start = pd.Timestamp(C.TRAIN_START)
    # coverage = # trading days from TRAIN_START onward; rank tickers, keep the most-covered N.
    cov = {tk: int((s.index >= start).sum()) for tk, s in series.items()}
    max_cov = max(cov.values())
    eligible = [tk for tk, c in cov.items() if c >= C.MIN_COVERAGE * max_cov]
    chosen = sorted(sorted(eligible, key=lambda tk: cov[tk], reverse=True)[:n_basket])
    if len(chosen) < 2:
        raise ValueError(f"only {len(chosen)} eligible tickers for {market!r}; need >= 2")
    wide = pd.DataFrame({tk: series[tk] for tk in chosen})
    wide = wide[wide.index >= start].dropna(how="any").sort_index()
    band = C.WINSOR_BAND.get(market, C.WINSOR_BAND["default"])
    return winsorize(wide, band)
