"""Trading-calendar + topic/date helpers, vendored (trimmed) from
C:\\luanvan\\data_eda\\src\\eda\\phase04_news_eda.py (2026-07-25) — copy only, data_eda itself is
not modified. Trimmed to just the pieces `news_embeddings.py`/`dual_news_features.py` need:
`TOPIC_CATEGORIES`, `effective_trading_date`, `_trading_calendar`, `SOURCE_DAYFIRST`, `VN_TZ`,
`MARKET_CLOSE_HOUR`. Dropped: EDA report generation, sentiment/topic-extraction reporting,
publish-time bucketing — none of that is used by the dual-group embedding aggregation path.

`_trading_calendar()` is adapted to read this project's own price data
(`data/processed/{TICKER}_processed.csv`, column `date`) instead of data_eda's own OHLCV dir —
same role (union of trading dates across all tickers), different (already-existing) source file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vendor_config import PRICE_DATA_DIR, VN30_TICKERS

# DD/MM/YYYY sources (verified on real data); cafef + news_articles are ISO.
SOURCE_DAYFIRST = {"ssi_articles": True, "vndirect_articles": True}
MARKET_CLOSE_HOUR = 15  # VN afternoon close, UTC+7
VN_TZ = "Asia/Ho_Chi_Minh"

# Topic → 7 EDA-Guide categories, by Vietnamese/English keyword.
TOPIC_CATEGORIES = {
    "earnings": ["lợi nhuận", "doanh thu", "kết quả kinh doanh", "quý", "earnings", "ebitda"],
    "dividend": ["cổ tức", "chia thưởng", "dividend"],
    "ma": ["mua sáp nhập", "sáp nhập", "mua lại", "merger", "acquisition"],
    "management": ["ban lãnh đạo", "tổng giám đốc", "ceo", "management", "bổ nhiệm"],
    "regulation": ["quy định", "pháp lý", "kiểm soát", "regulation", "ubck"],
    "macro": ["lạm phát", "lãi suất", "gdp", "cpi", "fed", "macro", "ngân hàng nhà nước"],
    "sector": ["ngành", "sector", "chuỗi cung ứng", "hàng không", "ngân hàng", "bất động sản"],
}


def np_searchsorted(calendar, values):
    """First index in sorted ``calendar`` >= each value. Pure (testable)."""
    return np.searchsorted(calendar, values, side="left")


def effective_trading_date(news_dt: pd.Series, trading_dates, close_hour: int = MARKET_CLOSE_HOUR) -> pd.Series:
    """Map each news datetime -> its effective trading date.

    News before the market close on a trading day -> same day; after close or on a
    non-trading day -> rolled forward to the next available trading date.
    NaT inputs stay NaT (never silently mapped to the calendar tail).
    Returns tz-naive date series.
    """
    td = pd.to_datetime(pd.Series(trading_dates)).dt.tz_localize(None).dt.normalize().sort_values().drop_duplicates()
    td_arr = td.values.astype("datetime64[D]")
    if len(td_arr) == 0:
        return pd.Series(pd.NaT, index=news_dt.index)

    local = pd.to_datetime(news_dt, errors="coerce", utc=True).dt.tz_convert(VN_TZ)
    after_close = local.dt.hour >= close_hour
    eff_day = local.dt.normalize()
    eff_day = eff_day.where(~after_close, eff_day + pd.Timedelta(days=1))
    eff_arr = eff_day.dt.tz_localize(None).values.astype("datetime64[D]")

    idx = np_searchsorted(td_arr, eff_arr)
    idx = np.clip(idx, 0, len(td_arr) - 1)  # tail news -> last trading date
    result = pd.Series(pd.to_datetime(td_arr[idx]).normalize(), index=news_dt.index)
    return result.where(local.notna().values, pd.NaT)  # propagate NaT


def _trading_calendar() -> pd.Series:
    # A couple of processed CSVs (e.g. VRE, VPB) carry a "+07:00" (Asia/Ho_Chi_Minh local time,
    # midnight) offset in `date` while the rest are tz-naive; pooling both as-is into one
    # `sorted()` call raises (mixed tz-aware/tz-naive Timestamps aren't comparable). Since the
    # offset is already VN local time at midnight, dropping the tz label (not converting to UTC,
    # which would shift the calendar day back by 7 hours) recovers the same trading date.
    dates = set()
    for ticker in VN30_TICKERS:
        p = PRICE_DATA_DIR / f"{ticker}_processed.csv"
        if p.exists():
            parsed = pd.to_datetime(pd.read_csv(p, encoding="utf-8")["date"], errors="coerce").dropna()
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_localize(None)
            dates.update(parsed)
    return pd.Series(sorted(dates))
