"""Pure calendar-feature computation (requirements.md §3, design.md §1-2).

`compute_calendar_vector(date_str)` is a deterministic function of a date string only -- no
ticker, no I/O, no external data source. It is cheap enough (a handful of arithmetic ops) that it
is computed on the fly inside the dataset's sequence-building loop (design.md §2) rather than
cached to a parquet panel like the PhoBERT-derived news/macro panels are.

10 features, broadcast identically across every ticker for a given calendar date:
    dow_sin, dow_cos           -- day-of-week, cyclical (Mon..Fri trading week, period 5)
    month_sin, month_cos       -- month-of-year, cyclical (period 12)
    tet_proximity, in_tet_window   -- proximity to nearest Tet (Lunar New Year), see TET_DATES
    is_month_end, is_quarter_end   -- calendar-day proxy (NOT real trading-calendar last day)
    earnings_proximity, in_earnings_window -- proxy for VN quarterly-report season (20-day
                                               post-quarter-end disclosure deadline)

All assumptions (A1-A4) are documented in requirements.md §4 and are NOT verified against any
authoritative source beyond well-known public Tet dates -- flagged there for user/advisor
spot-check before trusting downstream results.
"""
from __future__ import annotations

import calendar as _calendar_module
import math
from datetime import date, timedelta

import numpy as np

CALENDAR_FEATURE_NAMES = [
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "tet_proximity", "in_tet_window",
    "is_month_end", "is_quarter_end",
    "earnings_proximity", "in_earnings_window",
]

# Vietnamese Lunar New Year (Tet), Gregorian dates -- hardcoded public historical facts
# (requirements.md A1), NOT computed from a lunar-calendar algorithm. Covers the full price-data
# date range (2006-11-21 .. 2026-06-09, see requirements.md) with a small margin on both ends.
TET_DATES: dict[int, str] = {
    2005: "2005-02-09", 2006: "2006-01-29", 2007: "2007-02-18", 2008: "2008-02-07",
    2009: "2009-01-26", 2010: "2010-02-14", 2011: "2011-02-03", 2012: "2012-01-23",
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19", 2016: "2016-02-08",
    2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
    2025: "2025-01-29", 2026: "2026-02-17", 2027: "2027-02-06",
}
_TET_DATE_OBJS = sorted(date.fromisoformat(s) for s in TET_DATES.values())

# VN quarterly-report disclosure deadline: within 20 calendar days of quarter-end (requirements.md
# A2 -- market-wide proxy, NOT a real per-ticker filing date).
_QUARTER_END_MONTH_DAY = ((3, 31), (6, 30), (9, 30), (12, 31))
_EARNINGS_DEADLINE_MONTH_DAY = ((1, 20), (4, 20), (7, 20), (10, 20))
_EARNINGS_WINDOW_DAYS = 20

_TET_DECAY_DAYS = 10.0
_TET_WINDOW_DAYS = 10
_EARNINGS_DECAY_DAYS = 10.0
_MONTH_END_LOOKBACK_DAYS = 3


def _parse_date(date_str: str) -> date:
    """Accept 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' / ISO-with-T (dataset dates sometimes carry
    a time component, see dataset_dual_news._norm_date's identical truncation-to-10-chars)."""
    s = str(date_str).strip()
    for sep in (" ", "T"):
        if sep in s:
            s = s.split(sep)[0]
            break
    return date.fromisoformat(s[:10])


def _nearest_tet_distance_days(d: date) -> int:
    return min(abs((d - t).days) for t in _TET_DATE_OBJS)


def _is_month_end(d: date) -> bool:
    days_in_month = _calendar_module.monthrange(d.year, d.month)[1]
    return d.day > days_in_month - _MONTH_END_LOOKBACK_DAYS


def _nearest_earnings_deadline_distance_days(d: date) -> int:
    deadlines = [
        date(y, month, day)
        for y in (d.year - 1, d.year, d.year + 1)
        for month, day in _EARNINGS_DEADLINE_MONTH_DAY
    ]
    return min(abs((d - dl).days) for dl in deadlines)


def _in_earnings_window(d: date) -> bool:
    for y in (d.year - 1, d.year):
        for month, day in _QUARTER_END_MONTH_DAY:
            q_end = date(y, month, day)
            deadline = q_end + timedelta(days=_EARNINGS_WINDOW_DAYS)
            if q_end < d <= deadline:
                return True
    return False


def compute_calendar_vector(date_str: str) -> np.ndarray:
    """date_str -> float32 vector of length len(CALENDAR_FEATURE_NAMES) (10)."""
    d = _parse_date(date_str)

    trading_dow = d.weekday() % 5  # Mon=0..Fri=4; weekend (5,6) falls back onto Mon/Tue's angle
    dow_sin = math.sin(2 * math.pi * trading_dow / 5)
    dow_cos = math.cos(2 * math.pi * trading_dow / 5)

    month_sin = math.sin(2 * math.pi * (d.month - 1) / 12)
    month_cos = math.cos(2 * math.pi * (d.month - 1) / 12)

    tet_dist = _nearest_tet_distance_days(d)
    tet_proximity = math.exp(-tet_dist / _TET_DECAY_DAYS)
    in_tet_window = 1.0 if tet_dist <= _TET_WINDOW_DAYS else 0.0

    month_end = _is_month_end(d)
    is_month_end = 1.0 if month_end else 0.0
    is_quarter_end = 1.0 if (month_end and d.month in (3, 6, 9, 12)) else 0.0

    earn_dist = _nearest_earnings_deadline_distance_days(d)
    earnings_proximity = math.exp(-earn_dist / _EARNINGS_DECAY_DAYS)
    in_earnings_window = 1.0 if _in_earnings_window(d) else 0.0

    return np.array([
        dow_sin, dow_cos, month_sin, month_cos,
        tet_proximity, in_tet_window,
        is_month_end, is_quarter_end,
        earnings_proximity, in_earnings_window,
    ], dtype=np.float32)


# Named subsets of CALENDAR_FEATURE_NAMES, used by train_calendar_news_gate.py's
# `--calendar_groups` ablation flag to isolate which feature GROUP (if any) carries signal,
# instead of only ever testing all 10 columns together (2026-08-01 user request).
CALENDAR_FEATURE_GROUPS: dict[str, list[str]] = {
    "all": list(CALENDAR_FEATURE_NAMES),
    "tet_only": ["tet_proximity", "in_tet_window"],
    "earnings_only": ["earnings_proximity", "in_earnings_window"],
    "generic_calendar": ["dow_sin", "dow_cos", "month_sin", "month_cos",
                         "is_month_end", "is_quarter_end"],
}
