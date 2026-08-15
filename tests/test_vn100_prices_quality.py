"""Data-quality checks for the crawled VN100 daily price files.

Target: ``data/raw/prices/vn100/<TICKER>_ohlcv.csv`` (VN100 constituent set,
columns ``date,open,high,low,close,volume``). Read-only on the data; this module
never writes to the CSV files.

These files were crawled from Yahoo Finance (``.VN`` suffix, daily,
``auto_adjust=True``) matching the convention of the existing 33 top-level raw
files: plain ``YYYY-MM-DD`` date, prices in thousands-VND (Yahoo full-VND / 1000),
volume unchanged. Unlike VPB/VRE at the top level, dates here are always plain
(never tz-aware).

The check taxonomy mirrors ``tests/test_raw_prices_quality.py``:
  * Schema, dates, OHLC positivity, non-negative volume and NaN/inf are HARD
    assertions.
  * OHLC-consistency (high>=low, high>=max(open,close), low<=min(open,close)) is
    a REPORTED diagnostic surfaced via ``pytest.xfail``: Yahoo's raw VN feed
    carries a small number of single-day point defects (e.g. a mis-scaled high
    that drops below the low) present in both adjusted and unadjusted modes, so
    they are genuine source glitches, not a processing error. They are flagged
    per ticker (with example rows) rather than blocking the suite; downstream
    cleaning can drop or repair the affected rows.
  * Leading synthetic backfill and coverage are REPORTED diagnostics; a leading
    synthetic run over ``LEADING_SYNTHETIC_THRESHOLD`` rows is surfaced via
    ``pytest.xfail`` so it stays visible in ``-v`` output.
  * A fetch-completeness check asserts that every attempted VN100 ticker except
    the known-unavailable ones produced a file.

Run with:
    python -m pytest tests/test_vn100_prices_quality.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices" / "vn100"
TOPLEVEL_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "2026-08-16_vn100_crawl_report.md"

EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
EXPECTED_LAST_DATE = "2026-08-14"
LEADING_SYNTHETIC_THRESHOLD = 20
SHORT_HISTORY_THRESHOLD = 750  # ~3 trading years; below this flagged in report
OHLC_RTOL = 1e-5

# VN100 constituent set attempted by the crawl: VN30 large-cap + VN70 mid-cap.
# Source: HOSE VN100 index composition (2025 basket), per assistant knowledge;
# the repo had no authoritative machine-readable VN100 list (the hardcoded list
# in src/data/crawl_vietnam_stocks.py contains invalid/delisted tickers such as
# XYZ, VDM and duplicates, so it was not used).
VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "LPB", "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]
VN70 = [
    "AAA", "ANV", "ASM", "BCG", "BFC", "BMP", "BSI", "BWE", "CII", "CMG",
    "CTD", "CTR", "CTS", "DBC", "DCM", "DGC", "DGW", "DIG", "DPM", "DSE",
    "DXG", "DXS", "EIB", "EVF", "FRT", "FTS", "GEE", "GEX", "GMD", "HAG",
    "HAH", "HCM", "HDC", "HDG", "HHV", "HSG", "HT1", "HVN", "IJC", "IMP",
    "KBC", "KDC", "KDH", "KOS", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN",
    "PC1", "PDR", "PHR", "PNJ", "PPC", "PTB", "PVD", "PVT", "REE", "SBT",
    "SCS", "SIP", "SJS", "SZC", "TCH", "TLG", "VCG", "VCI", "VGC", "VHC",
    "VIX", "VND", "VPI", "VSC",
]
VN100_ATTEMPTED = list(dict.fromkeys(VN30 + VN70))

# Tickers known to be unavailable on Yahoo Finance at crawl time (no data).
KNOWN_FETCH_FAILURES = {"BCG"}


def discover_tickers() -> list[str]:
    """Return sorted ticker symbols from crawled ``*_ohlcv.csv`` files."""
    suffix = "_ohlcv.csv"
    return sorted(p.name[: -len(suffix)] for p in PRICES_DIR.glob(f"*{suffix}"))


def discover_toplevel_tickers() -> list[str]:
    """Return the existing top-level raw ticker universe (the 33 files)."""
    suffix = "_ohlcv.csv"
    return sorted(p.name[: -len(suffix)] for p in TOPLEVEL_DIR.glob(f"*{suffix}"))


TICKERS = discover_tickers()


def _read_raw(ticker: str) -> pd.DataFrame:
    return pd.read_csv(PRICES_DIR / f"{ticker}_ohlcv.csv", dtype=str)


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker with parsed ``date_norm`` and numeric OHLCV columns."""
    df = _read_raw(ticker).copy()
    df["date_norm"] = pd.to_datetime(
        df["date"].astype(str).str.split(" ").str[0],
        format="%Y-%m-%d",
        errors="coerce",
    )
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@dataclass
class Diagnostics:
    ticker: str
    is_new: bool
    rows: int
    first_date: str
    last_date: str
    leading_synthetic_run: int
    first_real_date: str
    flat_frac: float
    zerovol_frac: float
    n_high_lt_low: int
    n_oc_outside: int
    ohlc_examples: str
    short_history: bool


def _leading_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    if mask.all():
        return int(mask.size)
    return int(np.argmax(~mask))


def ohlc_violation_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in PRICE_COLS)
    v = df["volume"].to_numpy(dtype=float)
    hi_oc = np.maximum(o, c)
    lo_oc = np.minimum(o, c)
    return {
        "nonpositive": ~((o > 0) & (h > 0) & (l > 0) & (c > 0)),
        "high_lt_low": h < l * (1 - OHLC_RTOL),
        "high_lt_max_oc": h < hi_oc * (1 - OHLC_RTOL),
        "low_gt_min_oc": l > lo_oc * (1 + OHLC_RTOL),
        "negative_volume": v < 0,
    }


def _high_lt_low_mask(df: pd.DataFrame) -> np.ndarray:
    """Parkinson-affecting defect: high < low (H/L drive Parkinson variance)."""
    return ohlc_violation_masks(df)["high_lt_low"]


def _oc_outside_mask(df: pd.DataFrame) -> np.ndarray:
    """Non-Parkinson defect: close/open falls outside [low, high]."""
    m = ohlc_violation_masks(df)
    return m["high_lt_max_oc"] | m["low_gt_min_oc"]


def compute_diagnostics(ticker: str, existing: set[str]) -> Diagnostics:
    df = load_prices(ticker)
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol = df["volume"].to_numpy()

    flat = high == low
    leading_flat = _leading_true_run(flat)
    if leading_flat < n:
        first_real = df["date_norm"].iloc[leading_flat].strftime("%Y-%m-%d")
    else:
        first_real = "N/A (all flat)"

    hl = _high_lt_low_mask(df)
    examples = [df["date"].iloc[i].split(" ")[0] for i in np.flatnonzero(hl)[:5]]

    return Diagnostics(
        ticker=ticker,
        is_new=ticker not in existing,
        rows=n,
        first_date=df["date_norm"].iloc[0].strftime("%Y-%m-%d"),
        last_date=df["date_norm"].iloc[-1].strftime("%Y-%m-%d"),
        leading_synthetic_run=leading_flat,
        first_real_date=first_real,
        flat_frac=float(np.mean(flat)),
        zerovol_frac=float(np.mean(vol == 0)),
        n_high_lt_low=int(hl.sum()),
        n_oc_outside=int(_oc_outside_mask(df).sum()),
        ohlc_examples=", ".join(examples),
        short_history=n < SHORT_HISTORY_THRESHOLD,
    )


# --------------------------------------------------------------------------- #
# Fetch completeness.
# --------------------------------------------------------------------------- #


def test_fetch_completeness():
    """Every attempted VN100 ticker except known-unavailable ones has a file."""
    present = set(TICKERS)
    expected = set(VN100_ATTEMPTED) - KNOWN_FETCH_FAILURES
    missing = sorted(expected - present)
    assert not missing, f"unexpected missing crawled tickers: {missing}"
    # Known failures must indeed be absent (else update the known-failure set).
    still_failing = sorted(KNOWN_FETCH_FAILURES & present)
    assert not still_failing, (
        f"tickers marked as known failures were actually fetched: {still_failing}"
    )


# --------------------------------------------------------------------------- #
# Checks 1-4: hard assertions, parametrized per ticker.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ticker", TICKERS)
def test_schema(ticker):
    """Check 1: exactly the 6 expected columns, in order."""
    cols = list(_read_raw(ticker).columns)
    assert cols == EXPECTED_COLUMNS, f"{ticker}: columns {cols} != {EXPECTED_COLUMNS}"


@pytest.mark.parametrize("ticker", TICKERS)
def test_dates_valid(ticker):
    """Check 2: dates parseable, strictly increasing, unique, all weekdays."""
    df = load_prices(ticker)
    dates = df["date_norm"]
    assert not dates.isna().any(), f"{ticker}: unparseable date(s) present"

    deltas = dates.diff().dropna()
    non_increasing = deltas[deltas <= pd.Timedelta(0)]
    assert non_increasing.empty, (
        f"{ticker}: dates not strictly increasing at "
        f"{[df['date'].iloc[i] for i in non_increasing.index[:5]]}"
    )
    assert dates.is_unique, f"{ticker}: duplicate date(s) present"

    weekend = dates[dates.dt.dayofweek >= 5]
    assert weekend.empty, (
        f"{ticker}: {len(weekend)} weekend date(s), e.g. "
        f"{weekend.dt.strftime('%Y-%m-%d').tolist()[:5]}"
    )


@pytest.mark.parametrize("ticker", TICKERS)
def test_ohlc_positivity(ticker):
    """Hard check: all OHLC strictly positive and volume non-negative."""
    df = load_prices(ticker)
    masks = ohlc_violation_masks(df)

    def _examples(mask):
        idx = np.flatnonzero(mask)[:5]
        return [df["date"].iloc[i].split(" ")[0] for i in idx]

    assert not masks["nonpositive"].any(), (
        f"{ticker}: {int(masks['nonpositive'].sum())} non-positive OHLC row(s), "
        f"e.g. {_examples(masks['nonpositive'])}"
    )
    assert not masks["negative_volume"].any(), f"{ticker}: negative volume present"


@pytest.mark.parametrize("ticker", TICKERS)
def test_ohlc_consistency(ticker):
    """Flag check: high>=low, high>=max(o,c), low<=min(o,c).

    Violations are Yahoo single-day point defects (present in both adjusted and
    unadjusted feeds), so they are surfaced via ``xfail`` for visibility rather
    than failing the suite; the report lists every affected ticker and row.
    """
    d = compute_diagnostics(ticker, set(discover_toplevel_tickers()))
    total = d.n_high_lt_low + d.n_oc_outside
    if total:
        pytest.xfail(
            f"{ticker}: {d.n_high_lt_low} high<low (Parkinson-affecting) + "
            f"{d.n_oc_outside} open/close-outside-[low,high] defect row(s) "
            f"(Yahoo source glitch), high<low e.g. {d.ohlc_examples or 'none'}"
        )
    assert total == 0


@pytest.mark.parametrize("ticker", TICKERS)
def test_no_nan_or_inf(ticker):
    """Check 4: no NaN and no inf in any OHLCV column."""
    df = load_prices(ticker)
    arr = df[NUMERIC_COLS].to_numpy(dtype=float)
    assert not np.isnan(arr).any(), f"{ticker}: NaN present in OHLCV"
    assert np.isfinite(arr).all(), f"{ticker}: inf present in OHLCV"


# --------------------------------------------------------------------------- #
# Check 5: synthetic-backfill visibility guard (xfail so it stays visible).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ticker", TICKERS)
def test_leading_synthetic_backfill_within_threshold(ticker):
    """Check 5 guard: leading synthetic (flat) run should be small."""
    d = compute_diagnostics(ticker, set(discover_toplevel_tickers()))
    if d.leading_synthetic_run > LEADING_SYNTHETIC_THRESHOLD:
        pytest.xfail(
            f"{ticker}: {d.leading_synthetic_run} leading flat rows; recommend "
            f"trimming to first real trading date {d.first_real_date}"
        )
    assert d.leading_synthetic_run <= LEADING_SYNTHETIC_THRESHOLD


# --------------------------------------------------------------------------- #
# Check 6 + report generation.
# --------------------------------------------------------------------------- #


def build_report_rows() -> list[Diagnostics]:
    existing = set(discover_toplevel_tickers())
    return [compute_diagnostics(t, existing) for t in TICKERS]


def render_report(rows: list[Diagnostics]) -> str:
    existing = set(discover_toplevel_tickers())
    overlap = sorted(set(VN100_ATTEMPTED) & existing)
    new = sorted(set(VN100_ATTEMPTED) - existing)
    failed = sorted(set(VN100_ATTEMPTED) - set(TICKERS))

    header = (
        "| ticker | new/overlap | rows | first_date | last_date | "
        "leading_flat_run | first_real_date | short_history | high<low | oc_outside | PASS |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {d.ticker} | {'NEW' if d.is_new else 'overlap'} | {d.rows} | "
        f"{d.first_date} | {d.last_date} | {d.leading_synthetic_run} | "
        f"{d.first_real_date} | {d.short_history} | {d.n_high_lt_low} | {d.n_oc_outside} | "
        f"{'PASS' if (d.n_high_lt_low == 0 and d.n_oc_outside == 0 and d.last_date == EXPECTED_LAST_DATE) else 'FLAG'} |\n"
        for d in rows
    )

    short = [d for d in rows if d.short_history]
    flagged = [d for d in rows if d.leading_synthetic_run > LEADING_SYNTHETIC_THRESHOLD]
    defect = [d for d in rows if d.n_high_lt_low]
    total_hl = sum(d.n_high_lt_low for d in rows)
    total_oc = sum(d.n_oc_outside for d in rows)
    off = [d.ticker for d in rows if d.last_date != EXPECTED_LAST_DATE]

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "# VN100 daily OHLCV crawl and data-quality report\n\n"
        f"Generated: {stamp}\n\n"
        "## Source and scope\n\n"
        "VN100 constituent set = VN30 (large-cap) + VN70 (mid-cap). Source: HOSE "
        "VN100 index composition (2025 basket), per assistant knowledge. No "
        "authoritative machine-readable VN100 list existed in the repository "
        "(the hardcoded list in `src/data/crawl_vietnam_stocks.py` contains "
        "invalid/delisted tickers such as XYZ, VDM and duplicates, so it was not "
        "used).\n\n"
        f"Attempted: {len(VN100_ATTEMPTED)} tickers. "
        f"Crawled successfully: {len(TICKERS)}. "
        f"Failed to fetch: {len(failed)} ({failed or 'none'}).\n\n"
        "Crawler: Yahoo Finance via yfinance, `.VN` suffix, `interval='1d'`, "
        "`start='2006-01-01'`, `auto_adjust=True` (split/dividend adjusted). "
        "Output matches the existing top-level convention: plain `YYYY-MM-DD` "
        "date, prices in thousands-VND (Yahoo full-VND / 1000), volume unchanged. "
        "Rows beyond 2026-08-14 (latest VN trading day) are excluded. Files under "
        "`data/raw/prices/vn100/`; the existing 33 top-level files were not "
        "modified.\n\n"
        "## New vs overlap\n\n"
        f"Overlap with existing top-level universe ({len(overlap)}): "
        f"{', '.join(overlap)}\n\n"
        f"New tickers not previously present ({len(new)}): "
        f"{', '.join(new)}\n\n"
        "## Per-ticker outcome\n\n" + header + body + "\n"
        "## Fetch failures\n\n"
        + (f"{', '.join(failed)}\n" if failed else "None.\n")
        + "\n## Short-history tickers (< "
        f"{SHORT_HISTORY_THRESHOLD} rows)\n\n"
        + (
            "| ticker | rows | first_date |\n|---|---|---|\n"
            + "".join(f"| {d.ticker} | {d.rows} | {d.first_date} |\n" for d in short)
            if short
            else "None.\n"
        )
        + "\n## OHLC-consistency defects (Yahoo source glitches, flagged)\n\n"
        "These are present in both adjusted and unadjusted Yahoo feeds, so they "
        "are genuine single-day source defects, not a processing error. Flagged "
        "via `xfail` for downstream cleaning; they do not block the test suite. "
        f"Relative tolerance {OHLC_RTOL:g}.\n\n"
        "Two categories:\n\n"
        f"1. **high < low** ({total_hl} rows across {len(defect)} tickers): "
        "Parkinson-affecting, since Parkinson variance uses only high/low. These "
        "must be dropped or repaired before computing volatility. Magnitudes range "
        "from ~1% gaps to gross defects (e.g. ACB/TPB 2025-06-04..05 where the "
        "adjusted high collapses to a near-zero value). Table below.\n"
        f"2. **open/close outside [low, high]** ({total_oc} rows): NOT "
        "Parkinson-affecting (high >= low still holds on these rows); they stem "
        "from dividend/split back-adjustment applied unevenly across O/H/L/C, the "
        "same artifact documented for the top-level files. Per-ticker counts are "
        "in the `oc_outside` column of the per-ticker table above.\n\n"
        "### high < low rows (Parkinson-affecting)\n\n"
        + (
            "| ticker | high<low_rows | example_dates |\n|---|---|---|\n"
            + "".join(
                f"| {d.ticker} | {d.n_high_lt_low} | {d.ohlc_examples} |\n"
                for d in sorted(defect, key=lambda x: -x.n_high_lt_low)
            )
            if defect
            else "None.\n"
        )
        + "\n## Leading synthetic backfill > "
        f"{LEADING_SYNTHETIC_THRESHOLD} rows\n\n"
        + (
            "| ticker | leading_flat_run | first_real_date |\n|---|---|---|\n"
            + "".join(
                f"| {d.ticker} | {d.leading_synthetic_run} | {d.first_real_date} |\n"
                for d in flagged
            )
            if flagged
            else "None.\n"
        )
        + "\n## Coverage\n\n"
        + (
            f"All {len(rows)} crawled tickers end on {EXPECTED_LAST_DATE}.\n"
            if not off
            else f"Tickers whose last_date != {EXPECTED_LAST_DATE}: {off}\n"
        )
    )


def test_generate_quality_report():
    """Check 6: build the per-ticker table and write the markdown report."""
    rows = build_report_rows()
    assert len(rows) == len(TICKERS)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(rows), encoding="utf-8")
    assert REPORT_PATH.exists()

    off = [d.ticker for d in rows if d.last_date != EXPECTED_LAST_DATE]
    if off:
        print(f"Tickers whose last_date != {EXPECTED_LAST_DATE}: {off}")
