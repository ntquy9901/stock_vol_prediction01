"""Data-quality checks for the S&P500 daily price files (raw + processed).

Targets:
  * RAW: ``data/raw/prices/sp500/<TICKER>_ohlcv.csv`` (~500 US tickers, columns
    ``date,open,high,low,close,volume``; Yahoo Finance sourced, ``auto_adjust``).
  * PROCESSED: ``data/processed/sp500/<TICKER>_processed.csv`` (columns
    ``date,parkinson_volatility``; Parkinson variance = (ln(H/L)^2)/(4 ln2),
    dropna, upper-clipped at 0.1, per ``src/common/parkinson_utils.py``).

Read-only on the data; this module never writes to the CSV files (only the
markdown report under ``docs/reports/``).

Check taxonomy mirrors ``tests/test_raw_prices_quality.py`` /
``tests/test_vn100_prices_quality.py`` (adapted for a ~500-ticker US universe):

  RAW hard assertions (a violation fails the test):
    1. Schema -- exactly the 6 expected columns, in order.
    2. Dates  -- parseable, strictly increasing, unique, weekday-only. US
       holidays differ from the VN calendar, so the calendar itself is NOT
       enforced (only monotonic + no weekend + no duplicates).
    3. OHLC positivity + non-negative volume.
    4. No NaN / inf in any OHLCV column.

  RAW reported diagnostics (surfaced via ``pytest.xfail`` for visibility, not a
  hard fail, because these are documented Yahoo-feed glitch tolerances):
    5a. OHLC consistency: high >= low, open/close within [low, high]
        (rtol ``OHLC_RTOL``). Single-day Yahoo point defects.
    5b. Volume glitch: volume == 0 on a day the price moved (high != low).
    5c. Leading-flat synthetic backfill run over ``LEADING_SYNTHETIC_THRESHOLD``.
    6.  Coverage: per-ticker row counts; short histories (< ``SHORT_HISTORY_
        THRESHOLD`` rows) flagged in the report.

  PROCESSED hard assertions:
    P1. Schema -- exactly ``date,parkinson_volatility``, numeric values.
    P2. Dates  -- parseable, strictly increasing, unique, weekday-only.
    P3. Values -- finite, >= 0, and <= clip ceiling (0.1).

Run with (from repo root):
    PYTHONIOENCODING=utf-8 .venv_gpu_encode/Scripts/python.exe -m pytest \
        tests/test_sp500_prices_quality.py -q
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "prices" / "sp500"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "sp500"
REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports" / "2026-08-21_sp500_data_quality_report.md"
)

EXPECTED_RAW_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
EXPECTED_PROCESSED_COLUMNS = ["date", "parkinson_volatility"]
PRICE_COLS = ["open", "high", "low", "close"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]

EXPECTED_LAST_DATE = "2026-08-19"  # coverage expectation (diagnostic only)
CLIP_CEILING = 0.1  # upper clip applied in parkinson_utils.process_single_stock
MIN_EXPECTED_TICKERS = 500  # ~500-ticker S&P500 universe
LEADING_SYNTHETIC_THRESHOLD = 20
SHORT_HISTORY_THRESHOLD = 750  # ~3 trading years; below this flagged in report
# Relative tolerance for OHLC inequality checks -- absorbs float64 adjust noise
# while remaining orders below any genuine defect.
OHLC_RTOL = 1e-5


def _discover(directory: Path, suffix: str) -> list[str]:
    return sorted(p.name[: -len(suffix)] for p in directory.glob(f"*{suffix}"))


RAW_TICKERS = _discover(RAW_DIR, "_ohlcv.csv")
PROCESSED_TICKERS = _discover(PROCESSED_DIR, "_processed.csv")

# Read-only per-session caches so each CSV is parsed once.
_RAW_CACHE: dict[str, pd.DataFrame] = {}
_PROC_CACHE: dict[str, pd.DataFrame] = {}


def _read_raw(ticker: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{ticker}_ohlcv.csv", dtype=str)


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one raw ticker with parsed ``date_norm`` + numeric OHLCV columns."""
    if ticker not in _RAW_CACHE:
        df = _read_raw(ticker).copy()
        df["date_norm"] = pd.to_datetime(
            df["date"].astype(str).str.split(" ").str[0],
            format="%Y-%m-%d",
            errors="coerce",
        )
        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        _RAW_CACHE[ticker] = df
    return _RAW_CACHE[ticker]


def load_processed(ticker: str) -> pd.DataFrame:
    if ticker not in _PROC_CACHE:
        df = pd.read_csv(PROCESSED_DIR / f"{ticker}_processed.csv")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        _PROC_CACHE[ticker] = df
    return _PROC_CACHE[ticker]


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
        # Yahoo artifact: zero volume on a day the price moved (high != low).
        "volume_glitch": (v == 0) & (h != l),
    }


# --------------------------------------------------------------------------- #
# Diagnostics for the report.
# --------------------------------------------------------------------------- #


@dataclass
class RawDiag:
    ticker: str
    rows: int
    first_date: str
    last_date: str
    leading_synthetic_run: int
    first_real_date: str
    flat_frac: float
    zerovol_frac: float
    n_high_lt_low: int
    n_oc_outside: int
    n_volume_glitch: int
    n_nonpositive: int
    ohlc_examples: str
    short_history: bool


@dataclass
class ProcDiag:
    ticker: str
    rows: int
    first_date: str
    last_date: str
    zero_count: int
    zero_frac: float
    leading_zero_run: int
    n_negative: int
    n_nonfinite: int
    n_over_clip: int
    minimum: float
    median: float
    maximum: float


def compute_raw_diag(ticker: str) -> RawDiag:
    df = load_prices(ticker)
    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    flat = high == low
    leading_flat = _leading_true_run(flat)
    if leading_flat < n:
        first_real = df["date_norm"].iloc[leading_flat].strftime("%Y-%m-%d")
    else:
        first_real = "N/A (all flat)"

    masks = ohlc_violation_masks(df)
    hl = masks["high_lt_low"]
    oc_outside = masks["high_lt_max_oc"] | masks["low_gt_min_oc"]
    examples = [df["date"].iloc[i].split(" ")[0] for i in np.flatnonzero(hl)[:5]]

    return RawDiag(
        ticker=ticker,
        rows=n,
        first_date=df["date_norm"].iloc[0].strftime("%Y-%m-%d"),
        last_date=df["date_norm"].iloc[-1].strftime("%Y-%m-%d"),
        leading_synthetic_run=leading_flat,
        first_real_date=first_real,
        flat_frac=float(np.mean(flat)),
        zerovol_frac=float(np.mean(vol == 0)),
        n_high_lt_low=int(hl.sum()),
        n_oc_outside=int(oc_outside.sum()),
        n_volume_glitch=int(masks["volume_glitch"].sum()),
        n_nonpositive=int(masks["nonpositive"].sum()),
        ohlc_examples=", ".join(examples),
        short_history=n < SHORT_HISTORY_THRESHOLD,
    )


def compute_proc_diag(ticker: str) -> ProcDiag:
    df = load_processed(ticker)
    vol = df["parkinson_volatility"].to_numpy(dtype=float)
    is_zero = vol == 0.0

    leading = _leading_true_run(is_zero)
    finite = np.isfinite(vol)

    return ProcDiag(
        ticker=ticker,
        rows=len(df),
        first_date=str(df["date"].iloc[0].date()) if pd.notna(df["date"].iloc[0]) else "NaT",
        last_date=str(df["date"].iloc[-1].date()) if pd.notna(df["date"].iloc[-1]) else "NaT",
        zero_count=int(is_zero.sum()),
        zero_frac=float(is_zero.mean()),
        leading_zero_run=leading,
        n_negative=int((vol[finite] < 0).sum()),
        n_nonfinite=int((~finite).sum()),
        n_over_clip=int((vol[finite] > CLIP_CEILING + 1e-12).sum()),
        minimum=float(np.nanmin(vol)) if vol.size else float("nan"),
        median=float(np.nanmedian(vol)) if vol.size else float("nan"),
        maximum=float(np.nanmax(vol)) if vol.size else float("nan"),
    )


# =========================================================================
# Preconditions
# =========================================================================


def test_dirs_and_tickers_present():
    """Sanity: both dirs exist and ~500 tickers present, raw/processed aligned."""
    assert RAW_DIR.is_dir(), f"missing raw dir: {RAW_DIR}"
    assert PROCESSED_DIR.is_dir(), f"missing processed dir: {PROCESSED_DIR}"
    assert len(RAW_TICKERS) >= MIN_EXPECTED_TICKERS, (
        f"expected >= {MIN_EXPECTED_TICKERS} raw tickers, found {len(RAW_TICKERS)}"
    )
    missing_proc = sorted(set(RAW_TICKERS) - set(PROCESSED_TICKERS))
    assert not missing_proc, f"raw tickers without a processed file: {missing_proc[:10]}"


# =========================================================================
# RAW checks 1-4: hard assertions, parametrized per ticker.
# =========================================================================


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_schema(ticker):
    """Check 1: exactly the 6 expected columns, in order."""
    cols = list(_read_raw(ticker).columns)
    assert cols == EXPECTED_RAW_COLUMNS, f"{ticker}: columns {cols} != {EXPECTED_RAW_COLUMNS}"


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_dates_valid(ticker):
    """Check 2: dates parseable, strictly increasing, unique, weekday-only.

    US market holidays differ from the VN calendar, so only monotonic + unique +
    no-weekend are enforced (not a specific trading calendar).
    """
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


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_ohlc_positivity(ticker):
    """Check 3: all OHLC strictly positive and volume non-negative."""
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


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_no_nan_or_inf(ticker):
    """Check 4: no NaN and no inf in any OHLCV column."""
    df = load_prices(ticker)
    arr = df[NUMERIC_COLS].to_numpy(dtype=float)
    assert not np.isnan(arr).any(), f"{ticker}: NaN present in OHLCV"
    assert np.isfinite(arr).all(), f"{ticker}: inf present in OHLCV"


# =========================================================================
# RAW check 5a: OHLC consistency (Yahoo glitch tolerance -> xfail/report).
# =========================================================================


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_ohlc_consistency(ticker):
    """Flag: high>=low, open/close within [low,high] (rtol).

    Violations are Yahoo single-day point defects; surfaced via ``xfail`` for
    visibility rather than failing the suite, matching the VN100 test.
    """
    d = compute_raw_diag(ticker)
    total = d.n_high_lt_low + d.n_oc_outside
    if total:
        pytest.xfail(
            f"{ticker}: {d.n_high_lt_low} high<low (Parkinson-affecting) + "
            f"{d.n_oc_outside} open/close-outside-[low,high] row(s) "
            f"(Yahoo source glitch), high<low e.g. {d.ohlc_examples or 'none'}"
        )
    assert total == 0


# =========================================================================
# RAW check 5b: volume glitch (zero volume on a day the price moved).
# =========================================================================


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_volume_glitch(ticker):
    """Flag: volume == 0 while high != low (documented Yahoo glitch -> xfail)."""
    d = compute_raw_diag(ticker)
    if d.n_volume_glitch:
        pytest.xfail(
            f"{ticker}: {d.n_volume_glitch} row(s) with volume==0 but price moved "
            f"(high != low) -- Yahoo zero-volume glitch"
        )
    assert d.n_volume_glitch == 0


# =========================================================================
# RAW check 5c: leading synthetic backfill visibility guard.
# =========================================================================


@pytest.mark.parametrize("ticker", RAW_TICKERS)
def test_raw_leading_synthetic_backfill(ticker):
    """Guard: leading synthetic (flat) run should be small -> xfail if large."""
    d = compute_raw_diag(ticker)
    if d.leading_synthetic_run > LEADING_SYNTHETIC_THRESHOLD:
        pytest.xfail(
            f"{ticker}: {d.leading_synthetic_run} leading flat rows; recommend "
            f"trimming to first real trading date {d.first_real_date}"
        )
    assert d.leading_synthetic_run <= LEADING_SYNTHETIC_THRESHOLD


# =========================================================================
# PROCESSED checks P1-P3: hard assertions.
# =========================================================================


@pytest.mark.parametrize("ticker", PROCESSED_TICKERS)
def test_processed_schema(ticker):
    """P1: exactly two columns ``date,parkinson_volatility``; numeric values."""
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}_processed.csv")
    assert list(df.columns) == EXPECTED_PROCESSED_COLUMNS, (
        f"{ticker}: columns {list(df.columns)} != {EXPECTED_PROCESSED_COLUMNS}"
    )
    assert len(df) > 0, f"{ticker}: empty file"
    assert pd.api.types.is_numeric_dtype(df["parkinson_volatility"]), (
        f"{ticker}: parkinson_volatility is not numeric"
    )


@pytest.mark.parametrize("ticker", PROCESSED_TICKERS)
def test_processed_dates_valid(ticker):
    """P2: dates parseable, strictly increasing, unique, weekday-only."""
    df = load_processed(ticker)
    dates = df["date"]
    assert dates.notna().all(), f"{ticker}: {int(dates.isna().sum())} unparseable date(s)"
    assert dates.is_unique, f"{ticker}: duplicate dates present"
    assert dates.is_monotonic_increasing, f"{ticker}: dates not increasing"
    bad = dates[dates.dt.dayofweek >= 5]
    assert bad.empty, (
        f"{ticker}: {len(bad)} weekend date(s), e.g. "
        f"{bad.iloc[0].date() if len(bad) else ''}"
    )


@pytest.mark.parametrize("ticker", PROCESSED_TICKERS)
def test_processed_values_valid(ticker):
    """P3: parkinson_volatility finite, >= 0, and <= clip ceiling (0.1)."""
    vol = load_processed(ticker)["parkinson_volatility"].to_numpy(dtype=float)
    assert np.isfinite(vol).all(), f"{ticker}: contains NaN/inf"
    assert (vol >= 0.0).all(), f"{ticker}: negative volatility (min={vol.min()})"
    assert (vol <= CLIP_CEILING + 1e-12).all(), (
        f"{ticker}: value(s) exceed clip ceiling {CLIP_CEILING} (max={vol.max()})"
    )


# =========================================================================
# Check 6 + report generation.
# =========================================================================


def render_report(raw: list[RawDiag], proc: list[ProcDiag]) -> str:
    total_hl = sum(d.n_high_lt_low for d in raw)
    total_oc = sum(d.n_oc_outside for d in raw)
    total_glitch = sum(d.n_volume_glitch for d in raw)
    total_nonpos = sum(d.n_nonpositive for d in raw)

    hl_defect = sorted([d for d in raw if d.n_high_lt_low], key=lambda x: -x.n_high_lt_low)
    glitch = sorted([d for d in raw if d.n_volume_glitch], key=lambda x: -x.n_volume_glitch)
    backfill = [d for d in raw if d.leading_synthetic_run > LEADING_SYNTHETIC_THRESHOLD]
    short = sorted([d for d in raw if d.short_history], key=lambda x: x.rows)
    off = [d.ticker for d in raw if d.last_date != EXPECTED_LAST_DATE]

    proc_neg = [d for d in proc if d.n_negative]
    proc_nonfinite = [d for d in proc if d.n_nonfinite]
    proc_over_clip = [d for d in proc if d.n_over_clip]
    total_proc_zero = sum(d.zero_count for d in proc)
    total_proc_rows = sum(d.rows for d in proc)
    zero_frac_overall = total_proc_zero / total_proc_rows if total_proc_rows else 0.0

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out: list[str] = []
    out.append("# S&P500 daily OHLCV data-quality report (raw + processed)")
    out.append("")
    out.append(f"Generated: {stamp}")
    out.append("")
    out.append(
        f"Source: `data/raw/prices/sp500/*_ohlcv.csv` ({len(raw)} tickers) and "
        f"`data/processed/sp500/*_processed.csv` ({len(proc)} tickers). "
        "Yahoo Finance sourced (auto_adjust). Read-only; no CSV was modified."
    )
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(
        "RAW hard checks (schema, dates, OHLC positivity, NaN/inf) are enforced as "
        "assertions. OHLC-consistency, volume-glitch and leading-backfill are "
        "reported via `xfail` (documented Yahoo tolerances). PROCESSED checks "
        "(schema, dates, values >=0 and <= clip ceiling) are hard assertions."
    )
    out.append("")
    out.append(f"- RAW rows with high < low: **{total_hl}** (across {len(hl_defect)} tickers)")
    out.append(f"- RAW rows with open/close outside [low, high]: **{total_oc}**")
    out.append(f"- RAW non-positive OHLC rows: **{total_nonpos}**")
    out.append(
        f"- RAW volume-glitch rows (volume==0 while high!=low): **{total_glitch}** "
        f"(across {len(glitch)} tickers)"
    )
    out.append(
        f"- RAW leading-flat backfill (> {LEADING_SYNTHETIC_THRESHOLD} rows): "
        + (", ".join(f"{d.ticker}({d.leading_synthetic_run})" for d in backfill) or "none")
    )
    out.append(
        f"- RAW short-history tickers (< {SHORT_HISTORY_THRESHOLD} rows): {len(short)}"
    )
    out.append(
        f"- PROCESSED negative values: {sum(d.n_negative for d in proc)} "
        f"(tickers: {', '.join(d.ticker for d in proc_neg) or 'none'})"
    )
    out.append(
        f"- PROCESSED NaN/inf values: {sum(d.n_nonfinite for d in proc)} "
        f"(tickers: {', '.join(d.ticker for d in proc_nonfinite) or 'none'})"
    )
    out.append(
        f"- PROCESSED values over clip ceiling {CLIP_CEILING}: "
        f"{sum(d.n_over_clip for d in proc)} "
        f"(tickers: {', '.join(d.ticker for d in proc_over_clip) or 'none'})"
    )
    out.append(
        f"- PROCESSED zero-Parkinson rows: {total_proc_zero} / {total_proc_rows} "
        f"({zero_frac_overall:.4%} overall; H==L limit/flat days)"
    )
    out.append(
        f"- Coverage: tickers whose last_date != {EXPECTED_LAST_DATE}: "
        + (", ".join(off) if off else "none")
    )
    out.append("")

    # RAW high<low detail
    out.append("## RAW high < low rows (Parkinson-affecting)")
    out.append("")
    if hl_defect:
        out.append("| ticker | high<low_rows | example_dates |")
        out.append("|---|---|---|")
        for d in hl_defect:
            out.append(f"| {d.ticker} | {d.n_high_lt_low} | {d.ohlc_examples} |")
    else:
        out.append("None.")
    out.append("")

    # Volume glitch detail
    out.append("## RAW volume-glitch rows (volume==0 while price moved)")
    out.append("")
    if glitch:
        out.append("| ticker | volume_glitch_rows |")
        out.append("|---|---|")
        for d in glitch:
            out.append(f"| {d.ticker} | {d.n_volume_glitch} |")
    else:
        out.append("None.")
    out.append("")

    # Short history
    out.append(f"## RAW short-history tickers (< {SHORT_HISTORY_THRESHOLD} rows)")
    out.append("")
    if short:
        out.append("| ticker | rows | first_date | last_date |")
        out.append("|---|---|---|---|")
        for d in short:
            out.append(f"| {d.ticker} | {d.rows} | {d.first_date} | {d.last_date} |")
    else:
        out.append("None.")
    out.append("")

    # Coverage stats
    row_counts = [d.rows for d in raw]
    out.append("## RAW coverage")
    out.append("")
    out.append(
        f"Row counts: min={min(row_counts)}, median={int(np.median(row_counts))}, "
        f"max={max(row_counts)} across {len(raw)} tickers."
    )
    out.append("")

    return "\n".join(out) + "\n"


def test_generate_quality_report():
    """Check 6: build raw + processed diagnostics and write the markdown report."""
    raw = [compute_raw_diag(t) for t in RAW_TICKERS]
    proc = [compute_proc_diag(t) for t in PROCESSED_TICKERS]
    assert len(raw) == len(RAW_TICKERS)
    assert len(proc) == len(PROCESSED_TICKERS)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(raw, proc), encoding="utf-8")
    assert REPORT_PATH.exists()
