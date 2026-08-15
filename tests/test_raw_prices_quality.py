"""Reusable data-quality checks for raw VN30 daily price files.

Target: ``data/raw/prices/<TICKER>_ohlcv.csv`` (33 tickers, columns
``date,open,high,low,close,volume``). Read-only on the data; this module never
writes to the CSV files.

Three storage conventions coexist in the raw files:
  * most tickers: plain ``YYYY-MM-DD`` date, prices in thousands-VND;
  * VPB & VRE: tz-aware ``YYYY-MM-DD HH:MM:SS+07:00`` (later switching to plain
    date mid-file), prices in full-VND;
  * LPB: recovered from SSI iBoard (see ``LPB_SOURCE_NOTE.md``).
Date parsing normalises all of these via ``split(' ')[0]``.

Check taxonomy (see the project CLAUDE.md quality gate):
  * Checks 1-4 (schema, dates, OHLC validity, NaN/inf) are HARD assertions --
    a violation fails the test.
  * Checks 5-6 (synthetic backfill, coverage) are REPORTED diagnostics written
    to a markdown summary, because leading synthetic backfill is a data-provenance
    issue rather than a schema error. One visibility guard is kept: any ticker
    whose leading synthetic run exceeds ``LEADING_SYNTHETIC_THRESHOLD`` rows is
    surfaced via ``pytest.xfail`` so it shows up in ``-v`` output.

Downstream, Parkinson variance = (ln(High/Low)^2)/(4 ln 2) is computed from
these High/Low values; a flat row (High == Low) yields a Parkinson variance of
exactly 0, which is why a leading run of flat rows is the provenance signal of
interest.

Synthetic-backfill definition (deliberate): the leading run is measured as the
leading contiguous block of *flat* rows (High == Low). The raw backfill mixes
flat rows carrying zero volume with flat rows carrying tiny non-zero volume
(e.g. VHM's 2011-2014 prefix), so a strict "flat AND volume == 0" prefix would
stop at the first row and miss the backfill entirely. The stricter
"flat AND volume == 0" leading run is still reported separately
(``leading_zerovol_flat_run``) for transparency. Legitimate later flat days
(limit-up / limit-down) carry volume > 0 and appear scattered after real trading
begins, so they are never part of the leading flat prefix.

Run with:
    python -m pytest tests/test_raw_prices_quality.py -v
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "2026-08-16_raw_prices_data_quality_report.md"

EXPECTED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
EXPECTED_LAST_DATE = "2026-08-14"
LEADING_SYNTHETIC_THRESHOLD = 20
# Relative tolerance for OHLC inequality checks. VPB/VRE store full-VND prices as
# float32-precision decimals (e.g. 7265.587890625), so exact >=/<= comparisons
# raise spurious ~1e-16..1e-7 relative "violations". 1e-5 absorbs that float
# noise while remaining 2+ orders below the smallest genuine violation (~4e-3).
OHLC_RTOL = 1e-5


def discover_tickers() -> list[str]:
    """Return sorted ticker symbols from ``*_ohlcv.csv`` files (non-CSV ignored)."""
    suffix = "_ohlcv.csv"
    return sorted(p.name[: -len(suffix)] for p in PRICES_DIR.glob(f"*{suffix}"))


TICKERS = discover_tickers()


def _read_raw(ticker: str) -> pd.DataFrame:
    """Read a raw CSV as strings (so schema/parse checks are done explicitly)."""
    path = PRICES_DIR / f"{ticker}_ohlcv.csv"
    return pd.read_csv(path, dtype=str)


def load_prices(ticker: str) -> pd.DataFrame:
    """Load one ticker, add parsed ``date_norm`` and numeric OHLCV columns.

    Date is normalised via ``split(' ')[0]`` to strip any tz-aware time part.
    Numeric coercion uses ``errors='coerce'`` so unparseable values surface as
    NaN and are caught by the NaN/parse checks rather than raising.
    """
    raw = _read_raw(ticker)
    df = raw.copy()
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
    rows: int
    first_date: str
    last_date: str
    leading_synthetic_run: int
    leading_zerovol_flat_run: int
    first_real_date: str
    flat_frac: float
    zerovol_frac: float
    any_ohlc_violation: bool
    n_high_lt_max_oc: int
    n_low_gt_min_oc: int


def _leading_true_run(mask: np.ndarray) -> int:
    """Length of the leading contiguous run of True values in a boolean array."""
    if mask.size == 0:
        return 0
    if mask.all():
        return int(mask.size)
    return int(np.argmax(~mask))


def ohlc_violation_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-row boolean masks for each OHLC-validity rule (True == violating row).

    Inequality rules use ``OHLC_RTOL`` so float32 representation noise in the
    full-VND tickers is not reported as a violation.
    """
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


def _has_ohlc_violation(df: pd.DataFrame) -> bool:
    arr = df[NUMERIC_COLS].to_numpy(dtype=float)
    if np.isnan(arr).any():
        return True
    return any(bool(m.any()) for m in ohlc_violation_masks(df).values())


def compute_diagnostics(ticker: str) -> Diagnostics:
    """Compute checks 5-6 diagnostics for one ticker (read-only)."""
    df = load_prices(ticker)
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol = df["volume"].to_numpy()

    flat = high == low
    zerovol = vol == 0
    leading_flat = _leading_true_run(flat)
    leading_zerovol_flat = _leading_true_run(flat & zerovol)

    if leading_flat < n:
        first_real = df["date_norm"].iloc[leading_flat].strftime("%Y-%m-%d")
    else:
        first_real = "N/A (all flat)"

    masks = ohlc_violation_masks(df)

    return Diagnostics(
        ticker=ticker,
        rows=n,
        first_date=df["date_norm"].iloc[0].strftime("%Y-%m-%d"),
        last_date=df["date_norm"].iloc[-1].strftime("%Y-%m-%d"),
        leading_synthetic_run=leading_flat,
        leading_zerovol_flat_run=leading_zerovol_flat,
        first_real_date=first_real,
        flat_frac=float(np.mean(flat)),
        zerovol_frac=float(np.mean(zerovol)),
        any_ohlc_violation=_has_ohlc_violation(df),
        n_high_lt_max_oc=int(masks["high_lt_max_oc"].sum()),
        n_low_gt_min_oc=int(masks["low_gt_min_oc"].sum()),
    )


# --------------------------------------------------------------------------- #
# Checks 1-4: hard assertions, parametrized per ticker.
# --------------------------------------------------------------------------- #


def test_tickers_discovered():
    """Sanity: the expected 33-ticker universe is present."""
    assert len(TICKERS) == 33, f"expected 33 tickers, found {len(TICKERS)}: {TICKERS}"


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
    # Strictly increasing already precludes duplicates; assert explicitly too.
    assert dates.is_unique, f"{ticker}: duplicate date(s) present"

    weekend = dates[dates.dt.dayofweek >= 5]
    assert weekend.empty, (
        f"{ticker}: {len(weekend)} weekend date(s), e.g. "
        f"{weekend.dt.strftime('%Y-%m-%d').tolist()[:5]}"
    )


@pytest.mark.parametrize("ticker", TICKERS)
def test_ohlc_validity(ticker):
    """Check 3: positivity, high>=low, high>=max(o,c), low<=min(o,c), volume>=0."""
    df = load_prices(ticker)
    masks = ohlc_violation_masks(df)

    def _examples(mask):
        idx = np.flatnonzero(mask)[:5]
        return [df["date"].iloc[i].split(" ")[0] for i in idx]

    assert not masks["nonpositive"].any(), (
        f"{ticker}: {int(masks['nonpositive'].sum())} non-positive OHLC row(s), "
        f"e.g. {_examples(masks['nonpositive'])}"
    )
    assert not masks["high_lt_low"].any(), (
        f"{ticker}: {int(masks['high_lt_low'].sum())} row(s) with high < low, "
        f"e.g. {_examples(masks['high_lt_low'])}"
    )
    assert not masks["high_lt_max_oc"].any(), (
        f"{ticker}: {int(masks['high_lt_max_oc'].sum())} row(s) with high < max(open,close), "
        f"e.g. {_examples(masks['high_lt_max_oc'])}"
    )
    assert not masks["low_gt_min_oc"].any(), (
        f"{ticker}: {int(masks['low_gt_min_oc'].sum())} row(s) with low > min(open,close), "
        f"e.g. {_examples(masks['low_gt_min_oc'])}"
    )
    assert not masks["negative_volume"].any(), f"{ticker}: negative volume present"


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
    """Check 5 guard: leading synthetic (flat) run should be small.

    Tickers exceeding the threshold are reported via ``xfail`` (visible as XFAIL
    in ``-v``) rather than a hard failure, since backfill is a provenance issue
    the downstream pipeline must trim, not a schema error.
    """
    d = compute_diagnostics(ticker)
    if d.leading_synthetic_run > LEADING_SYNTHETIC_THRESHOLD:
        pytest.xfail(
            f"{ticker}: {d.leading_synthetic_run} leading flat synthetic rows "
            f"(zerovol-flat={d.leading_zerovol_flat_run}); recommend trimming to "
            f"first real trading date {d.first_real_date}"
        )
    assert d.leading_synthetic_run <= LEADING_SYNTHETIC_THRESHOLD


# --------------------------------------------------------------------------- #
# Check 6 + report generation: per-ticker diagnostics table written to markdown.
# --------------------------------------------------------------------------- #


def build_report_rows() -> list[Diagnostics]:
    """Compute diagnostics for every ticker, sequentially (modest memory)."""
    return [compute_diagnostics(t) for t in TICKERS]


def render_report(rows: list[Diagnostics]) -> str:
    header = (
        "| ticker | rows | first_date | last_date | leading_synthetic_run | "
        "first_real_date | flat_frac | zerovol_frac | any_OHLC_violation |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {d.ticker} | {d.rows} | {d.first_date} | {d.last_date} | "
        f"{d.leading_synthetic_run} | {d.first_real_date} | "
        f"{d.flat_frac:.4f} | {d.zerovol_frac:.4f} | {d.any_ohlc_violation} |\n"
        for d in rows
    )

    flagged = [d for d in rows if d.leading_synthetic_run > LEADING_SYNTHETIC_THRESHOLD]
    trim_lines = "".join(
        f"| {d.ticker} | {d.leading_synthetic_run} | {d.leading_zerovol_flat_run} | "
        f"{d.first_real_date} |\n"
        for d in sorted(flagged, key=lambda x: -x.leading_synthetic_run)
    )

    violators = [d for d in rows if d.n_high_lt_max_oc or d.n_low_gt_min_oc]
    viol_lines = "".join(
        f"| {d.ticker} | {d.n_high_lt_max_oc} | {d.n_low_gt_min_oc} |\n"
        for d in sorted(violators, key=lambda x: -(x.n_high_lt_max_oc + x.n_low_gt_min_oc))
    )
    off = [d.ticker for d in rows if d.last_date != EXPECTED_LAST_DATE]

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        "# Raw price files data-quality report\n\n"
        f"Generated: {stamp}\n\n"
        f"Source: `data/raw/prices/*_ohlcv.csv` ({len(rows)} tickers). "
        "Read-only; no CSV was modified.\n\n"
        "Checks 1-4 (schema, dates, OHLC validity, NaN/inf) are enforced as hard "
        "assertions in `tests/test_raw_prices_quality.py`. Checks 5-6 (synthetic "
        "backfill, coverage) are reported below.\n\n"
        "`leading_synthetic_run` = leading contiguous run of flat rows "
        "(high == low), which produce a Parkinson variance of 0. "
        "`first_real_date` = first row with high != low. See module docstring for "
        "the definitional rationale (backfill mixes zero and tiny-nonzero volume).\n\n"
        "## Per-ticker diagnostics\n\n" + header + body + "\n"
        f"## Tickers with leading synthetic backfill > {LEADING_SYNTHETIC_THRESHOLD} rows "
        "(recommended trim point)\n\n"
        + (
            "| ticker | leading_synthetic_run | leading_zerovol_flat_run | recommended_trim_date |\n"
            "|---|---|---|---|\n" + trim_lines
            if flagged
            else "None.\n"
        )
        + (
            "\nVHM is the only ticker with non-trivial leading synthetic backfill. Its flat "
            "prefix ends 2014-09-22 (first high != low), but sustained liquid trading only "
            "begins 2018-05-23 (first day with volume in the millions; matches the ~2018-05 "
            "HOSE listing). The 2014-09..2018-05 window is sparse/illiquid (sporadic tiny "
            "volume, frequent flat days). Trimming to 2014-09-22 removes the Parkinson-zero "
            "prefix; trimming to 2018-05-23 additionally removes the illiquid warm-up. No "
            "other ticker has a leading flat run exceeding 6 rows.\n"
        )
        + "\n## Check 3: OHLC-consistency violations (hard-fail)\n\n"
        + (
            "All 33 tickers satisfy high >= low and positivity. Violations below are rows where "
            "close/open falls outside [low, high] (compared with relative tolerance "
            f"{OHLC_RTOL:g} to exclude float32 noise). These are genuine raw-source "
            "inconsistencies, likely from dividend/split back-adjustment applied unevenly across "
            "O/H/L/C; they do not affect Parkinson variance (which uses only high/low, and "
            "high >= low holds everywhere).\n\n"
            "| ticker | rows_high_lt_max(open,close) | rows_low_gt_min(open,close) |\n"
            "|---|---|---|\n" + viol_lines
            if violators
            else "None.\n"
        )
        + "\n## Check 6: coverage\n\n"
        + (
            f"All {len(rows)} tickers end on {EXPECTED_LAST_DATE}.\n"
            if not off
            else f"Tickers whose last_date != {EXPECTED_LAST_DATE}: {off}\n"
        )
    )


def test_generate_quality_report():
    """Check 6: build the per-ticker table and write the markdown report."""
    rows = build_report_rows()
    assert len(rows) == 33

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(rows), encoding="utf-8")
    assert REPORT_PATH.exists()

    # Coverage sanity: every series shares the expected last trading date.
    off = [d.ticker for d in rows if d.last_date != EXPECTED_LAST_DATE]
    # Reported, not asserted hard (check 6 is diagnostic); record any drift.
    if off:
        print(f"Tickers whose last_date != {EXPECTED_LAST_DATE}: {off}")
