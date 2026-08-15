"""Data-quality audit for processed Parkinson volatility series.

Scope: ``data/processed/<TICKER>_processed.csv`` (33 VN30 tickers, columns
``date,parkinson_volatility``). Values are Parkinson VARIANCE (sigma^2),
produced by ``src/common/parkinson_utils.py`` (formula (ln(H/L)^2)/(4 ln2),
then dropna, then clip upper 0.1).

The audit is read-only on the data files. It enforces schema/date/value
invariants as hard assertions (checks 1-3) and emits per-ticker diagnostics
(checks 4-6: coverage, zero/degenerate diagnostics, distribution sanity) via a
summary that is also written to a markdown report.

Run: ``python -m pytest tests/test_processed_data_quality.py -v``

Out of scope: ``data/raw/prices`` (covered by tests/test_raw_prices_quality.py),
``processing_summary.csv`` (not a per-ticker series), and any archived subfolder.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- Configuration -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = (
    PROJECT_ROOT / "docs" / "reports"
    / "2026-08-16_processed_data_quality_audit_report.md"
)

EXPECTED_COLUMNS = ["date", "parkinson_volatility"]
CLIP_CEILING = 0.1  # upper clip applied in parkinson_utils.process_single_stock
EXPECTED_LAST_DATE = "2026-08-14"  # coverage expectation (diagnostic only)

# Diagnostic flag thresholds.
HIGH_ZERO_FRACTION = 0.10  # flag tickers with > 10% exactly-zero rows
LEADING_ZERO_BACKFILL = 20  # xfail marker when leading zero-run exceeds this


def _discover_tickers() -> list[str]:
    """Return sorted ticker symbols from ``*_processed.csv`` (excl. summary)."""
    files = sorted(DATA_DIR.glob("*_processed.csv"))
    return [f.name[: -len("_processed.csv")] for f in files]


TICKERS = _discover_tickers()

# Simple read-only cache so each CSV is parsed once per session.
_CACHE: dict[str, pd.DataFrame] = {}


def _load(ticker: str) -> pd.DataFrame:
    """Load one processed CSV. Parses ``date``; leaves values as float."""
    if ticker not in _CACHE:
        path = DATA_DIR / f"{ticker}_processed.csv"
        df = pd.read_csv(path)
        # Parse dates strictly (ISO YYYY-MM-DD); errors surface as NaT.
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        _CACHE[ticker] = df
    return _CACHE[ticker]


def _compute_stats(ticker: str) -> dict:
    """Compute per-ticker diagnostics (checks 4-6) for the summary report."""
    df = _load(ticker)
    vol = df["parkinson_volatility"].to_numpy(dtype=float)
    is_zero = vol == 0.0

    # Leading run of exactly-zero rows from the start of the series.
    leading = 0
    for v in vol:
        if v == 0.0:
            leading += 1
        else:
            break
    nonzero_idx = np.flatnonzero(~is_zero)
    first_nonzero_date = (
        df["date"].iloc[int(nonzero_idx[0])] if nonzero_idx.size else pd.NaT
    )

    return {
        "ticker": ticker,
        "rows": len(df),
        "first_date": df["date"].iloc[0],
        "last_date": df["date"].iloc[-1],
        "zero_count": int(is_zero.sum()),
        "zero_frac": float(is_zero.mean()),
        "leading_zero_run": leading,
        "first_nonzero_date": first_nonzero_date,
        "min": float(np.min(vol)),
        "median": float(np.median(vol)),
        "mean": float(np.mean(vol)),
        "max": float(np.max(vol)),
        "std": float(np.std(vol)),
        "clip_hits": int(np.isclose(vol, CLIP_CEILING).sum()),
    }


# =========================================================================
# Preconditions
# =========================================================================

def test_data_dir_and_tickers_present():
    """Sanity: data dir exists and the expected 33 per-ticker files are found."""
    assert DATA_DIR.is_dir(), f"missing data dir: {DATA_DIR}"
    assert len(TICKERS) == 33, f"expected 33 tickers, found {len(TICKERS)}: {TICKERS}"
    assert "processing_summary" not in TICKERS, "summary file must be excluded"


# =========================================================================
# Check 1 (hard): schema
# =========================================================================

@pytest.mark.parametrize("ticker", TICKERS)
def test_schema_columns(ticker: str):
    """Exactly two columns ``date,parkinson_volatility`` in order; parseable."""
    df = pd.read_csv(DATA_DIR / f"{ticker}_processed.csv")
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"{ticker}: columns {list(df.columns)} != {EXPECTED_COLUMNS}"
    )
    assert len(df) > 0, f"{ticker}: empty file"
    # parkinson_volatility must be a numeric dtype after read.
    assert pd.api.types.is_numeric_dtype(df["parkinson_volatility"]), (
        f"{ticker}: parkinson_volatility is not numeric"
    )


# =========================================================================
# Check 2 (hard): dates
# =========================================================================

@pytest.mark.parametrize("ticker", TICKERS)
def test_dates_valid(ticker: str):
    """Dates parseable, strictly increasing, no duplicates, all weekdays."""
    df = _load(ticker)
    dates = df["date"]

    assert dates.notna().all(), (
        f"{ticker}: {int(dates.isna().sum())} unparseable date(s)"
    )
    assert dates.is_unique, f"{ticker}: duplicate dates present"
    assert dates.is_monotonic_increasing, f"{ticker}: dates not increasing"
    # strictly increasing = monotonic increasing AND unique (both asserted above)

    weekday = dates.dt.dayofweek  # Mon=0 .. Sun=6
    bad = dates[weekday >= 5]
    assert bad.empty, (
        f"{ticker}: {len(bad)} weekend date(s), e.g. "
        f"{bad.iloc[0].date() if len(bad) else ''}"
    )


# =========================================================================
# Check 3 (hard): values
# =========================================================================

@pytest.mark.parametrize("ticker", TICKERS)
def test_values_valid(ticker: str):
    """parkinson_volatility finite, >= 0, and <= 0.1 (the clip ceiling)."""
    vol = _load(ticker)["parkinson_volatility"].to_numpy(dtype=float)

    assert np.isfinite(vol).all(), f"{ticker}: contains NaN/inf"
    assert (vol >= 0.0).all(), f"{ticker}: negative volatility (min={vol.min()})"
    assert (vol <= CLIP_CEILING + 1e-12).all(), (
        f"{ticker}: value(s) exceed clip ceiling {CLIP_CEILING} (max={vol.max()})"
    )


# =========================================================================
# Check 5 (diagnostic + visible marker): leading-zero backfill signature
# =========================================================================

@pytest.mark.parametrize("ticker", TICKERS)
def test_leading_zero_backfill_signature(ticker: str):
    """xfail (visible marker) when leading zero-run exceeds the threshold.

    A long leading run of exactly-zero variance at the start of the series is a
    backfill signature (H==L synthetic rows), not a hard data error. It is
    surfaced as xfail so it is visible in ``-v`` output without failing the run.
    """
    stats = _compute_stats(ticker)
    run = stats["leading_zero_run"]
    if run > LEADING_ZERO_BACKFILL:
        first_nz = stats["first_nonzero_date"]
        first_nz_str = first_nz.date() if pd.notna(first_nz) else "n/a"
        pytest.xfail(
            f"{ticker}: leading zero-run={run} rows (> {LEADING_ZERO_BACKFILL}), "
            f"first non-zero {first_nz_str} -> backfill signature"
        )
    assert run <= LEADING_ZERO_BACKFILL


# =========================================================================
# Checks 4-6 (diagnostics): summary + markdown report
# =========================================================================

def test_write_diagnostics_report():
    """Build per-ticker diagnostics and write the markdown audit report.

    Non-failing: this test asserts only that stats were produced for every
    ticker. Coverage, zero-fraction, and distribution figures are reported (not
    asserted). Flags (high zero-fraction, backfill, clip hits, degenerate
    variance) are listed for human review.
    """
    stats = [_compute_stats(t) for t in TICKERS]
    assert len(stats) == len(TICKERS)

    def _d(x):
        return x.date().isoformat() if pd.notna(x) else "NaT"

    lines: list[str] = []
    lines.append("# Processed Data Quality Audit - VN30 Parkinson Volatility")
    lines.append("")
    lines.append(f"Generated: {_dt.date.today().isoformat()}")
    lines.append(f"Source: `data/processed/*_processed.csv` ({len(stats)} tickers)")
    lines.append(
        "Values are Parkinson variance (sigma^2); clip ceiling = "
        f"{CLIP_CEILING}. Zero rows come from H==L days (limit-up/down or "
        "synthetic backfill)."
    )
    lines.append("")

    # Flags summary.
    high_zero = [s for s in stats if s["zero_frac"] > HIGH_ZERO_FRACTION]
    backfill = [s for s in stats if s["leading_zero_run"] > LEADING_ZERO_BACKFILL]
    clip = [s for s in stats if s["clip_hits"] > 0]
    degenerate = [s for s in stats if s["std"] == 0.0]
    late_last = [s for s in stats if _d(s["last_date"]) != EXPECTED_LAST_DATE]

    lines.append("## Flags")
    lines.append("")
    lines.append(
        f"- High zero-fraction (> {HIGH_ZERO_FRACTION:.0%}): "
        + (", ".join(f"{s['ticker']} ({s['zero_frac']:.1%})" for s in high_zero)
           or "none")
    )
    lines.append(
        f"- Leading-zero backfill (run > {LEADING_ZERO_BACKFILL}): "
        + (", ".join(
            f"{s['ticker']} (run={s['leading_zero_run']}, "
            f"first_nonzero={_d(s['first_nonzero_date'])})"
            for s in backfill) or "none")
    )
    lines.append(
        "- Rows at clip ceiling: "
        + (", ".join(f"{s['ticker']} ({s['clip_hits']})" for s in clip) or "none")
    )
    lines.append(
        "- Degenerate (zero variance): "
        + (", ".join(s["ticker"] for s in degenerate) or "none")
    )
    lines.append(
        f"- Last date != {EXPECTED_LAST_DATE}: "
        + (", ".join(f"{s['ticker']} ({_d(s['last_date'])})" for s in late_last)
           or "none")
    )
    lines.append("")

    # Per-ticker table.
    header = (
        "| ticker | rows | first_date | last_date | zero_count | zero_frac | "
        "leading_zero_run | first_nonzero_date | min | median | max | clip_hits |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append("## Per-ticker diagnostics")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for s in sorted(stats, key=lambda x: x["ticker"]):
        lines.append(
            f"| {s['ticker']} | {s['rows']} | {_d(s['first_date'])} | "
            f"{_d(s['last_date'])} | {s['zero_count']} | {s['zero_frac']:.4f} | "
            f"{s['leading_zero_run']} | {_d(s['first_nonzero_date'])} | "
            f"{s['min']:.2e} | {s['median']:.2e} | {s['max']:.2e} | "
            f"{s['clip_hits']} |"
        )
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    assert REPORT_PATH.is_file()
