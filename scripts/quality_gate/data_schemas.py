"""Pandera schemas for the project's data artifacts.

Schemas reflect the ACTUAL inspected columns (2026-08-08):

- data/processed/*.csv  (per-ticker): most files carry only ``date`` +
  ``parkinson_variance``; two (VPB, VRE) additionally carry OHLCV.
  ``processing_summary.csv`` is metadata (``ticker`` + ``num_records``) and is
  skipped. OHLCV checks are applied only when those columns are present.
- data/features/dual_group_news_panel.parquet: ``ticker`` + ``date`` +
  float embedding / norm / topic-count columns. Embedding rows are frequently
  all-NaN (days without news), so per-value non-null is NOT required and an
  entirely-NaN embedding column is tolerated; norm and topic-count columns are
  non-negative where present.

Each check returns ``(name, status, detail)`` where ``status`` is one of
``VALID`` / ``INVALID`` / ``MISSING``. A MISSING input (absent dir/file, no
per-ticker CSVs) is reported so the caller can SKIP rather than fake a pass or
raise a hard failure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NEWS_PANEL = PROJECT_ROOT / "data" / "features" / "dual_group_news_panel.parquet"

# Per-artifact status values.
VALID = "pass"
INVALID = "fail"
MISSING = "skip"

# Non per-ticker files that live in data/processed/ but are not price series.
_SKIP_PROCESSED = {"processing_summary.csv"}

# ---------------------------------------------------------------------------
# Processed per-ticker schema
# ---------------------------------------------------------------------------

_processed_base = {
    "date": Column(str, nullable=False, required=True),
    "parkinson_variance": Column(
        float, Check.ge(0), nullable=False, required=True
    ),
}

_processed_ohlcv = {
    "open": Column(float, Check.ge(0), nullable=False, required=False),
    "high": Column(float, Check.ge(0), nullable=False, required=False),
    "low": Column(float, Check.ge(0), nullable=False, required=False),
    "close": Column(float, Check.ge(0), nullable=False, required=False),
    # volume is int64 in the real files; keep dtype-agnostic, non-negative only.
    "volume": Column(None, Check.ge(0), nullable=False, required=False),
}

PROCESSED_SCHEMA = DataFrameSchema({**_processed_base, **_processed_ohlcv})


def _check_processed_file(path: Path) -> tuple[str, str, str]:
    name = f"processed:{path.name}"
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - report any read failure as invalid
        return (name, INVALID, f"read failed: {exc}")

    try:
        PROCESSED_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return (name, INVALID, f"schema violations: {exc.failure_cases.shape[0]}")

    # High >= Low where OHLCV present.
    if {"high", "low"}.issubset(df.columns):
        bad = int((df["high"] < df["low"]).sum())
        if bad:
            return (name, INVALID, f"{bad} rows with high < low")

    # Date parseable + strictly increasing (no duplicate/backward dates).
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        return (name, INVALID, f"{int(dates.isna().sum())} unparseable dates")
    if not dates.is_monotonic_increasing:
        return (name, INVALID, "dates not monotonic increasing")
    if dates.duplicated().any():
        return (name, INVALID, f"{int(dates.duplicated().sum())} duplicate dates")

    # No all-NaN columns (processed files carry only required series).
    all_nan = df.columns[df.isna().all()].tolist()
    if all_nan:
        return (name, INVALID, f"all-NaN columns: {all_nan}")

    return (name, VALID, f"{len(df)} rows, cols={list(df.columns)}")


# ---------------------------------------------------------------------------
# News panel schema
# ---------------------------------------------------------------------------

# ``date`` dtype is validated in code (any datetime64 unit) rather than pinned
# in the schema, so a real parquet stored as datetime64[ns] is not a false fail.
NEWS_PANEL_SCHEMA = DataFrameSchema(
    {
        "ticker": Column(str, nullable=False, required=True),
        "date": Column(nullable=False, required=True),
    },
    # Embedding/norm/topic columns are validated dynamically below (148 cols);
    # only structural columns are declared statically.
    strict=False,
)


def _check_news_panel(path: Path) -> tuple[str, str, str]:
    name = "news_panel"
    if not path.exists():
        return (name, MISSING, f"missing file: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return (name, INVALID, f"read failed: {exc}")

    try:
        NEWS_PANEL_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return (name, INVALID, f"schema violations: {exc.failure_cases.shape[0]}")

    # date must be a datetime dtype (any unit).
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        return (name, INVALID, f"date dtype not datetime: {df['date'].dtype}")

    # norm and topic-count columns must be non-negative where present.
    nonneg_cols = [
        c for c in df.columns if c.endswith("_norm") or c.endswith("_count")
    ]
    for col in nonneg_cols:
        col_min = df[col].min(skipna=True)
        if pd.notna(col_min) and col_min < 0:
            return (name, INVALID, f"{col} has negative values (min={col_min})")

    # Required structural columns must not be entirely NaN. Embedding/topic
    # columns are intentionally NOT checked: all-NaN is expected on news-free
    # days and is a documented, valid state.
    for col in ("ticker", "date"):
        if df[col].isna().all():
            return (name, INVALID, f"required column all-NaN: {col}")

    # Date monotonic per ticker (panel is sorted by ticker then date).
    for ticker, grp in df.groupby("ticker", sort=False):
        if not grp["date"].is_monotonic_increasing:
            return (name, INVALID, f"dates not monotonic for ticker {ticker}")

    return (
        name,
        VALID,
        f"{len(df)} rows, {df['ticker'].nunique()} tickers, {df.shape[1]} cols",
    )


def validate_data(
    processed_dir: Path = PROCESSED_DIR,
    news_panel: Path = NEWS_PANEL,
) -> list[tuple[str, str, str]]:
    """Validate all data artifacts.

    Returns a list of ``(name, status, detail)`` tuples, one per checked file
    plus the news panel. A missing directory, an empty processed directory, or a
    missing news panel yields ``MISSING`` (so the caller SKIPs it) rather than a
    fabricated pass or a hard failure.
    """
    results: list[tuple[str, str, str]] = []

    if not processed_dir.exists():
        results.append(("processed_dir", MISSING, f"missing dir: {processed_dir}"))
    else:
        csvs = sorted(
            p for p in processed_dir.glob("*.csv") if p.name not in _SKIP_PROCESSED
        )
        if not csvs:
            results.append(("processed_dir", MISSING, "no per-ticker CSVs found"))
        for path in csvs:
            results.append(_check_processed_file(path))

    results.append(_check_news_panel(news_panel))
    return results
