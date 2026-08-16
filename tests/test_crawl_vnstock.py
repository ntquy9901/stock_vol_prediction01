"""Unit tests for src.data.crawl_vnstock — fully mocked, NO network access."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.crawl_vnstock import (  # noqa: E402
    RAW_COLUMNS,
    crawl_universe,
    fetch_ticker,
    normalize_history,
    read_universe,
)


def _vnstock_frame(dates, vols):
    """Build a fake vnstock history frame (columns time,open,high,low,close,volume)."""
    return pd.DataFrame({
        "time": pd.to_datetime(dates),
        "open": [10.0] * len(dates),
        "high": [11.0] * len(dates),
        "low": [9.0] * len(dates),
        "close": [10.5] * len(dates),
        "volume": vols,
    })


class _FakeClient:
    """Mimics vnstock's ``Vnstock().stock(...)`` object: exposes ``.quote.history``."""

    def __init__(self, frame=None, exc=None):
        self._frame, self._exc = frame, exc

    @property
    def quote(self):
        return self

    def history(self, start, end, interval):
        if self._exc is not None:
            raise self._exc
        return self._frame


def test_normalize_history_schema_and_dtypes():
    df = _vnstock_frame(["2020-01-07", "2020-01-06"], [200, 100])  # unsorted on purpose
    out = normalize_history(df)
    assert list(out.columns) == RAW_COLUMNS
    assert out["date"].tolist() == ["2020-01-06", "2020-01-07"]  # sorted ascending
    assert out["date"].iloc[0] == "2020-01-06"                    # plain YYYY-MM-DD str
    assert str(out["volume"].dtype) == "int64"


def test_normalize_history_drops_duplicate_dates():
    df = _vnstock_frame(["2020-01-06", "2020-01-06"], [100, 100])
    assert len(normalize_history(df)) == 1


def test_normalize_history_empty_raises():
    with pytest.raises(ValueError):
        normalize_history(pd.DataFrame())


def test_fetch_ticker_uses_first_source_on_success():
    frame = _vnstock_frame(["2020-01-06"], [100])
    calls = []

    def factory(symbol, source):
        calls.append(source)
        return _FakeClient(frame=frame)

    df, source = fetch_ticker("FPT", client_factory=factory, sleeper=lambda s: None)
    assert source == "VCI" and len(df) == 1
    assert calls == ["VCI"]  # no rotation needed


def test_fetch_ticker_rotates_source_on_failure():
    frame = _vnstock_frame(["2020-01-06"], [100])
    calls = []

    def factory(symbol, source):
        calls.append(source)
        if source == "VCI":
            return _FakeClient(exc=ConnectionError("rate limited"))
        return _FakeClient(frame=frame)

    slept = []
    df, source = fetch_ticker("FPT", max_retries=2, client_factory=factory,
                              sleeper=slept.append)
    assert source == "KBS" and len(df) == 1
    assert calls == ["VCI", "VCI", "KBS"]  # exhausted VCI retries, then rotated
    assert len(slept) >= 1                   # backoff was applied on VCI failure


def test_fetch_ticker_raises_when_all_sources_fail():
    def factory(symbol, source):
        return _FakeClient(exc=ConnectionError("down"))

    with pytest.raises(RuntimeError, match="all sources"):
        fetch_ticker("FPT", max_retries=1, client_factory=factory, sleeper=lambda s: None)


def test_read_universe(tmp_path):
    (tmp_path / "FPT_ohlcv.csv").write_text("date\n")
    (tmp_path / "VNM_ohlcv.csv").write_text("date\n")
    assert read_universe(tmp_path) == ["FPT", "VNM"]


def test_crawl_universe_writes_files_and_records_outcome(tmp_path, monkeypatch):
    import src.data.crawl_vnstock as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)  # no real delay between tickers
    frame = _vnstock_frame(["2020-01-06", "2020-01-07"], [100, 200])

    def factory(symbol, source):
        if symbol == "BAD":
            return _FakeClient(exc=ValueError("no data"))
        return _FakeClient(frame=frame)

    results = crawl_universe(
        ["FPT", "BAD"], tmp_path,
        polite_sleep=0, logger=lambda m: None,
        client_factory=factory, max_retries=1, sleeper=lambda s: None,
    )
    assert results["FPT"]["ok"] is True and results["FPT"]["rows"] == 2
    assert results["FPT"]["first"] == "2020-01-06" and results["FPT"]["last"] == "2020-01-07"
    assert (tmp_path / "FPT_ohlcv.csv").exists()
    assert results["BAD"]["ok"] is False
    assert not (tmp_path / "BAD_ohlcv.csv").exists()


def test_crawl_universe_skips_existing_for_resume(tmp_path):
    (tmp_path / "FPT_ohlcv.csv").write_text("date,open,high,low,close,volume\n")  # pre-existing

    def factory(symbol, source):  # must NOT be called for the skipped ticker
        raise AssertionError("client_factory should not run for existing file")

    results = crawl_universe(
        ["FPT"], tmp_path, polite_sleep=0, skip_existing=True,
        logger=lambda m: None, client_factory=factory, sleeper=lambda s: None,
    )
    assert results["FPT"] == {"ok": True, "skipped": True}


@pytest.mark.smoke
def test_smoke_crawl_universe_end_to_end(tmp_path):
    """Boot the crawl pipeline end-to-end with a mocked client (no network)."""
    frame = _vnstock_frame(["2020-01-06", "2020-01-07"], [100, 200])
    results = crawl_universe(
        ["FPT"], tmp_path, polite_sleep=0, logger=lambda m: None,
        client_factory=lambda s, src: _FakeClient(frame=frame),
        sleeper=lambda s: None,
    )
    assert results["FPT"]["ok"] is True
    written = pd.read_csv(tmp_path / "FPT_ohlcv.csv", dtype={"date": str})
    assert list(written.columns) == RAW_COLUMNS
    assert written["date"].iloc[0] == "2020-01-06"
