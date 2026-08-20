"""Tests for src/data/download_sp500.py — no network (yfinance + the constituents fetch are mocked)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import download_sp500 as dl  # noqa: E402


def _yf_frame() -> pd.DataFrame:
    """A yfinance-like history frame: DatetimeIndex named 'Date' + OHLCV columns."""
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    idx.name = "Date"
    return pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [3.0, 4.0], "Low": [0.5, 1.0],
         "Close": [2.0, 3.0], "Volume": [100, None]},  # None volume must become int 0
        index=idx,
    )


def test_normalize_schema_and_types():
    out = dl.normalize(_yf_frame())
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert out["date"].tolist() == ["2020-01-02", "2020-01-03"]
    assert out["volume"].tolist() == [100, 0]                 # NaN volume -> 0
    assert str(out["volume"].dtype) == "int64"


def test_normalize_empty_raises():
    with pytest.raises(ValueError, match="empty history"):
        dl.normalize(pd.DataFrame())


def test_get_constituents_dot_to_dash(monkeypatch):
    fake = pd.DataFrame({"Symbol": ["AAPL", "BRK.B", "BF.B"], "Security": ["a", "b", "c"]})
    monkeypatch.setattr(dl.pd, "read_csv", lambda url: fake)
    assert dl.get_constituents("ignored") == ["AAPL", "BRK-B", "BF-B"]


def test_download_universe_writes_and_skips(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dl, "fetch_ticker", lambda s: (calls.append(s) or dl.normalize(_yf_frame())))
    res = dl.download_universe(["AAA", "BBB"], tmp_path, sleep=0.0)
    assert (tmp_path / "AAA_ohlcv.csv").exists() and (tmp_path / "BBB_ohlcv.csv").exists()
    assert res["AAA"] == {"ok": True, "rows": 2}
    # re-run: both already on disk -> skipped, fetch_ticker NOT called again
    calls.clear()
    res2 = dl.download_universe(["AAA", "BBB"], tmp_path, sleep=0.0)
    assert calls == [] and res2["AAA"]["skipped"] is True


def test_download_universe_reports_fetch_error(monkeypatch, tmp_path):
    def boom(sym):
        raise RuntimeError("network down")
    monkeypatch.setattr(dl, "fetch_ticker", boom)
    res = dl.download_universe(["ZZZ"], tmp_path, sleep=0.0)
    assert res["ZZZ"]["ok"] is False and "network down" in res["ZZZ"]["error"]
    assert not (tmp_path / "ZZZ_ohlcv.csv").exists()          # no file written on failure


@pytest.mark.smoke
def test_smoke_normalize_then_download(monkeypatch, tmp_path):
    """Boot the download path end to end on a mocked ticker (happy path)."""
    monkeypatch.setattr(dl, "fetch_ticker", lambda s: dl.normalize(_yf_frame()))
    res = dl.download_universe(["MMM"], tmp_path, sleep=0.0)
    written = pd.read_csv(tmp_path / "MMM_ohlcv.csv")
    assert res["MMM"]["ok"] and list(written.columns) == dl.OUT_COLS and len(written) == 2
