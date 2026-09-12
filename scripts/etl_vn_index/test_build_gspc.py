"""Unit + integration tests for the ^GSPC (S&P 500 index) builder (build_gspc.py)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_gspc as G  # noqa: E402


def _yahoo_fixture(path: Path):
    path.write_text(
        "Date,Adj Close,Close,High,Low,Open,Volume\n"
        "2016-09-12,2159.04,2159.04,2160.0,2119.0,2127.0,4000000000\n"
        "2000-01-03,1455.22,1455.22,1478.0,1438.36,1469.25,931800000\n",
        encoding="utf-8")


def _fred_fixture(path: Path):
    # real FRED: valid date every business day, value "." on market holidays
    path.write_text("observation_date,SP500\n2016-09-12,2159.10\n2016-09-13,2127.02\n2016-09-14,.\n",
                    encoding="utf-8")


def test_parse_yahoo(tmp_path):
    fx = tmp_path / "y.csv"
    _yahoo_fixture(fx)
    g = G.parse_yahoo(fx)
    assert list(g.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert g["date"].is_monotonic_increasing
    assert g["date"].iloc[0] == pd.Timestamp("2000-01-03")
    assert g["close"].iloc[-1] == pytest.approx(2159.04)


def test_parse_fred(tmp_path):
    fx = tmp_path / "f.csv"
    _fred_fixture(fx)
    f = G.parse_fred(fx)
    assert list(f.columns) == ["date", "close"]
    assert len(f) == 2                                  # the "." missing row is dropped
    assert f["close"].iloc[0] == pytest.approx(2159.10)


def test_main_end_to_end(tmp_path, monkeypatch):
    yfx, ffx = tmp_path / "y.csv", tmp_path / "f.csv"
    _yahoo_fixture(yfx); _fred_fixture(ffx)
    out_csv, out_prov = tmp_path / "gspc.csv", tmp_path / "prov.json"
    monkeypatch.setattr(G, "YAHOO_CSV", yfx)
    monkeypatch.setattr(G, "FRED_CSV", ffx)
    monkeypatch.setattr(G, "OUT_CSV", out_csv)
    monkeypatch.setattr(G, "OUT_PROV", out_prov)
    G.main()
    out = pd.read_csv(out_csv, parse_dates=["date"])
    assert out["date"].iloc[0] == pd.Timestamp("2000-01-03")
    assert out["date"].iloc[-1] == pd.Timestamp("2016-09-12")
    prov = json.loads(out_prov.read_text(encoding="utf-8"))
    assert prov["output_rows"] == len(out)
    # overlap is only 2016-09-12 (Yahoo has it; FRED has 09-12 and 09-13)
    assert prov["cross_check_close_yahoo_vs_fred"]["overlap_n"] == 1
    assert prov["sources"][1]["name"] == "FRED SP500"
