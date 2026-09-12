"""Unit + integration tests for the VNINDEX two-source stitch builder (build_vnindex.py).

Covers the display-unit parser, the Zenodo CSV parser, the Close cross-check, the source stitch, the
loud validator, the vnstock fetch wrapper (with an injected fake feed), and main() end-to-end on tmp
paths (fetch monkeypatched, no network)."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_vnindex as B  # noqa: E402


def test_parse_k_units():
    assert B._parse_k("538.93K") == pytest.approx(538930.0)
    assert B._parse_k("1.2M") == pytest.approx(1.2e6)
    assert B._parse_k("3B") == pytest.approx(3e9)
    assert B._parse_k("1,234") == pytest.approx(1234.0)
    assert B._parse_k("1.5") == pytest.approx(1.5)
    assert np.isnan(B._parse_k(""))
    assert np.isnan(B._parse_k("nan"))
    assert np.isnan(B._parse_k("abc"))


def _zenodo_fixture(path: Path):
    # Investing-style: descending date, DD/MM/YYYY, Vietnamese headers, K volume, one empty volume.
    path.write_text(
        "Ngày,Lần cuối,Mở,Cao,Thấp,KL,% Thay đổi\n"
        "03/01/2018,984.24,984.0,985.0,983.0,200.00K,0.10%\n"
        "02/01/2018,984.00,983.0,985.5,982.0,,0.05%\n"
        "31/07/2000,101.55,101.55,101.55,101.55,0.01K,1.55%\n",
        encoding="utf-8")


def test_parse_zenodo(tmp_path):
    fx = tmp_path / "z.csv"
    _zenodo_fixture(fx)
    z = B.parse_zenodo(fx)
    assert list(z.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert z["date"].is_monotonic_increasing
    assert z["date"].iloc[0] == pd.Timestamp("2000-07-31")
    assert z["close"].iloc[-1] == pytest.approx(984.24)
    assert z["volume"].iloc[0] == pytest.approx(10.0)      # 0.01K (2000-07-31, first after sort)
    assert np.isnan(z["volume"].iloc[1])                    # empty cell -> NaN (2018-01-02)


def _mkframe(dates, close):
    d = pd.to_datetime(dates)
    return pd.DataFrame({"date": d, "open": close, "high": close, "low": close,
                         "close": np.asarray(close, float), "volume": np.arange(len(close), dtype=float)})


def test_cross_check_stats():
    zen = _mkframe(["2018-01-02", "2018-01-03", "2018-01-04"], [100.0, 200.0, 300.0])
    vns = _mkframe(["2018-01-03", "2018-01-04", "2018-01-05"], [200.0, 303.0, 400.0])  # overlap 2 days
    xc = B.cross_check(zen, vns)
    assert xc["overlap_n"] == 2
    assert xc["close_pct_diff_max"] == pytest.approx(0.990099, abs=1e-4)   # |300-303|/303*100
    assert xc["worst_day"] == "2018-01-04"
    assert xc["n_days_gt_0.5pct"] == 1


def test_stitch_boundary_and_tags():
    zen = _mkframe(["2024-12-16", "2024-12-17"], [1260.0, 1265.0])   # 12-17 must be dropped (>= boundary)
    vns = _mkframe(["2024-12-17", "2024-12-18"], [1266.0, 1270.0])
    out = B.stitch(zen, vns)
    assert list(out["date"].dt.date.astype(str)) == ["2024-12-16", "2024-12-17", "2024-12-18"]
    assert out.loc[out["date"] == "2024-12-16", "vol_source"].iloc[0] == "zenodo_K_units"
    assert out.loc[out["date"] == "2024-12-17", "vol_source"].iloc[0] == "vnstock_raw_shares"
    assert out.loc[out["date"] == "2024-12-17", "close"].iloc[0] == 1266.0  # vnstock wins at boundary


def test_validate_pass_and_fail():
    good = _mkframe(["2020-01-01", "2020-01-02"], [100.0, 101.0])
    B.validate(good)                                              # no raise
    dup = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        B.validate(dup.sort_values("date").reset_index(drop=True))
    neg = _mkframe(["2020-01-01", "2020-01-02"], [100.0, -1.0])
    with pytest.raises(AssertionError):
        B.validate(neg)
    hl = _mkframe(["2020-01-01"], [100.0]); hl.loc[0, "high"] = 1.0; hl.loc[0, "low"] = 2.0
    with pytest.raises(AssertionError):
        B.validate(hl)
    uns = _mkframe(["2020-01-02", "2020-01-01"], [100.0, 101.0])  # not sorted
    with pytest.raises(AssertionError):
        B.validate(uns)


def _install_fake_vnstock(monkeypatch, frame):
    """Inject a fake vnstock.api.quote module whose Quote.history returns `frame`."""
    mod = types.ModuleType("vnstock.api.quote")

    class Quote:
        def __init__(self, symbol, source):
            pass

        def history(self, start, end, interval):
            return frame.rename(columns={"date": "time"})

    mod.Quote = Quote
    monkeypatch.setitem(sys.modules, "vnstock", types.ModuleType("vnstock"))
    monkeypatch.setitem(sys.modules, "vnstock.api", types.ModuleType("vnstock.api"))
    monkeypatch.setitem(sys.modules, "vnstock.api.quote", mod)


def test_fetch_vnstock_with_fake(monkeypatch):
    fr = _mkframe(["2025-01-02", "2025-01-03"], [1270.0, 1275.0])
    _install_fake_vnstock(monkeypatch, fr)
    got = B.fetch_vnstock("2025-01-01", "2025-12-31")
    assert list(got.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert got["date"].iloc[0] == pd.Timestamp("2025-01-02")


def test_main_end_to_end(tmp_path, monkeypatch):
    zfx = tmp_path / "zen.csv"
    _zenodo_fixture(zfx)
    out_csv = tmp_path / "vnindex.csv"
    out_prov = tmp_path / "prov.json"
    monkeypatch.setattr(B, "ZENODO_CSV", zfx)
    monkeypatch.setattr(B, "OUT_CSV", out_csv)
    monkeypatch.setattr(B, "OUT_PROV", out_prov)
    # supplement overlaps 2018-01-03 (Zenodo has it too) and extends past the boundary
    fr = _mkframe(["2018-01-03", "2024-12-18", "2025-01-02"], [984.30, 1266.0, 1270.0])
    _install_fake_vnstock(monkeypatch, fr)
    B.main()
    out = pd.read_csv(out_csv, parse_dates=["date"])
    assert out["date"].is_monotonic_increasing and out["date"].iloc[0] == pd.Timestamp("2000-07-31")
    assert out["date"].iloc[-1] == pd.Timestamp("2025-01-02")
    prov = json.loads(out_prov.read_text(encoding="utf-8"))
    assert prov["output_rows"] == len(out)
    assert prov["cross_check_close_zenodo_vs_vnstock"]["overlap_n"] == 1  # only 2018-01-03 overlaps
    assert prov["sources"][0]["license"] == "CC BY 4.0"
