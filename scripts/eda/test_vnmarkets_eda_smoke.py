"""Integration + smoke tests for vnmarkets_eda: a synthetic panel exercises every analyze_panel branch
deterministically, and a real-data-sample smoke boots the full pipeline on each live market (vn30/vn100/hose)
on a tiny ticker slice. Unique basenames avoid the pytest duplicate-module collision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.eda.vnmarkets_eda as E

_N = 300


def _dates(n=_N):
    return pd.bdate_range("2015-01-01", periods=n).strftime("%Y-%m-%d").to_numpy()


def _ohlcv(close, vol=True):
    close = np.asarray(close, float)
    df = pd.DataFrame({"date": _dates(len(close)), "open": close, "high": close * 1.02,
                       "low": close * 0.98, "close": close})
    if vol:
        df["volume"] = np.linspace(1000, 5000, len(close))
    return df


def _build_screened_panel(tmp: Path, monkeypatch):
    raw = tmp / "raw"
    proc = tmp / "proc"
    raw.mkdir()
    proc.mkdir()
    rng = np.random.default_rng(1)
    base = 20 + np.cumsum(rng.normal(scale=0.2, size=_N))
    base = np.abs(base) + 1.0

    # AAA: full data with a split jump, a stale run, and zero-range days
    a = base.copy()
    a[100] = a[99] * 2.5                       # split jump (+150%)
    a[150:157] = a[149]                        # stale run of 8
    aa = _ohlcv(a)
    aa.loc[200:205, "high"] = aa.loc[200:205, "low"]   # zero-range days
    aa.to_csv(raw / "AAA_ohlcv.csv", index=False)
    pd.DataFrame({"date": _dates(), "parkinson_variance": np.abs(rng.normal(scale=1e-3, size=_N))}
                 ).to_csv(proc / "AAA_processed.csv", index=False)

    # BBB: correlated with AAA, NO volume column (screened-in) -> covers missing-volume branches
    bb = _ohlcv(a * 1.01 + rng.normal(scale=0.1, size=_N), vol=False)
    bb.to_csv(raw / "BBB_ohlcv.csv", index=False)
    b_pk = np.abs(rng.normal(scale=1e-3, size=_N))
    b_pk[:50] = 0.0                            # some zero-Parkinson days
    pd.DataFrame({"date": _dates(), "parkinson_variance": b_pk}
                 ).to_csv(proc / "BBB_processed.csv", index=False)

    # EXC: full data but EXCLUDED by the screen -> keep_ticker False path
    _ohlcv(base * 0.9).to_csv(raw / "EXC_ohlcv.csv", index=False)
    pd.DataFrame({"date": _dates(), "parkinson_variance": np.abs(rng.normal(scale=1e-3, size=_N))}
                 ).to_csv(proc / "EXC_processed.csv", index=False)

    # NOPKCOL: screened-in raw, processed file WITHOUT the parkinson column
    _ohlcv(base * 1.1).to_csv(raw / "NOPKCOL_ohlcv.csv", index=False)
    pd.DataFrame({"date": _dates(), "other": np.arange(_N)}).to_csv(proc / "NOPKCOL_processed.csv", index=False)

    # NOPKFILE: raw only (no processed file) -> pf.exists() False
    _ohlcv(base * 1.2).to_csv(raw / "NOPKFILE_ohlcv.csv", index=False)

    # SHORT: <2 rows -> continue
    _ohlcv([10.0]).to_csv(raw / "SHORT_ohlcv.csv", index=False)

    # NOHIGH: missing a required OHLC column -> continue
    pd.DataFrame({"date": _dates(3), "open": [1, 2, 3], "low": [1, 2, 3],
                  "close": [1, 2, 3], "volume": [1, 2, 3]}).to_csv(raw / "NOHIGH_ohlcv.csv", index=False)

    monkeypatch.setitem(E.VE.PRICE, "synthscreen", raw)
    monkeypatch.setitem(E.PROCESSED, "synthscreen", proc)
    monkeypatch.setattr(E, "SCREEN_PANELS", {"synthscreen"})

    def fake_screen(files, **kw):
        return [f for f in files if Path(f).name.replace("_processed.csv", "") in {"AAA", "BBB", "NOPKCOL"}]

    monkeypatch.setattr(E.FS, "screen_files", fake_screen)
    return raw, proc


def _build_empty_panel(tmp: Path, monkeypatch):
    raw = tmp / "eraw"
    raw.mkdir()
    _ohlcv([10.0]).to_csv(raw / "S1_ohlcv.csv", index=False)          # too short
    pd.DataFrame({"date": _dates(3), "open": [1, 2, 3], "low": [1, 2, 3],
                  "close": [1, 2, 3]}).to_csv(raw / "S2_ohlcv.csv", index=False)  # missing high
    monkeypatch.setitem(E.VE.PRICE, "synthempty", raw)
    monkeypatch.setitem(E.PROCESSED, "synthempty", tmp / "noproc")


def test_analyze_panel_screened_all_branches(tmp_path, monkeypatch):
    _build_screened_panel(tmp_path, monkeypatch)
    s = E.analyze_panel("synthscreen", limit=None)
    # AAA/BBB/EXC/NOPKCOL/NOPKFILE counted (5 valid raw); SHORT/NOHIGH skipped
    assert s["raw_tickers"] == 7                     # all files listed
    assert s["screened_tickers"] == 3               # AAA,BBB,NOPKCOL screened-in
    assert s["dirty"]["zero_range"] >= 6            # AAA zero-range days
    assert s["dirty"]["split_jumps"] >= 1
    assert s["dirty"]["stale_run_tickers"] >= 1
    assert s["corr"]["n_tickers"] >= 2              # correlation over screened set
    assert np.isfinite(s["zero_parkinson_rate"])
    assert s["stats"]["market_pk"]["n"] > 0         # pk_frames non-empty -> market_pk built
    assert s["zero_parkinson_by_year"]              # by-year map populated


def test_analyze_panel_empty_branches(tmp_path, monkeypatch):
    _build_empty_panel(tmp_path, monkeypatch)
    s = E.analyze_panel("synthempty", limit=None)
    assert s["corr"]["n_tickers"] == 0              # empty correlation branch
    assert s["stats"]["parkinson"]["n"] == 0        # empty summary_stats branch
    assert np.all(np.isnan(s["_charts"]["abs_ret_acf"]))
    assert s["zero_parkinson_by_year"] == {}


def test_run_writes_all_outputs(tmp_path, monkeypatch):
    _build_screened_panel(tmp_path, monkeypatch)
    _build_empty_panel(tmp_path, monkeypatch)
    out = tmp_path / "out"
    res = E.run(panels=["synthscreen"], comparison_extra=["synthscreen", "synthempty"],
                limit=None, out_dir=out)
    written = res["written"]
    assert any("synthscreen_eda.html" in w for w in written)
    assert any("comparison.html" in w for w in written)
    assert any("comparison.md" in w for w in written)
    for w in written:
        assert Path(w).exists() and Path(w).stat().st_size > 200
    markets = [r["market"] for r in res["table_rows"]]
    assert "synthscreen" in markets and "synthempty" in markets   # extra analyzed once
    html = (out / "2026-08-30_synthscreen_eda.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html and "Executive summary" in html


@pytest.mark.smoke
@pytest.mark.parametrize("panel", ["vn30", "vn100", "hose"])
def test_real_data_sample_smoke(panel, tmp_path):
    """Boot the full pipeline on a tiny real-data slice of each live market (limit=3 tickers)."""
    s = E.analyze_panel(panel, limit=3)
    assert s["panel"] == panel
    assert s["total_rows"] > 0
    assert s["stats"]["returns"]["n"] > 0
    html = E.render_market_html(s)
    assert "<html>" in html and "data:image/png;base64," in html
    out = tmp_path / f"{panel}.html"
    out.write_text(html, encoding="utf-8")
    assert out.stat().st_size > 1000
