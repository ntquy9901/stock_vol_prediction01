"""Regression tests for the canonical-window sourcing in vnmarkets_eda (change 2026-09-07): the HAR and
volume z-score rolling windows are imported from pipeline_config (single source of truth) and the volume
z-score window is 22 (not the former hardcoded 20). Named ``test_vnmarkets_eda.py`` so the pre-push gate's
adjacent-test discovery (``test_<module>.py``) finds it and can verify coverage of the changed lines; the
broader branch/smoke coverage lives in ``test_vnmarkets_eda_smoke.py`` / ``test_vnmarkets_eda_detectors.py``.
Unique basenames avoid the pytest duplicate-module collision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.eda.vnmarkets_eda as E

_N = 60   # >= 22 so the monthly HAR and the 22-day volume z-score are both non-empty


def test_windows_sourced_from_canonical_config():
    """The EDA imports the canonical pipeline_config and the volume z-score window is 22 (was 20)."""
    assert E.PC.VOLUME_ZSCORE_WINDOW == 22           # the 20 -> 22 fix (project monthly convention)
    assert E.PC.HAR_WEEKLY_WINDOW == 5
    assert E.PC.HAR_MONTHLY_WINDOW == 22


def _build_panel(tmp: Path, monkeypatch):
    raw = tmp / "raw"; proc = tmp / "proc"; raw.mkdir(); proc.mkdir()
    dates = pd.bdate_range("2015-01-01", periods=_N).strftime("%Y-%m-%d").to_numpy()
    close = 20.0 + np.arange(_N) * 0.1
    pd.DataFrame({"date": dates, "open": close, "high": close * 1.02, "low": close * 0.98,
                  "close": close, "volume": 1000.0 + np.arange(_N) * 7.0}   # strictly increasing -> sd>0
                 ).to_csv(raw / "AAA_ohlcv.csv", index=False)
    pd.DataFrame({"date": dates, "parkinson_variance": np.linspace(1e-4, 2e-4, _N)}
                 ).to_csv(proc / "AAA_processed.csv", index=False)
    monkeypatch.setitem(E.VE.PRICE, "synthwin", raw)
    monkeypatch.setitem(E.PROCESSED, "synthwin", proc)
    return raw, proc


def test_volume_zscore_uses_window_22(tmp_path, monkeypatch):
    """Behavioural pin: a 22-day rolling z-score over N volume rows yields N-21 valid values (a 20-day
    window would give N-19), so the pooled count proves the window is 22. Also exercises analyze_panel's
    HAR/volume rolling and the renamed ``volume_zscore_22`` stat key (the changed lines)."""
    _build_panel(tmp_path, monkeypatch)
    s = E.analyze_panel("synthwin", limit=None)
    assert "volume_zscore_22" in s["stats"]                       # renamed key (was volume_zscore_20)
    assert len(s["_charts"]["vz"]) == _N - (E.PC.VOLUME_ZSCORE_WINDOW - 1)   # == N-21 for window 22
    assert s["stats"]["har_monthly"]["n"] == _N - (E.PC.HAR_MONTHLY_WINDOW - 1)
    assert np.isfinite(s["zero_parkinson_rate"])


def test_raw_n_equals_file_count_under_limit(tmp_path, monkeypatch):
    """The collapsed dead branch: raw_n reflects the (already-capped) raw file count in both limit modes."""
    _build_panel(tmp_path, monkeypatch)
    full = E.analyze_panel("synthwin", limit=None)
    capped = E.analyze_panel("synthwin", limit=1)
    assert full["raw_tickers"] == 1
    assert capped["raw_tickers"] == 1
