"""Tests for volume_io.load_log_volume (real ln(volume) from raw OHLCV, non-positive -> NaN, missing-file skip)."""
import numpy as np
import pandas as pd
import pytest

import config
import volume_io


def _write_ohlcv(path, dates, volume):
    n = len(dates)
    pd.DataFrame({
        "date": dates, "open": np.ones(n), "high": np.ones(n), "low": np.ones(n),
        "close": np.ones(n), "volume": volume,
    }).to_csv(path, index=False)


def test_load_log_volume_reads_and_masks_nonpositive(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(config, "RAW_VOL_DIR", {"hose": "raw"})
    dates = pd.bdate_range("2020-01-01", periods=4)
    _write_ohlcv(raw / "AAA_ohlcv.csv", dates, [10.0, 0.0, 100.0, -5.0])   # 0 and negative -> NaN
    _write_ohlcv(raw / "BBB_ohlcv.csv", dates, [1.0, 2.0, 4.0, 8.0])
    out = volume_io.load_log_volume("hose", ["AAA", "BBB", "MISSING"])     # MISSING has no file -> skipped
    assert list(out.columns) == ["AAA", "BBB"]
    assert np.isclose(out["AAA"].iloc[0], np.log(10.0))
    assert np.isnan(out["AAA"].iloc[1]) and np.isnan(out["AAA"].iloc[3])   # non-positive -> NaN
    assert np.isclose(out["BBB"].iloc[3], np.log(8.0))
    assert out.index.is_monotonic_increasing


def test_load_log_volume_no_files_returns_empty(tmp_path, monkeypatch):
    raw = tmp_path / "raw"; raw.mkdir()
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(config, "RAW_VOL_DIR", {"hose": "raw"})
    out = volume_io.load_log_volume("hose", ["NONE1", "NONE2"])
    assert out.empty


@pytest.mark.smoke
def test_load_log_volume_real_hose_slice():
    # real-data-sample smoke: one real HOSE ticker must load with finite ln(volume) values
    out = volume_io.load_log_volume("hose", ["AAA"])
    assert list(out.columns) == ["AAA"]
    assert out["AAA"].notna().any()
    assert np.isfinite(out["AAA"].dropna().to_numpy()).all()
    assert out.index.is_monotonic_increasing
