import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.process_parkinson_pipeline import main  # noqa: E402


def _write_ohlcv(path, dates, highs, lows):
    n = len(dates)
    pd.DataFrame({
        "date": dates,
        "open": [1.0] * n, "high": highs, "low": lows, "close": [1.0] * n,
        "volume": [100] * n,
    }).to_csv(path, index=False)


def test_cli_processes_arbitrary_universe_dir(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    out = tmp_path / "out"
    _write_ohlcv(raw / "ABC_ohlcv.csv", ["2020-01-01", "2020-01-02"], [1.1, 1.2], [1.0, 1.0])
    _write_ohlcv(raw / "XYZ_ohlcv.csv", ["2020-01-01", "2020-01-02"], [2.0, 2.1], [1.0, 1.0])

    rc = main(["--raw", str(raw), "--out", str(out)])
    assert rc == 0
    for tk in ("ABC", "XYZ"):
        proc = pd.read_csv(out / f"{tk}_processed.csv")
        assert list(proc.columns) == ["date", "parkinson_volatility"]
        assert (proc["parkinson_volatility"] >= 0).all()
        assert len(proc) == 2


def test_cli_normalizes_tzaware_dates(tmp_path):
    # VPB/VRE-style tz-aware datetime input must be normalized to plain YYYY-MM-DD (cross-universe).
    raw = tmp_path / "raw"; raw.mkdir()
    out = tmp_path / "out"
    _write_ohlcv(raw / "TZX_ohlcv.csv",
                 ["2021-03-01 00:00:00+07:00", "2021-03-02 00:00:00+07:00"], [1.5, 1.6], [1.0, 1.0])
    rc = main(["--raw", str(raw), "--out", str(out)])
    assert rc == 0
    proc = pd.read_csv(out / "TZX_processed.csv")
    assert proc["date"].tolist() == ["2021-03-01", "2021-03-02"]


def test_cli_missing_raw_dir_returns_error(tmp_path):
    rc = main(["--raw", str(tmp_path / "does_not_exist"), "--out", str(tmp_path / "out")])
    assert rc == 1
