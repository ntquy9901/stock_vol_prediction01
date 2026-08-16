import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.verify_raw_prices import verify_dir, verify_frame  # noqa: E402


def _df(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def test_clean_frame_reports_hard_ok():
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],   # Mon
        ["2020-01-07", 6.0, 8.0, 5.0, 7.0, 0],      # Tue, zero-volume flat? no (h!=low) -> price moved
    ])
    r = verify_frame(df)
    assert r["hard_ok"] is True
    assert r["zero_vol"] == 1
    assert r["zero_vol_price_moved"] == 1     # row 2 has volume 0 but high != low
    assert r["zero_vol_flat"] == 0


def test_zero_volume_flat_is_benign_diagnostic():
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],
        ["2020-01-07", 7.0, 7.0, 7.0, 7.0, 0],      # flat AND zero volume -> no-trade day
    ])
    r = verify_frame(df)
    assert r["hard_ok"] is True                # not a hard defect
    assert r["zero_vol_flat"] == 1 and r["zero_vol_price_moved"] == 0
    assert r["leading_zero_vol_run"] == 0      # zero-vol is the 2nd row, not leading


def test_float_noise_flat_is_not_price_moved():
    # high==low to float noise (~1e-15) with zero volume -> flat no-trade, NOT zero_vol_price_moved
    df = _df([
        ["2020-01-06", 50.0, 50.0, 50.0, 50.0, 100],
        ["2020-01-07", 50.0, 50.0 + 7e-15, 50.0, 50.0, 0],
    ])
    r = verify_frame(df)
    assert r["zero_vol_price_moved"] == 0
    assert r["zero_vol_flat"] == 1


def test_hard_defects_flagged():
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],
        ["2020-01-04", 6.0, 8.0, 5.0, 7.0, 100],   # Sat + non-monotonic date
        ["2020-01-07", -1.0, 8.0, 5.0, 7.0, 100],  # nonpositive open
    ])
    r = verify_frame(df)
    assert r["hard_ok"] is False
    assert r["date_weekend"] >= 1
    assert r["nonpositive"] >= 1


def test_leading_zero_volume_run_detected():
    rows = [["2020-01-06", 5.0, 5.0, 5.0, 5.0, 0]] * 25 + [["2020-02-10", 6.0, 8.0, 5.0, 7.0, 100]]
    # give distinct increasing weekday dates to isolate the leading-run check
    dates = pd.bdate_range("2020-01-06", periods=26).strftime("%Y-%m-%d")
    df = _df([[d] + rows[i][1:] for i, d in enumerate(dates)])
    r = verify_frame(df)
    assert r["leading_zero_vol_run"] == 25


def test_schema_mismatch_short_circuits():
    df = pd.DataFrame({"date": ["2020-01-06"], "close": [1.0]})
    r = verify_frame(df)
    assert r["schema_ok"] is False and r["hard_ok"] is False


def test_verify_dir(tmp_path):
    _df([["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100]]).to_csv(tmp_path / "AAA_ohlcv.csv", index=False)
    res = verify_dir(tmp_path)
    assert "AAA" in res and res["AAA"]["hard_ok"] is True
