import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean_ohlc import clean_dir, clean_frame  # noqa: E402


def _row(o, h, low, c, date="2020-01-01", vol=100):
    return {"date": date, "open": o, "high": h, "low": low, "close": c, "volume": vol}


def _valid(df):
    v = df[["open", "high", "low", "close"]].astype(float)
    return bool(
        (v > 0).all().all()
        and (v["high"] >= v["low"]).all()
        and (v["high"] >= v[["open", "close"]].max(axis=1)).all()
        and (v["low"] <= v[["open", "close"]].min(axis=1)).all()
    )


def test_fixes_high_lt_low():
    df = pd.DataFrame([_row(5.0, 3.0, 8.0, 6.0)])   # high 3 < low 8
    out, changed = clean_frame(df)
    assert changed > 0 and _valid(out)
    assert out["high"].iloc[0] == 8.0 and out["low"].iloc[0] == 3.0


def test_fixes_nonpositive_low_excludes_zero():
    df = pd.DataFrame([_row(5.11, 14.61, 0.0, 6.66)])   # low 0 must NOT propagate
    out, changed = clean_frame(df)
    assert changed > 0 and _valid(out)
    assert out["low"].iloc[0] == 5.11                    # min of positive {5.11,14.61,6.66}


def test_fixes_open_close_outside_range():
    df = pd.DataFrame([_row(9.0, 8.0, 7.0, 6.0)])        # open 9 > high 8
    out, _ = clean_frame(df)
    assert _valid(out) and out["high"].iloc[0] == 9.0


def test_valid_row_unchanged():
    df = pd.DataFrame([_row(6.0, 8.0, 5.0, 7.0)])
    out, changed = clean_frame(df)
    assert changed == 0 and _valid(out)


def test_idempotent():
    df = pd.DataFrame([_row(5.0, 3.0, 8.0, 6.0, "2020-01-01"),
                       _row(3.0, 4.0, 2.0, 3.5, "2020-01-02")])
    out1, _ = clean_frame(df)
    _, changed2 = clean_frame(out1)
    assert changed2 == 0


def test_all_nonpositive_row_left_untouched():
    df = pd.DataFrame([_row(0.0, 0.0, 0.0, 0.0)])        # unreconstructable
    out, changed = clean_frame(df)
    assert changed == 0                                  # left as-is, not silently zeroed further


def test_clean_dir_in_place(tmp_path):
    p = tmp_path / "XYZ_ohlcv.csv"
    pd.DataFrame([_row(5.0, 3.0, 8.0, 6.0)]).to_csv(p, index=False)
    res = clean_dir(tmp_path)
    assert res["XYZ"] > 0
    assert _valid(pd.read_csv(p))
