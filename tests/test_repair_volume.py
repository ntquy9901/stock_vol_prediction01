import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.repair_volume import repair_dir, repair_frame  # noqa: E402


def _df(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def test_glitch_volume_interpolated():
    # middle row: volume 0 but high != low (price moved) -> interpolated from neighbours (100, 300)
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],
        ["2020-01-07", 6.0, 8.0, 5.0, 7.0, 0],
        ["2020-01-08", 6.0, 8.0, 5.0, 7.0, 300],
    ])
    out, n = repair_frame(df)
    assert n == 1
    assert out["volume"].iloc[1] == 200      # linear interpolation between 100 and 300
    assert (out["volume"] > 0).all()


def test_float_noise_flat_not_repaired():
    # high == low to floating-point noise (~1e-15) is a FLAT no-trade day, NOT a price move ->
    # volume 0 must stay 0 (regression: exact `high != low` wrongly repaired these).
    df = _df([
        ["2020-01-06", 50.0, 50.0, 50.0, 50.0, 100],
        ["2020-01-07", 50.0, 50.0 + 7e-15, 50.0, 50.0, 0],   # noise range, no real move
    ])
    out, n = repair_frame(df)
    assert n == 0
    assert out["volume"].iloc[1] == 0


def test_legit_flat_zero_volume_untouched():
    # volume 0 AND high == low -> real no-trade day, must stay 0
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],
        ["2020-01-07", 7.0, 7.0, 7.0, 7.0, 0],   # flat, no trade
    ])
    out, n = repair_frame(df)
    assert n == 0
    assert out["volume"].iloc[1] == 0


def test_idempotent():
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],
        ["2020-01-07", 6.0, 8.0, 5.0, 7.0, 0],
        ["2020-01-08", 6.0, 8.0, 5.0, 7.0, 300],
    ])
    out1, _ = repair_frame(df)
    _, n2 = repair_frame(out1)
    assert n2 == 0


def test_leading_glitch_filled_from_right():
    # glitch at the very start (no left neighbour) -> back-filled from the right
    df = _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 0],    # glitch, price moved, no left neighbour
        ["2020-01-07", 6.0, 8.0, 5.0, 7.0, 500],
    ])
    out, n = repair_frame(df)
    assert n == 1
    assert out["volume"].iloc[0] == 500


def test_repair_dir(tmp_path):
    _df([
        ["2020-01-06", 6.0, 8.0, 5.0, 7.0, 100],
        ["2020-01-07", 6.0, 8.0, 5.0, 7.0, 0],
        ["2020-01-08", 6.0, 8.0, 5.0, 7.0, 300],
    ]).to_csv(tmp_path / "AAA_ohlcv.csv", index=False)
    res = repair_dir(tmp_path)
    assert res["AAA"] == 1
    assert (pd.read_csv(tmp_path / "AAA_ohlcv.csv")["volume"] > 0).all()
