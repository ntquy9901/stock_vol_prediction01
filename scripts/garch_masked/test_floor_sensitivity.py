"""Tests for the common-floor re-scoring helper (scripts/garch_masked/floor_sensitivity.py)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))
import floor_sensitivity as F  # noqa: E402


def test_score_common_floor_uniform_and_clip_rate():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    raw = np.array([0.5, 2.0, 3.0, -1.0])         # 2 of 4 below the floor (0.5 and -1.0)
    floor = np.full(4, 1.0)
    m, clip = F.score_common_floor(y, raw, floor)
    assert abs(clip - 0.5) < 1e-12                 # 2/4 clipped
    # eval = [1,2,3,1]; vs y=[1,2,3,4] -> only the last differs (4-1)^2=9 -> mse=9/4
    assert abs(m["mse"] - 9.0 / 4.0) < 1e-9
    assert np.isfinite(m["qlike"]) and np.isfinite(m["r2"])


def test_screen_files_liquidity_and_history(tmp_path):
    import pandas as pd
    def _mk(name, n, zero_frac):
        v = np.abs(np.random.default_rng(0).normal(2e-4, 5e-5, n))
        v[: int(n * zero_frac)] = 0.0
        f = tmp_path / f"{name}_processed.csv"
        pd.DataFrame({"date": range(n), "parkinson_volatility": v}).to_csv(f, index=False)
        return str(f)
    keep = _mk("KEEP", 300, 0.10)      # long + liquid -> kept
    short = _mk("SHORT", 100, 0.10)    # < 250 rows -> dropped
    illiq = _mk("ILLIQ", 300, 0.60)    # > 50% zero-var -> dropped
    out = F.screen_files([illiq, short, keep])   # unsorted input
    assert out == [keep]


def test_score_common_floor_no_clip_when_all_above():
    y = np.array([1.0, 2.0, 3.0])
    raw = np.array([1.5, 2.5, 3.5])
    floor = np.full(3, 1e-8)
    m, clip = F.score_common_floor(y, raw, floor)
    assert clip == 0.0                              # nothing below 1e-8
    # eval == raw (unchanged) -> mse = mean((y-raw)^2) = mean([0.25,0.25,0.25]) = 0.25
    assert abs(m["mse"] - 0.25) < 1e-9
