"""Tests for the HOSE spike-robustness analysis (pure logic: shock mask + analyze). The real-data
walk-forward driver `_perobs` is pragma-excluded; its output shape is exercised here via a synthetic stream."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hose_spike_robustness as SR  # noqa: E402


def test_shock_mask_windows():
    dates = pd.to_datetime(
        ["2019-01-01", "2020-03-15", "2022-06-01", "2025-04-15", "2024-01-01"]).to_numpy()
    m = SR._shock_mask(dates)
    assert list(m) == [False, True, True, True, False]        # COVID / 2022 / Apr-2025 windows flagged


def _stream(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6
    gbm = np.maximum(y + rng.standard_normal(n) * 1e-6, 1e-8)     # GBM close to y
    har = np.maximum(y + rng.standard_normal(n) * 1e-5, 1e-8)     # HAR noisier -> worse
    dates = pd.bdate_range("2019-06-01", periods=n).to_numpy()    # spans a shock window
    fold = np.repeat(np.arange(4), n // 4)
    return {1: (y, gbm, har, dates, fold)}


def test_analyze_structure_and_verdict():
    res = SR.analyze(_stream())
    r = res["h1"]
    assert set(r) >= {"n", "n_excluded_shock", "qlike_all", "qlike_excl_shock", "gbm_vs_har_gain_pct",
                      "gbm_vs_har_dm_p", "verdict_survives", "per_fold_qlike", "storm_decile"}
    assert len(r["per_fold_qlike"]) == 4                          # one entry per fold
    assert 0 < r["n_excluded_shock"] < r["n"]                     # some (not all) rows fall in a shock window
    assert set(r["gbm_vs_har_gain_pct"]) == {"all", "excl_shock"}
    assert set(r["storm_decile"]) == {"top_decile_gbm_qlike", "calm_decile_gbm_qlike",
                                      "top_decile_share_of_total"}
    assert isinstance(r["verdict_survives"], bool)


def test_analyze_shock_exclusion_changes_n():
    res = SR.analyze(_stream())
    r = res["h1"]
    # excluding shock rows must reduce the scored count used for the excl-shock QLIKE
    assert r["n_excluded_shock"] >= 1
    assert r["qlike_all"]["gbm"] > 0 and r["qlike_excl_shock"]["gbm"] > 0
