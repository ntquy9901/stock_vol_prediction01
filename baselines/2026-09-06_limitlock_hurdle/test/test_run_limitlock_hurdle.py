"""Unit tests for the limit-lock hurdle experiment's pure helpers (TEST-FIRST)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

_spec = importlib.util.spec_from_file_location("_llh", HERE.parent / "code" / "run_limitlock_hurdle.py")


def _load():
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)
    return m


def test_ols_recovers_linear_and_floors():
    m = _load()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y = 3.0 + 2.0 * x[:, 0] - 1.0 * x[:, 1]
    assert np.allclose(m.ols_fit_predict(x, y, x, floor=-1e9), y, atol=1e-6)
    assert (m.ols_fit_predict(x, y * 0 - 5.0, x, floor=0.0) >= 0.0).all()


def test_har_design_flattens():
    m = _load()
    har5 = np.arange(4 * 3 * 5, dtype=float).reshape(4, 3, 5)
    d = m._har_design(har5)
    assert d.shape == (12, 5)
    assert np.allclose(d[0], har5[0, 0])


def test_logistic_separates_classes():
    m = _load()
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((100, 3)) - 2.0
    x1 = rng.standard_normal((100, 3)) + 2.0
    x = np.vstack([x0, x1]); y = np.r_[np.zeros(100), np.ones(100)]
    proba = m.logistic_fit_predict_proba(x, y, x)
    assert proba.shape == (200,)
    assert (proba[:100] < 0.5).mean() > 0.9 and (proba[100:] >= 0.5).mean() > 0.9


def test_logistic_single_class_returns_constant_base_rate():
    m = _load()
    x = np.random.default_rng(2).standard_normal((50, 3))
    proba = m.logistic_fit_predict_proba(x, np.zeros(50), x)
    assert np.allclose(proba, 0.0)              # base rate of an all-negative train target
    proba1 = m.logistic_fit_predict_proba(x, np.ones(50), x)
    assert np.allclose(proba1, 1.0)


def test_apply_hurdle_overrides_above_threshold():
    m = _load()
    harx = np.array([1.0, 2.0, 3.0, 4.0])
    proba = np.array([0.1, 0.8, 0.5, 0.95])
    out = m.apply_hurdle(harx, proba, threshold=0.7, lock_value=0.01)
    assert np.allclose(out, [1.0, 0.01, 3.0, 0.01])
    # per-node lock_value broadcasts
    out2 = m.apply_hurdle(harx.reshape(2, 2), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          threshold=0.5, lock_value=np.array([0.02, 0.03]))
    assert np.allclose(out2, [[0.02, 2.0], [3.0, 0.03]])


def test_subset_inside_and_outside():
    m = _load()
    pred = {(0, "d1"): (1.0, 1.1), (0, "d2"): (2.0, 2.1), (1, "d1"): (3.0, 3.1)}
    lock = {(0, "d2")}
    assert set(m._subset(pred, lock, inside=True)) == {(0, "d2")}
    assert set(m._subset(pred, lock, inside=False)) == {(0, "d1"), (1, "d1")}


def test_confusion_counts():
    m = _load()
    yt = np.array([1, 1, 0, 0, 1])
    proba = np.array([0.9, 0.4, 0.8, 0.1, 0.95])
    c = m._confusion(yt, proba, threshold=0.5)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_pooled_maps_masked_entries():
    m = _load()
    D = SimpleNamespace(
        y_te=np.array([[1.0, 2.0], [3.0, 4.0]]),
        tmask_te=np.array([[True, False], [True, True]]),
        d_te=np.array(["2026-01-01", "2026-01-02"]),
        N=2,
    )
    pred_flat = np.array([10.0, 20.0, 30.0, 40.0])
    pooled = m._pooled(pred_flat, D, "te")
    assert set(pooled) == {(0, "2026-01-01"), (0, "2026-01-02"), (1, "2026-01-02")}
    assert pooled[(0, "2026-01-01")] == (1.0, 10.0)


def test_metrics_full_splits_lock_and_nonlock():
    m = _load()
    floor = 1e-8
    pred = {(0, "d1"): (0.001, 0.001), (0, "d2"): (0.0, 0.002), (1, "d1"): (0.0012, 0.0011)}
    lock = {(0, "d2")}
    mm = m._metrics_full(pred, floor, lock)
    assert mm["n_lockdays"] == 1
    assert mm["qlike_robust"] is not None and mm["qlike_lockdays"] is not None
    # lock-day QLIKE (over-forecast on a zero target) is far worse than the non-lock-day QLIKE
    assert mm["qlike_lockdays"] > mm["qlike_robust"]
    # empty lock set -> qlike_lockdays None, robust defined
    mm2 = m._metrics_full(pred, floor, set())
    assert mm2["qlike_lockdays"] is None and mm2["qlike_robust"] is not None and mm2["n_lockdays"] == 0


def test_read_aligned_reindexes_to_panel(tmp_path):
    m = _load()
    import pandas as pd
    csv = tmp_path / "AAA.csv"
    pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-07"],
                  "daily_return": [0.01, -0.07, 0.02],
                  "zero_range_flag": [False, True, False]}).to_csv(csv, index=False)
    other = tmp_path / "ZZZ.csv"     # a file whose ticker is NOT in the panel -> must be skipped
    pd.DataFrame({"date": ["2025-01-02"], "daily_return": [0.0], "zero_range_flag": [False]}).to_csv(other, index=False)
    dates = pd.DatetimeIndex(["2025-01-02", "2025-01-03", "2025-01-06"])   # 01-06 not in the file
    panel = SimpleNamespace(tickers=["AAA"], dates=dates, N=1)
    ret, lock = m._read_aligned([str(csv), str(other)], panel)
    assert ret.shape == (3, 1) and lock.shape == (3, 1)
    assert ret[0, 0] == 0.01 and lock[1, 0] == 1.0
    assert np.isnan(ret[2, 0]) and np.isnan(lock[2, 0])   # off-file date -> NaN


def test_read_aligned_and_panel_features_on_real_vn30_slice():
    """Real-data-sample smoke (per the project's data-pipeline test rule): read a small slice of the real
    VN30 FPT file and confirm the 2025-04-10 limit-lock day is flagged and the causal features are finite."""
    m = _load()
    import pandas as pd
    repo = HERE.parents[2]
    fpt = repo / "data" / "processed_enriched" / "vn30" / "FPT.csv"
    if not fpt.exists():                                    # pragma: no cover - only when data absent
        import pytest
        pytest.skip("real VN30 data not present")
    df = pd.read_csv(fpt, parse_dates=["date"])
    sl = df[(df["date"] >= "2025-03-01") & (df["date"] <= "2025-04-15")]
    dates = pd.DatetimeIndex(sl["date"])
    panel = SimpleNamespace(tickers=["FPT"], dates=dates, N=1)
    ret, lock = m._read_aligned([str(fpt)], panel)
    ll = m._panel_limitlock(ret, lock, 1)
    assert ll.shape == (len(dates), 1, 3) and np.isfinite(ll).all()
    di = list(dates).index(pd.Timestamp("2025-04-10"))
    assert lock[di, 0] == 1.0                               # 2025-04-10 is a real zero-range lock day
    assert ll[di, 0, 2] > 0.0                               # near_limit_freq elevated by the preceding limit moves


def test_hurdle_improves_qlike_on_synthetic_lock_day():
    """A synthetic lock day (true var ~0) that HAR-X over-forecasts: overriding it to the floor must cut
    the pooled QLIKE (the core hypothesis, in miniature)."""
    m = _load()
    import metrics as M
    floor = 1e-8
    # 4 normal cells (HAR-X ~ correct) + 1 lock cell (true 0, HAR-X forecasts 0.002)
    y = np.array([0.001, 0.0012, 0.0009, 0.0011, 0.0])
    harx = np.array([0.001, 0.0012, 0.0009, 0.0011, 0.002])
    proba = np.array([0.0, 0.0, 0.0, 0.0, 0.99])          # classifier flags only the lock cell
    hurdle = m.apply_hurdle(harx, proba, threshold=0.7, lock_value=1e-6)
    assert M.qlike(y, hurdle, floor) < M.qlike(y, harx, floor)
