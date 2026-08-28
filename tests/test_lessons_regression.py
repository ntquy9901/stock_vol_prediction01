"""Consolidated LESSONS-LEARNED regression suite.

Each test codifies a documented lesson (CLAUDE.md and/or a memory note) as an executable invariant that
runs against the REAL repo function (not a re-implementation), so a past bug cannot silently reappear. This
is the single, gate-run home for "lessons -> re-runnable test cases". Pure numpy/pandas/scipy -- no torch,
no data files -- so it runs in the base interpreter and is fast enough for the pre-push gate.

Run: python -m pytest tests/test_lessons_regression.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src" / "common",
           REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "eda"):
    sys.path.insert(0, str(_p))


# --- Lesson: Directional Accuracy = sign of CHANGES, not sign of values (CLAUDE.md 3.B critical bug) ---
def test_diracc_uses_sign_of_changes_not_values():
    """Volatility is non-negative, so sign(y_true)==sign(y_pred) is trivially ~100%. The metric MUST use
    np.diff (sign of day-over-day CHANGE). Guard: a monotone-up target with one wrong-direction step scores
    75%, never 100%."""
    import evaluation as EV
    y_true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])           # strictly increasing
    y_pred = np.array([0.1, 0.05, 0.3, 0.35, 0.5])          # step 1 goes DOWN (wrong direction)
    acc = EV.directional_accuracy(y_true, y_pred)
    acc = acc / 100.0 if acc > 1.5 else acc                 # accept fraction or percent
    assert abs(acc - 0.75) < 1e-9                            # 3/4 correct; the sign-of-values bug would give 1.0


# --- Lesson: Temporal split is CHRONOLOGICAL, never random (CLAUDE.md 3.A, data-leakage prevention) ---
def test_temporal_split_is_chronological_no_overlap():
    import temporal_split as TS
    dates = pd.bdate_range("2015-01-01", periods=100)
    df = pd.DataFrame({"date": dates, "x": np.arange(100.0)})
    tr, va, te = TS.temporal_split_dataframe(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(tr) + len(va) + len(te) == 100
    assert tr["date"].max() < va["date"].min()              # train strictly before val
    assert va["date"].max() < te["date"].min()              # val strictly before test
    assert list(tr["date"]) == sorted(tr["date"])           # order preserved (not shuffled)


def test_temporal_split_fails_loud_on_nat_and_duplicate_dates():
    # Lesson (CLAUDE.md / src comment): mixed tz-aware/tz-naive dates coerce to NaT and duplicate dates in a
    # pooled frame must FAIL LOUD, not be silently sorted around (the VPB/VRE tz bug; pooled-not-split bug).
    import temporal_split as TS
    bad = pd.DataFrame({"date": ["2015-01-01", "not-a-date", "2015-01-03"], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        TS.temporal_split_dataframe(bad)
    dup = pd.DataFrame({"date": ["2015-01-01", "2015-01-01", "2015-01-02"], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        TS.temporal_split_dataframe(dup)


# --- Lesson: the `parkinson_volatility` target is VARIANCE (sigma^2), not sigma (memory: parkinson_target_is_variance) ---
def test_parkinson_target_is_variance_not_stddev():
    import volatility_estimators as VE
    # H/L = e  ->  ln(H/L) = 1  ->  Parkinson VARIANCE = 1/(4 ln2) ~ 0.36067 (NOT sqrt of it ~ 0.6006)
    lo = 10.0
    hi = 10.0 * np.e
    df = pd.DataFrame({"date": [1], "open": [15.0], "high": [hi], "low": [lo], "close": [15.0], "volume": [1]})
    park = VE.estimators_from_ohlcv(df)["parkinson"].iloc[0]
    assert abs(park - 1.0 / (4.0 * np.log(2.0))) < 1e-9      # variance-scale
    assert abs(park - np.sqrt(1.0 / (4.0 * np.log(2.0)))) > 0.2  # NOT the std-dev scale


# --- Lesson: QLIKE clamps BOTH target and prediction to one SHARED positivity floor (memory: eda_gnn_result H2) ---
def test_qlike_shared_floor_and_zero_at_exact():
    import metrics as M
    y = np.array([0.02, 0.05, 0.1, 0.2])
    assert M.qlike(y, y.copy()) == pytest.approx(0.0, abs=1e-12)   # exact forecast -> 0
    # both operands clamped to the SAME floor: tiny/zero/negative targets stay finite, never -inf/log(0)
    yt = np.array([0.0, -1e-20, 1e-15, 0.05])
    pt = np.array([1e-20, 0.0, 1e-12, 0.05])
    assert np.isfinite(M.qlike(yt, pt))
    # the floor must be applied to the TARGET too (asymmetric flooring would make QLIKE non-comparable):
    # comparing a model to itself is always exactly 0 regardless of how tiny the values are
    assert M.qlike(np.array([1e-30, 1e-30]), np.array([1e-30, 1e-30])) == pytest.approx(0.0, abs=1e-12)


# --- Lesson: date-clustered DM (HLN) does NOT overstate significance the way naive per-obs does on a
#     dependent panel (~sqrt(#stocks) inflation) (paper methodology; memory: eda_gnn_result / paper targets) ---
def test_date_clustered_dm_less_significant_than_naive_per_obs():
    import metrics as M
    import stats as ST
    n_stocks, n_dates = 20, 30
    rng = np.random.default_rng(0)
    # a modest, real per-DATE loss advantage shared by ALL stocks that day (within-date dependence)
    delta_date = 0.02 + rng.normal(0, 0.1, n_dates)
    loss_b, loss_a, dates = [], [], []
    for d in range(n_dates):
        base = np.abs(rng.normal(1.0, 0.05, n_stocks))
        loss_a.extend(base)
        loss_b.extend(base + delta_date[d])                 # b larger -> A favoured, same delta all stocks
        dates.extend([d] * n_stocks)
    loss_a = np.array(loss_a); loss_b = np.array(loss_b); dates = np.array(dates)
    naive = M.diebold_mariano(loss_a, loss_b, h=1)           # treats 600 obs as independent -> OVERSTATES
    clustered = ST.date_clustered_dm(loss_a, loss_b, dates, h=1)  # aggregates to 30 date-means -> honest
    assert clustered["n_dates"] == n_dates
    assert clustered["p_value"] > naive.p_value              # clustered is LESS significant (no inflation)


# --- Lesson: a normalizer must be fit AND APPLIED, and the round-trip must invert (CLAUDE.md 5 /
#     memory: normalizer_fit_never_applied_recurrence -- "fit scaler then forget .transform()") ---
def test_volatility_normalizer_transform_applied_and_invertible():
    import data_normalization as DN
    train = np.array([0.02, 0.03, 0.05, 0.08, 0.1, 0.12]).reshape(-1, 1)
    norm = DN.VolatilityNormalizer()
    norm.fit(train)
    z = norm.transform(train)
    assert not np.allclose(z, train)                        # transform ACTUALLY changed the data (not a no-op)
    assert abs(float(np.mean(z))) < 1e-6                     # standardized ~ mean 0
    back = norm.inverse_transform(z)
    np.testing.assert_allclose(back, train, rtol=1e-9, atol=1e-12)   # round-trip recovers the original scale
