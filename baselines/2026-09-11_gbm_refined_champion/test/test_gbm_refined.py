"""Unit + smoke tests for the refined-champion feature builders (dummy data, no heavy SP500 load).
Verifies shapes, value ranges, leakage-safe temporal alignment, and the design-matrix column counts."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "baselines" / "2026-09-11_gbm_refined_champion" / "code"))
import gbm_refined_walkforward as R  # noqa: E402

T, N, H = 80, 3, 5


@pytest.fixture
def panel():
    dates = pd.date_range("2020-01-01", periods=T, freq="D")
    rng = np.random.default_rng(0)
    pk = np.abs(rng.normal(1e-4, 5e-5, (T, N)))
    feats = np.zeros((T, N, 5)); feats[:, :, 0] = pk
    anchors = np.arange(30, T - H)
    return SimpleNamespace(tickers=["A", "B", "C"], dates=dates, pk=pk, feats=feats, N=N,
                           anchors=anchors, target_dates=dates.to_numpy()[anchors + H])


def _earn():
    return {"A": np.array(["2020-02-10"], dtype="datetime64[D]"),
            "B": np.array(["2020-02-20"], dtype="datetime64[D]")}


def test_signed_dist_next_and_prev():
    ed = np.array(["2020-01-10", "2020-01-20"], dtype="datetime64[D]")
    tgt = np.array(["2020-01-08", "2020-01-12"], dtype="datetime64[D]")
    nxt, prv = R._signed_dist(tgt, ed)
    assert nxt[0] == 2 and prv[0] == R.GE.CAP        # before first release: next=2, no prior
    assert nxt[1] == 8 and prv[1] == 2               # between: next=8 (to 20th), prev=2 (from 10th)


def test_signed_dist_no_earnings_returns_cap():
    tgt = np.array(["2020-01-08"], dtype="datetime64[D]")
    nxt, prv = R._signed_dist(tgt, None)
    assert nxt[0] == R.GE.CAP and prv[0] == R.GE.CAP


def test_asym_earnings_shapes_and_range(panel):
    asym = R.asym_earnings_panels(panel, _earn())
    assert set(asym) == set(R.EARN4)
    for k in R.EARN4:
        assert asym[k].shape == (len(panel.anchors), N)
        assert asym[k].min() >= 0.0 and asym[k].max() <= 1.0
    # ticker C has no earnings -> all-zero proximity
    assert asym["earn_pre"][:, 2].max() == 0.0 and asym["earn_post"][:, 2].max() == 0.0


def test_spike_panels_shapes_and_bounds(panel):
    spk = R.spike_panels(panel)
    assert set(spk) == set(R.SPIKE_KEYS)
    for k in R.SPIKE_KEYS:
        assert spk[k].shape == (T, N)
    sr = spk["spike_rate63"]
    assert np.nanmin(sr) >= 0.0 and np.nanmax(sr) <= 1.0     # a frequency is in [0,1]


def test_estimator_panels_alignment(panel):
    with tempfile.TemporaryDirectory() as td:
        files = []
        for j, tk in enumerate(panel.tickers):
            p = Path(td) / f"{tk}.csv"; files.append(str(p))
            pd.DataFrame({"date": panel.dates, "garman_klass_variance": panel.pk[:, j],
                          "rogers_satchell_variance": panel.pk[:, j] * 2,
                          "yang_zhang_n20": panel.pk[:, j] * 3}).to_csv(p, index=False)
        est = R.estimator_panels(panel, files)
    assert est.shape == (T, N, 3)
    # RS column (index 1) should be ~2x GK column (index 0)
    np.testing.assert_allclose(est[:, :, 1], est[:, :, 0] * 2, rtol=1e-5)


def test_missing_estimator_column_stays_nan(panel):
    with tempfile.TemporaryDirectory() as td:
        files = []
        for j, tk in enumerate(panel.tickers):
            p = Path(td) / f"{tk}.csv"; files.append(str(p))
            pd.DataFrame({"date": panel.dates, "garman_klass_variance": panel.pk[:, j]}).to_csv(p, index=False)
        est = R.estimator_panels(panel, files)
    assert not np.isnan(est[:, :, 0]).all()          # GK present
    assert np.isnan(est[:, :, 1]).all()              # RS absent -> NaN (HistGBM handles natively)


@pytest.mark.smoke
def test_design_matrix_column_counts(panel):
    extras = R.G.extra_feature_panels(panel.feats)
    asym = R.asym_earnings_panels(panel, _earn())
    spk = R.spike_panels(panel)
    est = np.full((T, N, 3), np.nan, np.float32)
    a = panel.anchors; pos = np.arange(len(a)); har5 = panel.feats[a][:, :, :5]
    Xr = R._design_refined(har5, extras, est, spk, asym, a, pos)
    Xe = R._earn_design(har5, extras, est, spk, asym, a, pos)
    assert Xr.shape == (len(a) * N, 5 + 6 + 3 + 4 + 4)   # refined = 22 cols
    assert Xe.shape == (len(a) * N, 5 + 6 + 2)           # committed GBM+earn = 13 cols


def test_gbm_fit_runs_and_predicts():
    """_gbm_fit builds a gamma-loss GBM and predicts positive values (exercises the fit helper directly)."""
    rng = np.random.default_rng(0)
    x = rng.random((200, 3)); y = np.abs(rng.normal(1e-4, 5e-5, 200)) + R.FL
    m = R._gbm_fit(x, y)
    p = m.predict(x[:10])
    assert p.shape == (10,) and np.all(np.isfinite(p))


def test_design_alignment_values(panel):
    """Value-level regression guard for the dual index space: estimators/spike are indexed by the DATE-axis
    `anchors` (origin t) while earnings are indexed by the positional `pos` (anchor axis). A future anchors<->pos
    swap would keep the shapes but corrupt which row each feature lands on / pull a wrong-day value — this test
    fails on that. Uses distinguishable per-(index,ticker) values so placement is unambiguous."""
    a = panel.anchors; pos = np.arange(len(a))
    extras = {k: np.zeros((T, N)) for k in R.G.EXTRA_KEYS}
    est = np.zeros((T, N, 3), float)
    for t in range(T):
        for j in range(N):
            for c in range(3):
                est[t, j, c] = t * 1000.0 + j * 10.0 + c          # encodes (date-index, ticker, channel)
    spk = {ki: np.fromfunction(lambda t, j, o=oi: t + j * 0.1 + o, (T, N), dtype=float)
           for oi, ki in enumerate(R.SPIKE_KEYS)}
    asym = {ki: np.fromfunction(lambda p, j, o=oi: p * 7.0 + j + o * 0.01, (len(a), N), dtype=float)
            for oi, ki in enumerate(R.EARN4)}
    har5 = panel.feats[a][:, :, :5]
    X = R._design_refined(har5, extras, est, spk, asym, a, pos)
    ai, j = 2, 1                                                  # anchor #2, ticker #1 -> flat row
    r = ai * N + j
    est0 = 5 + 6                                                  # first estimator column index
    # estimator must carry the ORIGIN-date value est[anchors[ai], j, 0], NOT est[ai, j, 0] (the pos value)
    assert X[r, est0] == est[a[ai], j, 0]
    assert X[r, est0] != est[ai, j, 0]                           # a[ai] != ai here, so an anchors<->pos swap fails
    assert X[r, est0 + 3] == spk[R.SPIKE_KEYS[0]][a[ai], j]      # spike also origin-date indexed
    assert X[r, est0 + 7] == asym["earn_prox"][pos[ai], j]       # earnings positional (anchor-axis) indexed
