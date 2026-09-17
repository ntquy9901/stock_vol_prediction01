"""Tests for the Gamma-GLM base-margin anchor + XGBoost gamma residual baseline (Hướng 3): GLM log-link eta,
base-margin propagation, model plumbing, DM/verdict logic, spike mask, and a walk-forward run smoke with the
gate-required over/under-fit evidence keys. Synthetic panels keep the driver fast without real data."""
import numpy as np
import pandas as pd

import glm_anchor_config as C
import glm_anchor as GA
import run_glm_anchor as R
import vn_gbm_graph_stage1 as S1


# --------------------------------------------------------------------------- synthetic data
def _ticker_frame(seed, start="2021-01-01", end="2023-06-30"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    pk = np.empty(n)
    pk[0] = 3e-4
    for t in range(1, n):
        pk[t] = max(1e-6, 0.85 * pk[t - 1] + 0.15 * 3e-4 + rng.normal(0, 3e-5))
    s = pd.Series(pk)
    base = pd.DataFrame({
        "date": dates, "parkinson_variance": pk,
        "har_daily": pk, "har_weekly": s.rolling(5, min_periods=1).mean().to_numpy(),
        "har_monthly": s.rolling(22, min_periods=1).mean().to_numpy(),
        "volume_zscore_22": rng.normal(0, 1, n), "market_pk": pk * 0.9,
        "daily_return": rng.normal(0, 0.01, n)})
    return S1._feat(base)


def _frames(n_tickers=3):
    return {f"TK{i}": _ticker_frame(seed=i).assign(ticker=f"TK{i}", sector=0) for i in range(n_tickers)}


def _fake_loader(market):
    return _frames(), {f"TK{i}": 0 for i in range(3)}, {}


def _tiny(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


# --------------------------------------------------------------------------- GLM eta + base margin
def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    y = np.exp(-8.0 + 0.5 * x[:, 0])                       # positive target, log-linear in x0
    df = pd.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "y": y})
    return df.iloc[:300], df.iloc[300:]


def test_fit_glm_eta_loglink_and_causal():
    tr, te = _toy()
    combo = pd.concat([te, tr])
    eta_tr, eta_co = GA.fit_glm_eta(tr, combo, ["f0", "f1"])
    assert eta_tr.shape == (len(tr),) and eta_co.shape == (len(combo),)
    assert np.isfinite(eta_tr).all() and np.isfinite(eta_co).all()
    # log-link GLM recovers the log-linear signal: eta increases with f0 (positive coef)
    lo = combo["f0"] < combo["f0"].quantile(0.25)
    hi = combo["f0"] > combo["f0"].quantile(0.75)
    assert eta_co[hi.to_numpy()].mean() > eta_co[lo.to_numpy()].mean()


def test_xgb_gamma_uses_base_margin_as_link_offset():
    # if the target already equals exp(margin), the residual trees learn ~0 and predictions track exp(margin)
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"f0": rng.normal(0, 1, 400), "f1": rng.normal(0, 1, 400)})
    m_tr = -8.0 + 0.3 * tr["f0"].to_numpy()
    tr["y"] = np.exp(m_tr)
    te = tr.iloc[:50].copy()
    m_te = m_tr[:50]
    pred = GA.xgb_gamma(tr, te, ["f0", "f1"], seed=0, base_margin_tr=m_tr, base_margin_co=m_te)
    assert np.corrcoef(np.log(pred), m_te)[0, 1] > 0.9      # prediction tracks the base margin
    # and a plain fit (no margin) gives a different, still-positive prediction
    plain = GA.xgb_gamma(tr, te, ["f0", "f1"], seed=0)
    assert (plain > 0).all() and not np.allclose(plain, pred)


def test_xgb_gamma_predictions_stay_within_floor_and_cap():
    # numerical guard (code_review finding 2): even an extreme base margin that would overflow the exp-link
    # must yield finite predictions clipped to [FL, PRED_CAP].
    rng = np.random.default_rng(2)
    tr = pd.DataFrame({"f0": rng.normal(0, 1, 300), "f1": rng.normal(0, 1, 300)})
    tr["y"] = np.abs(rng.normal(0, 1, 300)) * 1e-4 + 1e-6
    te = tr.iloc[:40].copy()
    extreme = np.full(len(tr), 50.0)                           # huge log-margin -> exp overflow without the cap
    p = GA.xgb_gamma(tr, te, ["f0", "f1"], seed=0, base_margin_tr=extreme, base_margin_co=extreme[:40])
    assert np.isfinite(p).all() and (p >= GA.FL).all() and (p <= C.PRED_CAP + 1e-12).all()


def test_predict_xgb_and_glm_xgb_positive():
    tr, te = _toy()
    combo = pd.concat([te, tr])
    for fn in (GA.predict_xgb, GA.predict_glm_xgb):
        p = fn(tr, combo, ["f0", "f1"], (0, 1))
        assert p.shape == (len(combo),) and (p > 0).all() and np.isfinite(p).all()


# --------------------------------------------------------------------------- pure logic
def test_glm_alpha_is_sklearn_default():
    # regression guard (code_review finding 8): GLM_ALPHA must stay at sklearn GammaRegressor's default (1.0);
    # a lighter penalty under-regularises the z-scored eta -> over-dispersion -> negative MSE-R2 that trips gate.
    from sklearn.linear_model import GammaRegressor
    assert C.GLM_ALPHA == GammaRegressor().alpha == 1.0


def test_verdict_and_success():
    assert R.verdict(0.5, 0.01) and not R.verdict(-0.1, 0.01) and not R.verdict(0.5, 0.20)
    assert R.success({1: {"verdict": {"beats": True}}, 5: {"verdict": {"beats": True}}})
    assert not R.success({1: {"verdict": {"beats": True}}})            # missing h5 -> fail


def test_safe_dm_degenerate_and_valueerror(monkeypatch):
    e = np.array([1.0, 2.0, 3.0, 4.0]); d = pd.to_datetime(["2021-01-01"] * 4).to_numpy()
    assert R._safe_dm(e, e.copy(), d, 1)["p_value"] == 1.0            # identical loss -> degenerate p=1
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda *a, **k: (_ for _ in ()).throw(ValueError("hln")))
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 1.0             # DM raises -> degenerate


def test_spike_mask_flags_windows():
    d = pd.to_datetime(["2019-06-01", "2020-03-15", "2022-06-01", "2025-04-10"]).to_numpy()
    assert list(R._spike_mask(d)) == [False, True, True, True]


def test_load_earn_sp500_passthrough_and_hose_missing(monkeypatch, tmp_path):
    assert R._load_earn("sp500", {"A": 1}) == {"A": 1}                # sp500 keeps its own edates
    monkeypatch.setattr(R, "REPO", tmp_path)                         # no crawled parquet under tmp
    assert R._load_earn("hose", {}) == {}                            # hose w/o parquet -> edates unchanged


def test_metrics5_keys():
    y = np.array([1e-4, 2e-4, 3e-4]); p = np.array([1.1e-4, 1.9e-4, 3.2e-4])
    assert set(R._metrics5(y, p)) == {"mse", "rmse", "mae", "r2", "qlike"}


# --------------------------------------------------------------------------- run smoke
def test_run_hose_structure_evidence_perfold_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    # non-overlapping spike window so the fold's test dates survive exclusion -> spike_robustness is computed
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("1990-01-01", "1990-01-02"),))
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert (tmp_path / "glm_anchor_hose_smoke_h1.json").exists()
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.ORDER:
            assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert set(doc["dm"]) == {"GLM+XGB_vs_GBME", "GLM+XGB_vs_XGB"}
    assert "per_fold_qlike" in doc and "spike_robustness" in doc and "verdict" in doc
    import overfit_check as OF
    _ok, probs = OF.check_result_evidence(doc)
    assert all("missing" not in p for p in probs), probs


def test_run_sp500_no_spike_block(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    doc = docs[1]
    assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
    assert (tmp_path / "glm_anchor_sp500_smoke_h1.json").exists()


def test_run_skips_empty_folds_multi(monkeypatch, tmp_path):
    # non-smoke: synthetic data ends mid-2023, so later S1.FOLDS have empty test windows -> the skip `continue`
    # fires while early folds run (multi-fold pooling).
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "HORIZONS", (1,))
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False)
    assert docs[1]["n_folds"] >= 1


def test_run_omits_robustness_when_all_test_in_spike(monkeypatch, tmp_path):
    # default 2022 spike window covers all of fold0's test dates -> keep empty -> spike_robustness omitted
    _tiny(monkeypatch)
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]
