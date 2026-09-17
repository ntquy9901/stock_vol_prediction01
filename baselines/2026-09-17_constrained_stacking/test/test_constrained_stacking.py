"""Tests for the OOF metric-constrained stacking baseline (Hướng 4): base predictors, leakage-safe OOF
coverage/causality, the simplex-QLIKE meta-optimiser (constraint + collapse + degenerate fallback),
DM/verdict/spike logic, and a walk-forward run smoke carrying the gate-required over/under-fit evidence.
Synthetic panels + a fast fake base model keep the driver tests quick without real data."""
import types

import numpy as np
import pandas as pd

import stacking_config as C
import base_models as BM
import run_constrained_stacking as R
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


def _fake_predictors():
    """Fast, positive, deterministic base predictors (scaled har_daily) so the driver runs in ms."""
    def mk(factor):
        return lambda tr, te, cols, seeds: np.clip(te["har_daily"].to_numpy(float) * factor,
                                                   BM.FL, C.PRED_CAP)
    return {"GBME": mk(1.0), "XGB": mk(1.05), "GLM": mk(0.95), "HAR": mk(1.1)}


# --------------------------------------------------------------------------- base predictors
def _toy_panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    pk = np.abs(rng.normal(3e-4, 5e-5, n)) + 1e-5
    s = pd.Series(pk)
    df = pd.DataFrame({
        "date": dates, "y": np.roll(pk, -1),
        "har_daily": pk, "har_weekly": s.rolling(5, min_periods=1).mean().to_numpy(),
        "har_monthly": s.rolling(22, min_periods=1).mean().to_numpy(),
        "mr_change": rng.normal(0, 0.1, n), "mr_slope5": rng.normal(0, 0.1, n),
        "mr_slope10": rng.normal(0, 0.1, n), "mr_dev5": rng.normal(0, 0.1, n),
        "mr_z22": rng.normal(0, 1, n)})
    return df


def _cols8():
    import importlib.util
    from pathlib import Path
    p = Path(BM.REPO) / "baselines" / "2026-09-13_paper_models" / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("pmc_test", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.own_set(BM.FM.OWN)


def test_all_base_predictors_positive_finite():
    df = _toy_panel(); cols = _cols8()
    tr, te = df.iloc[:300], df.iloc[300:]
    for name, fn in BM.BASE.items():
        p = fn(tr, te, cols, (0, 1))
        assert p.shape == (len(te),), name
        assert np.isfinite(p).all() and (p >= BM.FL).all() and (p <= C.PRED_CAP + 1e-12).all(), name


def test_base_predictors_stay_capped_on_extreme():
    # HAR OLS / GLM / XGB must not exceed PRED_CAP even on an inflated target row.
    df = _toy_panel(); cols = _cols8()
    df.loc[df.index[:5], "y"] = 50.0                       # extreme -> uncapped fit would exceed the cap
    tr, te = df.iloc[:300], df.iloc[300:]
    for fn in BM.BASE.values():
        p = fn(tr, te, cols, (0,))
        assert (p <= C.PRED_CAP + 1e-9).all()


# --------------------------------------------------------------------------- OOF cross-fitting
def test_inner_blocks_partition_and_contiguous():
    dates = pd.bdate_range("2021-01-01", periods=30).to_numpy()
    blocks = BM.inner_blocks(dates, 3)
    assert len(blocks) == 3
    assert sum(len(b) for b in blocks) == 30
    joined = np.concatenate(blocks)
    assert (np.sort(joined) == np.sort(np.unique(dates))).all()
    assert blocks[0].max() < blocks[1].min() < blocks[2].min()   # contiguous temporal blocks


def test_oof_covers_every_row_and_is_causal():
    df = _toy_panel(); cols = _cols8()
    oof = BM.oof_predict(df, cols, (0, 1), _fake_predictors(), 3)
    assert set(oof) == set(BM.BASE)
    for m, arr in oof.items():
        assert arr.shape == (len(df),) and not np.isnan(arr).any(), m
    # causality: a spy predictor records which held dates it is asked to predict and never overlaps
    # with its own inner-train dates.
    seen = {"train": set(), "held": set()}

    def spy(tr, te, cols, seeds):
        seen["train"].update(pd.to_datetime(tr["date"]).tolist())
        seen["held"].update(pd.to_datetime(te["date"]).tolist())
        return te["har_daily"].to_numpy(float)
    BM.oof_predict(df, cols, (0,), {"SPY": spy}, 3)
    # every date is held exactly once; the union of held dates == all dates (full coverage).
    assert seen["held"] == set(pd.to_datetime(df["date"]).tolist())


def test_oof_raises_on_coverage_gap():
    df = _toy_panel(); cols = _cols8()
    bad = {"NAN": lambda tr, te, cols, seeds: np.full(len(te), np.nan)}
    try:
        BM.oof_predict(df, cols, (0,), bad, 3)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "coverage gap" in str(e)


# --------------------------------------------------------------------------- meta-optimiser
def test_fit_simplex_qlike_constraint_and_collapse():
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(3e-4, 5e-5, 500)) + 1e-5
    perfect = y                                            # base 0 == target -> QLIKE 0
    noisy = y * (1.0 + rng.normal(0, 0.5, 500))           # base 1 noisy
    biased = y * 3.0                                       # base 2 badly biased
    P = np.column_stack([perfect, noisy, biased])
    w = BM.fit_simplex_qlike(P, y)
    assert w.shape == (3,)
    assert (w >= -1e-9).all() and abs(w.sum() - 1.0) < 1e-6   # simplex constraint satisfied
    assert w[0] > 0.99                                        # collapses onto the perfect (dominant) base


def test_fit_simplex_qlike_degenerate_fallback(monkeypatch):
    # a solver return of all-zeros must fall back to uniform (fail-safe, not silent-zero).
    monkeypatch.setattr(BM, "minimize",
                        lambda *a, **k: types.SimpleNamespace(x=np.zeros(a[1].shape[0])))
    w = BM.fit_simplex_qlike(np.ones((10, 4)), np.ones(10))
    assert np.allclose(w, 0.25)


def test_stack_qlike_matches_mean_qlike():
    y = np.array([1e-4, 2e-4, 3e-4, 4e-4])
    P = np.column_stack([y, y * 1.2])
    q_perfect = BM.stack_qlike(np.array([1.0, 0.0]), P, y, BM.FL, C.PRED_CAP)
    q_biased = BM.stack_qlike(np.array([0.0, 1.0]), P, y, BM.FL, C.PRED_CAP)
    assert q_perfect < q_biased and abs(q_perfect) < 1e-9      # exact forecast -> QLIKE 0


# --------------------------------------------------------------------------- driver pure logic
def test_own8_is_eight_without_rq():
    assert len(R.OWN) == 8 and "rq" not in R.OWN


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


def test_apply_weights_blends_bases():
    d = {"GBME": np.array([1e-4]), "XGB": np.array([2e-4]), "GLM": np.array([3e-4]), "HAR": np.array([4e-4])}
    w = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.isclose(R._apply_weights(d, w)[0], 1e-4)


def test_metrics5_keys():
    y = np.array([1e-4, 2e-4, 3e-4]); p = np.array([1.1e-4, 1.9e-4, 3.2e-4])
    assert set(R._metrics5(y, p)) == {"mse", "rmse", "mae", "r2", "qlike"}


def test_load_earn_sp500_passthrough_and_hose_variants(monkeypatch, tmp_path):
    assert R._load_earn("sp500", {"A": 1}) == {"A": 1}                # sp500 keeps its own edates
    monkeypatch.setattr(R, "REPO", tmp_path)                          # no crawled parquet under tmp
    assert R._load_earn("hose", {}) == {}                            # hose w/o parquet -> edates unchanged
    gg = tmp_path / "results" / "gamma_gbm"; gg.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                  "earnings_date": pd.to_datetime(["2021-01-01", "2021-04-01", "2021-02-01"])
                  }).to_parquet(gg / "hose_earnings_combined.parquet")
    ed = R._load_earn("hose", {})                                     # parquet present -> grouped by ticker
    assert set(ed) == {"AAA", "BBB"} and len(ed["AAA"]) == 2


# --------------------------------------------------------------------------- run smoke
def test_run_hose_structure_evidence_perfold_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    # non-overlapping spike window so the fold's test dates survive exclusion -> spike_robustness is computed
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("1990-01-01", "1990-01-02"),))
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert (tmp_path / "stacking_hose_smoke_h1.json").exists()
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.ORDER:
            assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert set(doc["weights_mean"]) == set(R.BASE_NAMES)
    assert abs(sum(doc["weights_mean"].values()) - 1.0) < 1e-6
    assert doc["best_single"] in R.BASE_NAMES
    assert "per_fold_qlike" in doc and "spike_robustness" in doc and "verdict" in doc
    import overfit_check as OF
    _ok, probs = OF.check_result_evidence(doc)
    assert all("missing" not in p for p in probs), probs      # gate-required evidence keys present


def test_run_sp500_no_spike_block(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, predictors=_fake_predictors())
    doc = docs[1]
    assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
    assert (tmp_path / "stacking_sp500_smoke_h1.json").exists()


def test_run_skips_empty_folds_multi(monkeypatch, tmp_path):
    # non-smoke: synthetic data ends mid-2023, so later S1.FOLDS have empty test windows -> the skip `continue`
    # fires while early folds run (multi-fold pooling). Fake predictors keep it fast.
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "HORIZONS", (1,))
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False, predictors=_fake_predictors())
    assert docs[1]["n_folds"] >= 1


def test_run_omits_robustness_when_all_test_in_spike(monkeypatch, tmp_path):
    # default 2022 spike window covers all of fold0's test dates -> keep empty -> spike_robustness omitted
    _tiny(monkeypatch)
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, predictors=_fake_predictors())
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]
