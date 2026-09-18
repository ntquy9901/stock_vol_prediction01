"""Tests for the leaf-graph v2 (RF-GAP / KeRF) baseline: leaf-Hamming (knn), the RF-GAP proper-proximity
normalisation, KeRF large-leaf down-weighting, vectorised-vs-reference consistency, scheme-aware smoothing +
alpha fit, causality, streaming-metric equivalence, DM/verdict/spike/v2-vs-v1 logic, and a walk-forward run
smoke carrying the gate-required over/under-fit evidence keys. Synthetic panels keep the driver fast."""
import numpy as np
import pandas as pd
import pytest

import rfgap_config as C
import rfgap as RG
import run_rfgap as R
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


def _frames(n_tickers=4):
    return {f"TK{i}": _ticker_frame(seed=i).assign(ticker=f"TK{i}", sector=0) for i in range(n_tickers)}


def _fake_loader(market):
    return _frames(), {f"TK{i}": 0 for i in range(4)}, {}


def _tiny(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    y = np.exp(-8.0 + 0.5 * x[:, 0])                       # positive target, log-linear in x0
    df = pd.DataFrame({"f0": x[:, 0], "f1": x[:, 1], "y": y})
    return df.iloc[:300], df.iloc[300:]


# --------------------------------------------------------------------------- leaf-Hamming (knn scheme)
def test_day_similarity_identical_disjoint_partial():
    ident = np.array([[3, 7, 1, 4], [3, 7, 1, 4]])
    assert np.allclose(RG.day_similarity(ident), 1.0)
    disjoint = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    s2 = RG.day_similarity(disjoint)
    assert np.isclose(s2[0, 1], 0.0) and np.allclose(np.diag(s2), 1.0)
    partial = np.array([[1, 2, 3, 4], [1, 2, 9, 8]])       # 2/4 trees match
    assert np.isclose(RG.day_similarity(partial)[0, 1], 0.5)


def test_knn_neighbour_mean_top_k_excludes_self_and_singleton():
    pred = np.array([10.0, 20.0, 30.0])
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    assert np.isclose(RG.knn_neighbour_mean(pred, sim, k=1)[0], 20.0)
    assert np.isclose(RG.knn_neighbour_mean(pred, sim, k=2)[0], 25.0)
    assert np.allclose(RG.knn_neighbour_mean(np.array([5.0]), np.array([[1.0]]), k=3), [5.0])


# --------------------------------------------------------------------------- RF-GAP proper proximity
def test_rfgap_weights_are_row_stochastic_and_self_zero():
    # 4 stocks, 3 trees; every stock shares a leaf with >=1 other stock in every tree -> proper proximity
    leaves = np.array([[1, 5, 9], [1, 6, 9], [2, 5, 8], [2, 6, 8]])
    W = RG.day_weights(leaves, "rfgap")
    assert np.allclose(np.diag(W), 0.0)                     # a stock never smooths itself
    assert np.allclose(W.sum(1), 1.0)                       # RF-GAP normalisation: neighbour weights sum to 1
    assert (W >= 0).all()


def test_rfgap_isolated_stock_row_is_zero():
    # stock 2 lands in a unique leaf in every tree -> no co-member -> all-zero row (caller keeps its base pred)
    leaves = np.array([[1, 1, 1], [1, 1, 1], [7, 8, 9]])
    W = RG.day_weights(leaves, "rfgap")
    assert np.allclose(W[2], 0.0)
    assert np.isclose(W[0, 1], 1.0)                         # stocks 0,1 only ever co-occur with each other


def test_rfgap_leave_one_out_mean_matches_hand_computation():
    # single tree, one leaf shared by 3 stocks: LOO mean of stock0 = mean(y1, y2)
    leaves = np.array([[4], [4], [4]])
    pred = np.array([10.0, 20.0, 30.0])
    nm = RG.neighbour_mean(pred, leaves, "rfgap")
    assert np.allclose(nm, [25.0, 20.0, 15.0])             # (20+30)/2, (10+30)/2, (10+20)/2


# --------------------------------------------------------------------------- KeRF large-leaf down-weight
def test_kerf_downweights_neighbours_reached_through_large_leaves():
    # A shares a SMALL leaf (size 2) with B in tree0, and a LARGE leaf (size 4) with C,D,E in tree1.
    leaves = np.array([[0, 9],    # A
                       [0, 1],    # B  (small-leaf co-member of A in tree0)
                       [2, 9],    # C  (large-leaf co-member of A in tree1)
                       [3, 9],    # D
                       [4, 9]])   # E
    Wr = RG.day_weights(leaves, "rfgap")
    Wk = RG.day_weights(leaves, "rfgap_kerf", kerf_func="inv")
    # small-leaf neighbour B gains weight, large-leaf neighbour C loses weight under KeRF
    assert Wk[0, 1] > Wr[0, 1]
    assert Wk[0, 2] < Wr[0, 2]
    assert np.isclose(Wr.sum(1)[0], 1.0) and np.isclose(Wk.sum(1)[0], 1.0)   # both stay row-stochastic
    # exact hand values: rfgap B=1/2,C=1/6 ; kerf(inv) B=2/3,C=1/9
    assert np.isclose(Wr[0, 1], 0.5) and np.isclose(Wr[0, 2], 1 / 6)
    assert np.isclose(Wk[0, 1], 2 / 3) and np.isclose(Wk[0, 2], 1 / 9)


def test_kerf_invsqrt_weight_and_bad_func():
    assert np.isclose(RG._kerf_weight(4.0, "invsqrt"), 0.5)
    assert np.isclose(RG._kerf_weight(4.0, "inv"), 0.25)
    with pytest.raises(ValueError):
        RG._kerf_weight(4.0, "nope")


# --------------------------------------------------------------------------- vectorised == reference matrix
@pytest.mark.parametrize("scheme,kf", [("rfgap", None), ("rfgap_kerf", "inv"), ("rfgap_kerf", "invsqrt")])
def test_vectorised_neighbour_mean_matches_day_weights(scheme, kf):
    rng = np.random.default_rng(7)
    m, t = 12, 9
    leaves = rng.integers(0, 4, (m, t))                    # small leaf alphabet -> many co-occurrences
    pred = rng.uniform(1e-4, 5e-4, m)
    W = RG.day_weights(leaves, scheme, kerf_func=kf)
    ref = np.where(W.sum(1, keepdims=True) > 0, W @ pred, pred)   # all-zero row -> own pred
    got = RG.neighbour_mean(pred, leaves, scheme, kerf_func=kf)
    assert np.allclose(got, ref, atol=1e-12)


def test_knn_day_weights_and_singleton_neighbour_mean():
    # knn day_weights builds a hard top-k row-stochastic matrix; k=1 -> each row a single 1.0 neighbour
    leaves = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]])
    W = RG.day_weights(leaves, "knn", k=1)
    assert np.allclose(W.sum(1), 1.0) and np.allclose(np.diag(W), 0.0)
    assert np.isclose(W[0, 1], 1.0)                        # stock0's top-1 is its identical-leaf twin stock1
    assert np.allclose(RG.neighbour_mean(np.array([9e-4]), np.array([[1, 2]]), "knn"), [9e-4])   # singleton


def test_rfgap_tree_with_no_co_members_is_skipped():
    # tree0 gives everyone a co-member (shared leaves), tree1 puts every stock in a UNIQUE leaf -> skipped
    leaves = np.array([[1, 7], [1, 8], [1, 9]])
    W = RG.day_weights(leaves, "rfgap")
    # only tree0 contributes: 3 stocks share one leaf -> uniform LOO weights 1/2 to each of the other two
    assert np.allclose(W, np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]))


def test_neighbour_mean_and_weights_reject_unknown_scheme():
    with pytest.raises(ValueError):
        RG.neighbour_mean(np.array([1.0, 2.0]), np.array([[1], [1]]), "bogus")
    with pytest.raises(ValueError):
        RG.day_weights(np.array([[1], [1]]), "bogus")


def test_singleton_cross_section_returns_base():
    assert np.allclose(RG.neighbour_mean(np.array([3e-4]), np.array([[1, 2, 3]]), "rfgap"), [3e-4])
    assert np.allclose(RG.day_weights(np.array([[1, 2]]), "rfgap"), np.zeros((1, 1)))


# --------------------------------------------------------------------------- smoothing + alpha fit
def _one_day_leaves():
    return np.array([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8]])   # stocks 0,1 identical; stock2 disjoint


@pytest.mark.parametrize("scheme", ["knn", "rfgap", "rfgap_kerf"])
def test_smooth_day_alpha0_identity_and_alpha1_neighbour_mean(scheme):
    pred = np.array([10.0, 40.0, 100.0])
    leaves = _one_day_leaves()
    assert np.allclose(RG.smooth_day(pred, leaves, scheme, C.K_NEIGHBOURS, 0.0, C.KERF_FUNC), pred)
    nm1 = RG.smooth_day(pred, leaves, scheme, 1, 1.0, C.KERF_FUNC)
    assert np.isclose(nm1[0], 40.0) and np.isclose(nm1[1], 10.0)   # 0<->1 swap (identical leaves)
    if scheme != "knn":
        assert np.isclose(nm1[2], 100.0)     # rfgap: stock2 has no leaf co-member -> isolated -> base unchanged
    else:
        assert np.isclose(nm1[2], 10.0)      # knn: hard top-k always assigns a neighbour even with 0 similarity


def test_smooth_day_singleton_and_monotone_in_alpha():
    assert np.allclose(RG.smooth_day(np.array([7.0]), np.array([[1, 2, 3]]), "rfgap", 10, 0.5), [7.0])
    pred = np.array([10.0, 40.0, 100.0])
    leaves = _one_day_leaves()
    lo = RG.smooth_day(pred, leaves, "rfgap", 1, 0.2)[0]
    hi = RG.smooth_day(pred, leaves, "rfgap", 1, 0.8)[0]
    assert 10.0 < lo < hi < 40.0


def test_smooth_all_groups_by_date_fastpath_and_no_cross_day_mix():
    pred = np.array([10.0, 40.0, 100.0, 200.0])
    leaves = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]])
    dates = np.array(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"])
    assert np.allclose(RG.smooth_all(pred, leaves, dates, "rfgap", 10, 0.0), pred)     # identity fast path
    sm = RG.smooth_all(pred, leaves, dates, "rfgap", 1, 1.0)
    assert np.allclose(sm, [40.0, 10.0, 200.0, 100.0])                                 # per-day swaps only


def test_smoothing_is_per_day_causal():
    pred = np.array([10.0, 40.0, 100.0, 200.0])
    leaves = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
    dates = np.array(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"])
    base = RG.smooth_all(pred, leaves, dates, "rfgap", 1, 0.5)
    pred2 = pred.copy(); pred2[2:] = [999.0, 888.0]
    other = RG.smooth_all(pred2, leaves, dates, "rfgap", 1, 0.5)
    assert np.allclose(base[:2], other[:2])                        # day-1 output invariant to day-2 changes


def test_fit_alpha_prefers_zero_when_smoothing_hurts_and_positive_when_helps():
    # persistent per-stock targets: base is exact -> any smoothing raises QLIKE -> alpha 0
    pred = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4])
    dates = np.array(["2021-01-01"] * 6)
    leaves = np.tile(np.arange(6)[:, None], (1, 8)).astype(int)
    a0, q0 = RG.fit_alpha(pred.copy(), pred, leaves, dates, "rfgap", C.K_NEIGHBOURS, C.ALPHA_GRID, C.KERF_FUNC)
    assert a0 == 0.0 and a0 in C.ALPHA_GRID and q0 >= 0.0
    # noisy base around a shared truth: neighbour mean is a better forecast -> positive alpha
    truth = np.array([2e-4, 2e-4, 2e-4, 2e-4])
    noisy = np.array([1e-4, 3e-4, 1e-4, 3e-4])
    lv = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    a1, _ = RG.fit_alpha(truth, noisy, lv, np.array(["d"] * 4), "rfgap", C.K_NEIGHBOURS, C.ALPHA_GRID, C.KERF_FUNC)
    assert a1 > 0.0


# --------------------------------------------------------------------------- booster plumbing + guard
def test_fit_predict_booster_positive_and_leaf_matrix_shape():
    tr, te = _toy()
    bst = RG.fit_booster(tr, ["f0", "f1"], seed=0)
    p = RG.predict_booster(bst, te[["f0", "f1"]].to_numpy(float))
    assert (p > 0).all() and np.isfinite(p).all() and (p <= C.PRED_CAP + 1e-12).all()
    L = RG.leaf_matrix(bst, te[["f0", "f1"]].to_numpy(float))
    assert L.shape == (len(te), C.XGB_N_ESTIMATORS) and L.dtype == np.int32


def test_predict_booster_clips_to_cap_and_seed_ensemble(monkeypatch):
    tr, te = _toy()
    bst = RG.fit_booster(tr, ["f0", "f1"], seed=0)
    monkeypatch.setattr(C, "PRED_CAP", 1e-4)
    p = RG.predict_booster(bst, te[["f0", "f1"]].to_numpy(float))
    assert (p <= 1e-4 + 1e-18).all() and (p >= RG.FL).all()
    monkeypatch.undo()
    combo = pd.concat([te, tr])
    pe = RG.predict_xgb(tr, combo, ["f0", "f1"], (0, 1))
    assert pe.shape == (len(combo),) and (pe > 0).all() and np.isfinite(pe).all()


# --------------------------------------------------------------------------- streaming metrics
def test_stream_matches_direct_pooled_metrics():
    rng = np.random.default_rng(1)
    y = rng.uniform(1e-4, 9e-4, 500)
    p = np.clip(y + rng.normal(0, 5e-5, 500), 1e-6, None)
    st = R._Stream()
    for i in range(0, 500, 100):                                   # stream in 5 folds
        st.update(y[i:i + 100], p[i:i + 100])
    direct = R._metrics5(y, p)
    got = st.finalize()
    for kk in ("mse", "rmse", "mae", "r2", "qlike"):
        assert np.isclose(got[kk], direct[kk], rtol=1e-9, atol=1e-12), kk


def test_stream_and_metrics5_constant_y_r2_zero():
    y = np.full(10, 3e-4); p = y + 1e-6
    assert R._metrics5(y, p)["r2"] == 0.0
    st = R._Stream(); st.update(y, p)
    assert st.finalize()["r2"] == 0.0


# --------------------------------------------------------------------------- pure driver logic
def test_verdict_and_success():
    assert R.verdict(0.5, 0.01) and not R.verdict(-0.1, 0.01) and not R.verdict(0.5, 0.20)
    beat = {1: {"v2_vs_v1": {"XGB+rfgap": {"beats_knn": True}, "XGB+rfgap+kerf": {"beats_knn": False}}}}
    assert R.success(beat)
    robust = {1: {"v2_vs_v1": {"XGB+rfgap": {"beats_knn": False, "as_good_more_robust": True},
                               "XGB+rfgap+kerf": {"beats_knn": False}}}}
    assert R.success(robust)
    assert not R.success({1: {"v2_vs_v1": {"XGB+rfgap": {"beats_knn": False},
                                           "XGB+rfgap+kerf": {"beats_knn": False}}}})
    assert not R.success({})


def test_safe_dm_degenerate_and_valueerror(monkeypatch):
    e = np.array([1.0, 2.0, 3.0, 4.0]); d = pd.to_datetime(["2021-01-01"] * 4).to_numpy()
    assert R._safe_dm(e, e.copy(), d, 1)["p_value"] == 1.0
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda *a, **k: (_ for _ in ()).throw(ValueError("hln")))
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 1.0


def test_dm_block_and_spike_for():
    err = {R.GBME: np.array([2.0, 2.0]), R.XGB: np.array([2.0, 2.0]),
           R.KNN: np.array([1.0, 1.0]), R.RFGAP: np.array([0.5, 0.5]), R.KERF: np.array([0.5, 0.5])}
    dates = pd.to_datetime(["2019-06-01", "2019-06-02"]).to_numpy()
    dm, gain = R._dm_block(err, dates, 1, R.XGB)
    assert set(gain) == {R.KNN, R.RFGAP, R.KERF} and gain[R.RFGAP] > gain[R.KNN] > 0
    blk, per = R._spike_for(err, dates, 1, R.XGB, R.SMOOTHED)
    assert blk["n_spike_obs"] == 0 and set(per) == {R.KNN, R.RFGAP, R.KERF}
    # all-in-spike -> empty per-variant block
    sp_dates = pd.to_datetime(["2022-06-01", "2022-06-02"]).to_numpy()
    blk2, per2 = R._spike_for(err, sp_dates, 1, R.XGB, R.SMOOTHED)
    assert per2 == {} and blk2["n_ex_spike_obs"] == 0


def test_spike_mask_flags_windows():
    d = pd.to_datetime(["2019-06-01", "2020-03-15", "2022-06-01", "2025-04-10"]).to_numpy()
    assert list(R._spike_mask(d)) == [False, True, True, True]


def test_load_earn_paths(monkeypatch, tmp_path):
    assert R._load_earn("sp500", {"A": 1}) == {"A": 1}
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert R._load_earn("hose", {}) == {}


def test_load_earn_hose_reads_real_parquet():
    ed = R._load_earn("hose", {})
    assert isinstance(ed, dict) and len(ed) > 0


def test_metrics5_keys():
    y = np.array([1e-4, 2e-4, 3e-4]); p = np.array([1.1e-4, 1.9e-4, 3.2e-4])
    assert set(R._metrics5(y, p)) == {"mse", "rmse", "mae", "r2", "qlike"}


# --------------------------------------------------------------------------- run smoke
def test_run_hose_structure_evidence_perfold_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("1990-01-01", "1990-01-02"),))   # non-overlapping -> spike computed
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert (tmp_path / "rfgap_hose_smoke_h1.json").exists()
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.ORDER:
            assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert set(doc["dm"]) == {"vs_XGB", "vs_GBME", "vs_knn"}
    assert set(doc["alpha"]) == set(R.SMOOTHED)
    assert "per_fold_qlike" in doc and "spike_robustness" in doc and "v2_vs_v1" in doc
    for m in C.V2_VARIANTS:
        assert {"gain_pct_vs_knn", "dm_p_vs_knn", "beats_knn", "as_good_more_robust"} <= set(doc["v2_vs_v1"][m])
    import overfit_check as OF
    _ok, probs = OF.check_result_evidence(doc)
    assert all("missing" not in p for p in probs), probs


def test_run_sp500_no_spike_block(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    doc = docs[1]
    assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
    assert doc["v2_vs_v1"] and "more_spike_robust" not in doc["v2_vs_v1"][C.V2_VARIANTS[0]]
    assert (tmp_path / "rfgap_sp500_smoke_h1.json").exists()


def test_run_skips_empty_folds_multi(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False, horizons=(1,))
    assert docs[1]["n_folds"] >= 1


def test_run_omits_robustness_when_all_test_in_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)   # default 2022 window covers fold0
    assert "spike_robustness" in docs[1] and "qlike_ex_spike" not in docs[1]["spike_robustness"]
    assert "per_fold_qlike" in docs[1]
