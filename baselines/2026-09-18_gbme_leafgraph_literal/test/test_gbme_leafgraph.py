"""Tests for the literal GBME + leaf-cooccurrence-graph baseline.

Focus (the NEW logic): HGBR leaf extraction correctness (reconstruction guard vs ``_raw_predict``), the
vectorised traversal on a hand-built tree, GBME parity with ``full_matrix.gbm``, the graph smoothing helpers,
causal split slicing, ``_pool_doc`` (per-fold / spike / keep-empty), ``_safe_dm``, alpha selection, and a run
smoke that verifies the walk-forward produces a well-formed result doc.
"""
import json

import numpy as np
import pandas as pd
import pytest

import full_matrix as FM
import metrics as M_
import gbme_lg_config as C
import hgbr_leaf as HL
import leaf_graph as LG
import run_gbme_leafgraph as R

FL = FM.FL
GBME, GBMELG = R.GBME, R.GBMELG


# --------------------------------------------------------------------------- fixtures / helpers
def _fit_frame(n=400, seed=0):
    """A small train frame with OWN-8 (+EARN-4) columns and a positive gamma-ish target ``y``."""
    rng = np.random.default_rng(seed)
    cols = R.OWN + FM.EARN
    X = rng.gamma(2.0, 0.01, size=(n, len(cols)))
    df = pd.DataFrame(X, columns=cols)
    df["y"] = np.maximum(0.5 * X[:, 0] + rng.gamma(1.0, 0.005, n), FL)
    df["date"] = pd.Timestamp("2020-01-01")
    df["ticker"] = "AAA"
    return df, cols


def _hand_tree():
    """A hand-built 3-node HGBR-style tree: root splits feature 0 at 0.5, two leaves (value 1.0 / 2.0)."""
    dt = np.dtype([("value", float), ("feature_idx", np.intp), ("num_threshold", float),
                   ("left", np.intp), ("right", np.intp), ("is_leaf", np.uint8)])
    return np.array([(0.0, 0, 0.5, 1, 2, 0), (1.0, 0, 0.0, 0, 0, 1), (2.0, 0, 0.0, 0, 0, 1)], dtype=dt)


def _synth_frames(n_tickers=8):
    """Synthetic per-ticker frames spanning 2021-06..2024-06 (business days) with OWN-8 columns present, so
    ``FM.panel`` yields a real walk-forward panel: the first fold has < MIN_ROWS train rows (skipped) and a
    later fold runs."""
    dates = pd.bdate_range("2021-06-01", "2024-06-01")
    frames = {}
    for i in range(n_tickers):
        rng = np.random.default_rng(100 + i)
        n = len(dates)
        pk = np.maximum(rng.gamma(2.0, 5e-4, n), FL)
        d = pd.DataFrame({"date": dates, "parkinson_variance": pk})
        d["har_daily"] = pk
        d["har_weekly"] = pd.Series(pk).rolling(5, min_periods=1).mean().to_numpy()
        d["har_monthly"] = pd.Series(pk).rolling(22, min_periods=1).mean().to_numpy()
        d["rq"] = np.sqrt(pd.Series(pk ** 2).rolling(5, min_periods=1).mean().to_numpy())  # FM.OWN dropna needs it
        lpk = np.log(pk)
        d["mr_change"] = pd.Series(lpk).diff(1).fillna(0.0).to_numpy()
        d["mr_slope5"] = pd.Series(lpk).diff(5).fillna(0.0).to_numpy() / 5.0
        d["mr_slope10"] = pd.Series(lpk).diff(10).fillna(0.0).to_numpy() / 10.0
        d["mr_dev5"] = (lpk - pd.Series(lpk).rolling(5, min_periods=1).mean().to_numpy())
        d["mr_z22"] = (lpk - pd.Series(lpk).rolling(22, min_periods=1).mean().to_numpy())
        d["ticker"] = f"T{i}"
        d["sector"] = i % 3
        frames[f"T{i}"] = d
    return frames


# --------------------------------------------------------------------------- hgbr_leaf: traversal
def test_traverse_hand_tree():
    nodes = _hand_tree()
    X = np.array([[0.4], [0.6], [0.5]])          # <=0.5 -> left(node1,val1); >0.5 -> right(node2,val2)
    node, val = HL._traverse_tree(nodes, X)
    assert node.tolist() == [1, 2, 1]
    assert val.tolist() == [1.0, 2.0, 1.0]


def test_traverse_all_leaf_root():
    """A single-leaf root: every sample stays at node 0 (the while loop body never runs)."""
    dt = _hand_tree().dtype
    nodes = np.array([(3.0, 0, 0.0, 0, 0, 1)], dtype=dt)
    node, val = HL._traverse_tree(nodes, np.array([[0.1], [9.9]]))
    assert node.tolist() == [0, 0]
    assert val.tolist() == [3.0, 3.0]


# --------------------------------------------------------------------------- hgbr_leaf: fit / predict / recon
def test_fit_gbme_matches_full_matrix():
    """Single-source guard: our fitted GBME reproduces ``full_matrix.gbm`` predictions exactly (same params)."""
    tr, cols = _fit_frame(300, seed=1)
    te, _ = _fit_frame(60, seed=2)
    ours = HL.predict_gbme(HL.fit_gbme(tr, cols, 0), te[cols].to_numpy(float))
    ref = FM.gbm(tr, te, cols, 0)
    assert np.allclose(ours, ref, rtol=0, atol=0)


def test_predict_gbme_floor():
    tr, cols = _fit_frame(200, seed=3)
    m = HL.fit_gbme(tr, cols, 0)
    p = HL.predict_gbme(m, tr[cols].to_numpy(float), floor=0.5)
    assert (p >= 0.5).all()


def test_leaf_matrix_reconstruction_ok():
    """The reconstruction guard passes on a genuine fit and the leaf matrix has one column per fitted tree."""
    tr, cols = _fit_frame(400, seed=4)
    m = HL.fit_gbme(tr, cols, 0)
    X = tr[cols].to_numpy(float)
    lm = HL.leaf_matrix(m, X, check=True)
    assert lm.shape == (len(tr), len(m._predictors))
    assert lm.dtype == np.int32
    # recompute the reconstruction identity vs sklearn's own _raw_predict ground truth (catches a traversal bug)
    vsum = np.zeros(len(tr))
    for stage in m._predictors:
        nd = stage[0].nodes
        node, val = HL._traverse_tree(nd, X)
        vsum += val
    recon = float(m._baseline_prediction) + vsum
    raw = np.asarray(m._raw_predict(X), float).ravel()
    assert np.max(np.abs(recon - raw)) <= C.RECON_TOL


def test_leaf_matrix_reconstruction_raises_on_drift():
    """If the private layout no longer reconstructs ``_raw_predict``, the guard raises (fail-loud, not silent)."""
    tr, cols = _fit_frame(200, seed=5)
    m = HL.fit_gbme(tr, cols, 0)
    X = tr[cols].to_numpy(float)
    m._raw_predict = lambda _X: np.zeros((len(_X), 1))     # simulate a broken/ changed layout
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        HL.leaf_matrix(m, X, check=True)


def test_leaf_matrix_check_false_skips_guard():
    tr, cols = _fit_frame(150, seed=6)
    m = HL.fit_gbme(tr, cols, 0)
    X = tr[cols].to_numpy(float)
    m._raw_predict = lambda _X: np.zeros((len(_X), 1))     # would fail the guard, but check=False skips it
    lm = HL.leaf_matrix(m, X, check=False)
    assert lm.shape[0] == len(tr)


def test_leaf_matrix_empty_rows():
    """Zero-row input: the guard's ``max_err`` short-circuits to 0.0 (no np.max over empty)."""
    tr, cols = _fit_frame(120, seed=7)
    m = HL.fit_gbme(tr, cols, 0)
    lm = HL.leaf_matrix(m, np.empty((0, len(cols))), check=True)
    assert lm.shape == (0, len(m._predictors))


# --------------------------------------------------------------------------- leaf_graph helpers
def test_day_similarity():
    leaves = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])   # rows 0,1 identical; 2 disjoint
    s = LG.day_similarity(leaves)
    assert np.allclose(np.diag(s), 1.0)
    assert s[0, 1] == pytest.approx(1.0)
    assert s[0, 2] == pytest.approx(0.0)


def test_knn_neighbour_mean_and_singleton():
    assert LG.knn_neighbour_mean(np.array([5.0]), np.array([[1.0]]), 3).tolist() == [5.0]
    pred = np.array([1.0, 2.0, 3.0])
    sim = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
    nbr = LG.knn_neighbour_mean(pred, sim, 1)               # each row's single most-similar neighbour
    assert nbr[0] == pytest.approx(2.0) and nbr[1] == pytest.approx(1.0)


def test_smooth_day_alpha_bounds():
    pred = np.array([1.0, 3.0])
    leaves = np.array([[1, 1], [1, 1]])                    # identical leaf-vectors -> perfect neighbours
    assert LG.smooth_day(pred, leaves, 1, 0.0).tolist() == [1.0, 3.0]        # alpha 0 = identity
    assert LG.smooth_day(np.array([7.0]), np.array([[1]]), 1, 0.5).tolist() == [7.0]  # singleton
    out = LG.smooth_day(pred, leaves, 1, 1.0)               # alpha 1 = neighbour mean (swap)
    assert out.tolist() == [3.0, 1.0]


def test_smooth_all_grouping_and_fastpath():
    pred = np.array([1.0, 3.0, 10.0, 30.0])
    leaves = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
    dates = np.array(["d1", "d1", "d2", "d2"])
    assert LG.smooth_all(pred, leaves, dates, 1, 0.0).tolist() == pred.tolist()   # alpha 0 fast path
    out = LG.smooth_all(pred, leaves, dates, 1, 1.0)        # each day smoothed independently
    assert out.tolist() == [3.0, 1.0, 30.0, 10.0]


def test_fit_alpha_picks_grid_argmin():
    """fit_alpha must return the grid value with the strictly-lowest val QLIKE (recomputed independently)."""
    y = np.array([1.0, 1.0])
    pred = np.array([1.0, 3.0])                            # identical leaves -> alpha=0.5 averages to (2,2)
    leaves = np.array([[1, 1], [1, 1]])                    # closer to y than either endpoint on QLIKE
    dates = np.array(["d", "d"])
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    a, q = LG.fit_alpha(y, pred, leaves, dates, 1, grid, FL)
    # independent per-grid QLIKE recompute (does not reuse fit_alpha's argmin)
    qs = [float(np.mean(M_.per_obs_qlike(y, LG.smooth_all(pred, leaves, dates, 1, g), floor=FL))) for g in grid]
    assert a == grid[int(np.argmin(qs))]
    assert q == pytest.approx(min(qs))
    assert a != 0.0                                        # the graph genuinely helps here -> non-trivial alpha


def test_fit_alpha_tie_breaks_to_smaller():
    """On an exact QLIKE tie the ascending grid + strict ``<`` update keeps the SMALLER alpha (graph inert)."""
    y = np.array([1.0, 1.0])
    pred = np.array([1.0, 1.0])                            # already exact -> every alpha gives identical QLIKE
    leaves = np.array([[1, 1], [1, 1]])
    dates = np.array(["d", "d"])
    a, _ = LG.fit_alpha(y, pred, leaves, dates, 1, (0.0, 0.5, 1.0), FL)
    assert a == 0.0


# --------------------------------------------------------------------------- driver helpers
def test_metrics5_keys():
    m = R._metrics5(np.array([1.0, 2.0]), np.array([1.1, 1.9]))
    assert set(m) == {"mse", "rmse", "mae", "r2", "qlike"}


def test_verdict_and_success():
    assert R.verdict(0.5, 0.01) is True
    assert R.verdict(0.5, 0.9) is False                    # gain positive but not significant
    assert R.verdict(-0.1, 0.01) is False                  # significant but wrong sign
    assert R.success({1: {"verdict": {"beats": True}}, 5: {"verdict": {"beats": True}}}) is True
    assert R.success({1: {"verdict": {"beats": True}}}) is False


def test_safe_dm_paths():
    dates = np.array([np.datetime64("2020-01-01"), np.datetime64("2020-01-02")])
    same = np.array([0.1, 0.2])
    assert R._safe_dm(same, same.copy(), dates, 1)["p_value"] == 1.0     # identical -> degenerate
    one_date = np.array([np.datetime64("2020-01-01")] * 2)
    assert R._safe_dm(np.array([0.1, 0.2]), np.array([0.3, 0.4]), one_date, 1)["p_value"] == 1.0  # ValueError
    good = R._safe_dm(np.array([0.5, 0.4, 0.6, 0.5]), np.array([0.1, 0.2, 0.15, 0.2]),
                      np.array([np.datetime64("2020-01-0" + str(i)) for i in (1, 2, 3, 4)]), 1)
    assert 0.0 <= good["p_value"] <= 1.0


def test_spike_mask():
    dates = np.array([np.datetime64("2020-03-15"), np.datetime64("2019-01-01")])
    assert R._spike_mask(dates).tolist() == [True, False]


def test_load_earn(tmp_path, monkeypatch):
    assert R._load_earn("sp500", {"X": np.array([1])}) == {"X": np.array([1])}     # sp500 branch: unchanged
    monkeypatch.setattr(R, "REPO", tmp_path)
    (tmp_path / "results" / "gamma_gbm").mkdir(parents=True)
    assert R._load_earn("hose", {"orig": 1}) == {"orig": 1}                        # hose, no parquet -> orig
    e = pd.DataFrame({"ticker": ["AAA", "AAA"], "earnings_date": pd.to_datetime(["2020-01-01", "2020-06-01"])})
    e.to_parquet(tmp_path / "results" / "gamma_gbm" / "hose_earnings_combined.parquet")
    got = R._load_earn("hose", {"orig": 1})                                        # hose + parquet -> crawled
    assert "AAA" in got and len(got["AAA"]) == 2


def test_fold_predictions_causal_slicing():
    """The combo split (te|tr|va) must map each block back to its SOURCE frame — not merely match lengths.

    Distinct block sizes (te=30, tr=200, va=70) plus a per-row content check (the GBME prediction for each
    split equals predict_gbme on that exact source frame) would catch a te/va offset swap in the ``sl`` closure,
    which a length-only assert cannot."""
    tr, cols = _fit_frame(300, seed=8)
    tr_e = tr.iloc[:200].copy()
    va = tr.iloc[200:270].copy()
    te = tr.iloc[270:].copy()
    assert (len(te), len(tr_e), len(va)) == (30, 200, 70)          # all different -> a swap changes lengths too
    out, alpha = R._fold_predictions(tr_e, va, te, cols, (0,))
    # content alignment: seed-0 base for each split == predict_gbme on that source frame's rows
    model = HL.fit_gbme(tr_e, cols, 0)
    for split, src in (("te", te), ("tr", tr_e), ("va", va)):
        expect = HL.predict_gbme(model, src[cols].to_numpy(float))
        assert np.allclose(out[split][GBME], expect)
    assert 0.0 <= alpha <= 1.0
    assert (out["te"][GBMELG] >= FL).all()


def _mini_doc_inputs(dates_te):
    """Craft one-fold pooled inputs for _pool_doc with controllable test dates."""
    n = len(dates_te)
    rng = np.random.default_rng(0)
    y = np.maximum(rng.gamma(2.0, 5e-4, n), FL)
    pg = y * 1.1
    pl = y * 1.05
    preds = {"te": {GBME: [pg], GBMELG: [pl]},
             "tr": {GBME: [pg], GBMELG: [pl]},
             "va": {GBME: [pg], GBMELG: [pl]}}
    yy = {"te": [y], "tr": [y], "va": [y]}
    return preds, yy, [np.array(dates_te)]


def test_pool_doc_perfold_and_spike():
    dates = [np.datetime64("2023-02-0" + str(i)) for i in (1, 2, 3, 4)]        # outside spike windows
    preds, yy, dts = _mini_doc_inputs(dates)
    doc = R._pool_doc(1, "hose", (0,), preds, yy, dts, [0.3], per_fold=True, spike=True)
    assert doc["n_folds"] == 1
    assert "per_fold_qlike" in doc and "spike_robustness" in doc
    assert doc["spike_robustness"]["n_ex_spike_obs"] == 4
    assert set(doc["metrics"]) == {GBME, GBMELG}


def test_pool_doc_spike_all_inside_window():
    dates = [np.datetime64("2020-03-1" + str(i)) for i in (1, 2, 3, 4)]        # all inside COVID window
    preds, yy, dts = _mini_doc_inputs(dates)
    doc = R._pool_doc(1, "hose", (0,), preds, yy, dts, [0.3], per_fold=False, spike=True)
    assert "spike_robustness" not in doc                                       # keep.any() False -> skipped
    assert "per_fold_qlike" not in doc


def test_pool_doc_no_spike_no_perfold():
    """SP500-like: per_fold and spike both off -> neither optional block is added."""
    dates = [np.datetime64("2023-02-0" + str(i)) for i in (1, 2, 3, 4)]
    preds, yy, dts = _mini_doc_inputs(dates)
    doc = R._pool_doc(5, "sp500", (0,), preds, yy, dts, [0.0], per_fold=False, spike=False)
    assert "spike_robustness" not in doc and "per_fold_qlike" not in doc
    assert doc["h"] == 5


# --------------------------------------------------------------------------- run smoke (integration)
@pytest.mark.smoke
def test_run_smoke(tmp_path):
    frames = _synth_frames(8)

    def load_fn(_market):
        return frames, {}, {}

    docs = R.run("hose", load_fn=load_fn, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert doc["n_folds"] == 1                                  # smoke caps to one eligible fold
    assert set(doc["metrics"]) == {GBME, GBMELG}
    assert "beats" in doc["verdict"]
    saved = json.loads((tmp_path / "gbme_lg_hose_smoke_h1.json").read_text())
    assert saved["h"] == 1 and 0.0 <= saved["alpha"]["mean"] <= 1.0
