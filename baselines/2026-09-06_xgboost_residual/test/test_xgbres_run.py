"""Runner helpers (column subsets, flatten/scatter roundtrip, lock keys, metric block, aggregation),
overfit-gate xgb detection, HAR-X parity with the delivered edge_hmatched split, and a real-data smoke."""
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "quality_gate"))

import run_xgboost_residual as R  # noqa: E402
import overfit_check as OF  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def test_feature_column_indices_ladder():
    groups = ["stock", "stock", "market", "market"]
    cols = R.feature_column_indices(groups, n_graph=3)
    assert cols["XGB_resid_base"] == [0, 1]
    assert cols["XGB_resid_market"] == [0, 1, 2, 3]
    assert cols["XGB_resid_graph"] == [0, 1, 2, 3, 4, 5, 6]
    assert cols["XGB_direct"] == [0, 1, 2, 3]


def test_design_and_place_roundtrip():
    t, n, f = 6, 3, 4
    Fall = np.arange(t * n * f, dtype=float).reshape(t, n, f)
    anchors = np.array([1, 3, 5])
    mask = np.array([[True, False, True], [True, True, False], [False, True, True]])
    rows = R.design_rows(Fall, anchors, mask, [0, 2])
    assert rows.shape == (int(mask.sum()), 2)
    back = R.place_rows(rows[:, 0], mask, (3, n))
    assert np.array_equal(back[mask], rows[:, 0])
    assert (back[~mask] == 0).all()


def test_lock_keys_fold():
    y_te = np.array([[0.02, 1e-10], [5e-9, 0.03]])          # (0,1) tiny target; (1,0) below 2*floor
    zrf = np.array([[0.0, 0.0], [0.0, 1.0]])                # (1,1) zero_range_flag
    mask = np.ones((2, 2), bool)
    keys = R.lock_keys_fold(y_te, zrf, mask, ["d0", "d1"], mult=2.0, floor=1e-8)
    assert (0, "d0") not in keys                             # normal cell
    assert (1, "d0") in keys                                 # tiny target
    assert (0, "d1") in keys                                 # below 2*floor
    assert (1, "d1") in keys                                 # zero_range_flag


def test_agg_and_nf_rows():
    agg = R._agg([{"mse": 1.0, "qlike": 2.0, "r2": 0.5, "n": 10},
                  {"mse": 3.0, "qlike": 4.0, "r2": 0.1, "n": 20}])
    assert agg["n"] == 30 and agg["qlike"] == 3.0
    nfloor = np.array([0.1, 0.2, 0.3])
    mask = np.array([[True, False, True]])
    np.testing.assert_array_equal(R._nf_rows(nfloor, mask), np.array([0.1, 0.3]))


def test_metrics_block_reports_lock_and_counts():
    pooled = {(0, "2025-04-10"): (1e-9, 0.5), (1, "2025-05-01"): (0.03, 0.031),
              (0, "2025-05-01"): (0.02, 0.021), (1, "2025-06-01"): (0.04, 0.04)}
    lock = {(0, "2025-04-10")}
    m = R._metrics_block(pooled, lock, 1e-8)
    assert m["n_lock"] == 1 and m["n_ticker_date"] == 4 and m["n_unique_dates"] == 3
    assert m["qlike_nonlock"] < m["qlike"]                   # dropping the lock cell lowers QLIKE
    assert 0.0 <= m["lock_qlike_share"] <= 1.0


def test_reference_models_missing_artifact_returns_empty():
    assert R._reference_models("vn30", 999) == {}          # no edge_hmatched artifact for h999


def test_reference_models_present_has_volga_qlike():
    ref = R._reference_models("vn30", 1)
    if ref:                                                 # artifact present in this repo
        assert "VolGA_qlike" in ref and "dm_VolGA_vs_HARX" in ref


def test_overfit_gate_detects_xgb_as_learned():
    assert OF.looks_learned("XGB_resid_base")
    assert OF.looks_learned("XGB_direct")
    assert not OF.looks_learned("HAR-X")


def test_harx_pooled_parity_with_edge_hmatched():
    """The HAR-X baseline this study scores must equal the delivered edge_hmatched HAR-X (same lb10/
    folds_target=7 split) -- otherwise the XGBoost-vs-HAR-X comparison is not on common ground."""
    import json
    import math
    ref_p = REPO / "results" / "edge_hmatched" / "edgehm_vn30_h1.json"
    if not ref_p.exists():
        pytest.skip("edge_hmatched reference not present")
    ref_q = json.loads(ref_p.read_text())["metrics"]["HAR-X"]["qlike"]
    files = R._glob.glob(R.enriched_glob("vn30"))
    keep = R.frozen_universe(files, R.C.LOOKBACK, 1)
    panel = R.build_enriched_panel(files, R.C.LOOKBACK, 1, keep)
    wf = R.VolgaWFConfig(lookback=R.C.LOOKBACK, horizon=1, folds_target=R.C.FOLDS_TARGET)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = R.make_folds(n, ts, K, wf.val, wf.horizon)
    fl = R.training_config().qlike_floor
    pooled = {}
    for fold in folds:
        D = R.pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = R.pc.POS_FLOOR_FRAC * D.t_mean + R.pc.POS_FLOOR_EPS
        _, harx = R._har_ols_preds(D, fl, nfloor)
        pooled.update(R.RMR._pred_dict(harx["te"], D.y_te, D.tmask_te, D.d_te, D.N))
    q = R.RMR._metrics(pooled, fl)["qlike"]
    assert abs(q - ref_q) < 1e-6, f"HAR-X parity broken: {q} vs {ref_q}"


@pytest.mark.smoke
def test_smoke_run_vn30_h1():
    res = R.run("vn30", 1, smoke=True)
    assert set(res["metrics"]) >= {"HAR", "HAR-X"} | set(R._MODELS)
    for m in R._MODELS:
        mm = res["metrics"][m]
        assert mm["n"] > 0 and np.isfinite(mm["qlike"]) and mm["qlike"] > 0   # positive floored forecasts -> finite QLIKE
        assert res["fit_diagnostics"][m]["status"] in ("ok", "overfit", "underfit", "unknown")
        assert mm["n_ticker_date"] >= mm["n_unique_dates"]                     # never conflate obs with dates
    assert res["selections_per_fold"]                                          # guardrail choice recorded
    assert res["train_metrics"] and res["val_metrics"]                         # overfit evidence present
