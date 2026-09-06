"""Unit tests for the horizon-matched vol->PK edge builder (directed_vol2pk_hmatched)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))

import run_edge_hmatched as EH  # noqa: E402
import masked_rich as MR  # noqa: E402


def test_h1_zerofloor_reproduces_delivered_edge():
    """At h=1 with the significance floor off (z=0), the horizon-matched builder must reproduce the
    delivered fixed lag-1 edge exactly (same lead-lag, same Top-K)."""
    rng = np.random.default_rng(0)
    n, T = 5, 120
    vshock = rng.standard_normal((T, n))
    sqrt_pk = np.abs(rng.standard_normal((T, n))) + 0.1
    last_row = T - 1
    A_new = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, last_row, horizon=1, top_k=MR.EDGE_TOP_K, alpha=None)
    A_ref = MR._directed_vol2pk(vshock, sqrt_pk, last_row, MR.EDGE_TOP_K)
    assert np.allclose(A_new, A_ref, atol=1e-6)


def test_detects_lead_lag_at_matching_horizon_only():
    """Plant: source i=1 volume(t) == target j=0 sqrt_pk(t+2) (a lag-2 lead). The h=2 edge must find
    a strong 0<-1 link; the h=1 edge (wrong lag) must not."""
    rng = np.random.default_rng(1)
    n, T = 3, 200
    r = rng.standard_normal(T)                       # iid -> ~0 autocorr at lag 1
    sqrt_pk = np.abs(rng.standard_normal((T, n))) + 0.1
    sqrt_pk[:, 0] = np.abs(r) + 0.1                  # target j=0 driven by r
    vshock = rng.standard_normal((T, n))
    vshock[:-2, 1] = np.abs(r[2:])                   # source i=1 at t == target 0 at t+2
    last_row = T - 1
    A2 = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, last_row, horizon=2, top_k=1, alpha=None)
    A1 = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, last_row, horizon=1, top_k=1, alpha=None)
    assert abs(A2[0, 1]) > 0.8              # strong at the matching horizon
    assert abs(A1[0, 1]) < 0.3             # weak at the wrong horizon


def test_significance_floor_prunes_noise_vs_nofloor():
    """On pure noise the Bonferroni floor keeps FAR fewer off-diagonal edges than the unfloored Top-K
    (which saturates), i.e. the graph collapses toward the no-graph fallback where there is no signal."""
    rng = np.random.default_rng(2)
    n, T = 8, 400
    vshock = rng.standard_normal((T, n))
    sqrt_pk = np.abs(rng.standard_normal((T, n))) + 0.1
    A_floor = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, T - 1, horizon=5, top_k=5, alpha=0.05)
    A_none = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, T - 1, horizon=5, top_k=5, alpha=None)
    d_floor, d_none = EH._edge_density(A_floor), EH._edge_density(A_none)
    assert d_none > 0.5                    # unfloored Top-K saturates on n=8 (5/7 per row)
    assert d_floor < 0.5 * d_none          # floor prunes most spurious edges
    assert np.allclose(np.diag(A_floor), 1.0)


def test_selfloop_always_one():
    rng = np.random.default_rng(3)
    vshock = rng.standard_normal((40, 4))
    sqrt_pk = np.abs(rng.standard_normal((40, 4))) + 0.1
    A = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, 39, horizon=3, top_k=2)
    assert np.allclose(np.diag(A), 1.0)


def test_select_models_parses_and_orders():
    assert EH._select_models("VolGA") == ("VolGA",)
    assert EH._select_models("VolGA,LSTM") == ("LSTM", "VolGA")          # canonical order
    assert EH._select_models("LSTM, VolGA ") == ("LSTM", "VolGA")


def test_select_models_rejects_bad_and_empty():
    import pytest
    with pytest.raises(SystemExit):
        EH._select_models("VolGA_hm")      # retired name -> not selectable
    with pytest.raises(SystemExit):
        EH._select_models("XGBoost")
    with pytest.raises(SystemExit):
        EH._select_models(" , ")


def test_dm_plan_paper_set_and_subset():
    # paper set: LSTM (no-graph) + VolGA (graph, fixed edge) -> graph-value + baseline contrasts
    full = EH._dm_plan(("LSTM", "VolGA"))
    assert [p[0] for p in full] == ["VolGA_vs_LSTM", "VolGA_vs_HAR-X"]
    # VolGA only -> just the HAR-X baseline contrast
    assert EH._dm_plan(("VolGA",)) == [("VolGA_vs_HAR-X", "VolGA", "HAR-X")]
    # LSTM only -> LSTM vs HAR-X
    assert EH._dm_plan(("LSTM",)) == [("LSTM_vs_HAR-X", "LSTM", "HAR-X")]


def test_provenance_records_run_config():
    p = EH._provenance(lookback=10, batch=32, epochs=16, qlike_floor=1e-8, edge_top_k=5, limit_lock_mult=2.0)
    assert p == {"lookback": 10, "batch": 32, "epochs": 16, "qlike_floor": 1e-8,
                 "edge_top_k": 5, "limit_lock_mult": 2.0}
    assert EH._provenance(22, None, 16, 1e-8, 5, 2.0)["batch"] is None   # batch=None (training_config default) preserved


def test_cell_rows_flattens_pred_dict():
    pred = {(0, "2026-01-01"): (1.0, 1.1), (2, "2026-01-02"): (3.0, 2.9)}
    rows = EH._cell_rows(pred, "VolGA", "test", 3, ["AAA", "BBB", "CCC"])
    assert len(rows) == 2
    r = {(x["ticker"], x["date"]): x for x in rows}
    assert r[("AAA", "2026-01-01")] == {"model": "VolGA", "split": "test", "fold": 3,
                                        "ticker": "AAA", "date": "2026-01-01", "y_true": 1.0, "y_pred": 1.1}
    assert r[("CCC", "2026-01-02")]["y_pred"] == 2.9   # node idx 2 -> tickers[2]
    assert EH._cell_rows({}, "HAR", "train", 0, []) == []   # empty split -> no rows


def test_split_dates_reads_target_dates():
    from types import SimpleNamespace
    panel = SimpleNamespace(target_dates=np.array(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                                                  dtype="datetime64[D]"))
    assert EH._split_dates(panel, slice(0, 2)) == ["2025-01-01", "2025-01-02"]
    assert EH._split_dates(panel, slice(2, 4)) == ["2025-01-03", "2025-01-04"]


def test_fold_cell_rows_all_splits_no_dtr_attr():
    # Regression for the --dump-cells bug: MaskedRichData exposes NO d_tr/d_va/d_te; dates must come from
    # panel.target_dates. This mock D deliberately lacks d_* to catch any getattr(D,'d_tr') regression.
    from types import SimpleNamespace
    ones = np.ones((2, 2), dtype=bool)
    D = SimpleNamespace(N=2,
                        y_tr=np.array([[1.0, 2.0], [3.0, 4.0]]), tmask_tr=ones,
                        y_va=np.array([[5.0, 6.0], [7.0, 8.0]]), tmask_va=ones,
                        y_te=np.array([[9.0, 10.0], [11.0, 12.0]]), tmask_te=ones)
    panel = SimpleNamespace(
        tickers=["AAA", "BBB"],
        target_dates=np.array(["2025-01-01", "2025-01-02", "2025-01-03",
                               "2025-01-04", "2025-01-05", "2025-01-06"], dtype="datetime64[D]"))
    fold = SimpleNamespace(train=slice(0, 2), val=slice(2, 4), forecast=slice(4, 6))
    har = {"tr": np.full((2, 2), 0.1), "va": np.full((2, 2), 0.2), "te": np.full((2, 2), 0.3)}
    harx = {"tr": np.full((2, 2), 0.4), "va": np.full((2, 2), 0.5), "te": np.full((2, 2), 0.6)}
    # deep model "VolGA": two seeds, ensembled -> mean; keys train/val/test
    fold_out = {"VolGA": [{"train": np.full((2, 2), 1.0), "val": np.full((2, 2), 2.0), "test": np.full((2, 2), 3.0)},
                          {"train": np.full((2, 2), 3.0), "val": np.full((2, 2), 4.0), "test": np.full((2, 2), 5.0)}]}
    rows = EH._fold_cell_rows(D, panel, fold, fold_out, har, harx, ("VolGA",), fi=0)
    assert len(rows) == 3 * 3 * 4          # 3 models x 3 splits x (2 anchors x 2 nodes)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert {r["model"] for r in rows} == {"HAR", "HAR-X", "VolGA"}
    # dates come from panel.target_dates at each split's slice
    assert {r["date"] for r in rows if r["split"] == "train"} == {"2025-01-01", "2025-01-02"}
    assert {r["date"] for r in rows if r["split"] == "test"} == {"2025-01-05", "2025-01-06"}
    # VolGA test pred = mean(3.0, 5.0) = 4.0; HAR-X train = 0.4
    v = next(r for r in rows if r["model"] == "VolGA" and r["split"] == "test" and r["ticker"] == "AAA")
    assert v["y_pred"] == 4.0 and v["y_true"] == 9.0
    hx = next(r for r in rows if r["model"] == "HAR-X" and r["split"] == "train" and r["ticker"] == "BBB")
    assert hx["y_pred"] == 0.4


def test_progress_line_format():
    assert EH._progress("fold 1/7 start", 2.5) == "[edgehm] fold 1/7 start (2.5 min)"
    assert EH._progress("x", 0.04) == "[edgehm] x (0.0 min)"


def test_filter_limitlock_drops_floored_targets():
    floor = 1e-8
    pooled = {("A", "d1"): (1e-3, 1e-3),   # normal -> keep
              ("A", "d2"): (1e-8, 5e-4),   # target at floor (limit-lock) -> drop
              ("B", "d1"): (2e-8, 1e-3),   # target == 2*floor (not > threshold) -> drop
              ("B", "d2"): (3e-8, 1e-3)}   # target > 2*floor -> keep
    kept, dropped = EH._filter_limitlock(pooled, floor, 2.0)
    assert dropped == 2
    assert set(kept) == {("A", "d1"), ("B", "d2")}


def test_agg_split_metrics_means_and_total_n():
    d1 = {"mse": 1.0, "qlike": 0.4, "r2": 0.5, "n": 10}
    d2 = {"mse": 3.0, "qlike": 0.6, "r2": 0.1, "n": 30}
    agg = EH._agg_split_metrics([d1, d2])
    assert agg["mse"] == 2.0 and agg["qlike"] == 0.5 and abs(agg["r2"] - 0.3) < 1e-9
    assert agg["n"] == 40                       # n is summed, metrics are averaged
    # classify_fit consumes qlike+r2 -> the agg dict is a valid fit-verdict input
    assert set(("mse", "qlike", "r2", "n")) <= set(agg)


def test_edge_density_range():
    A = np.eye(5, dtype=np.float32); A[0, 1] = 0.5; A[2, 3] = -0.4
    assert 0.0 <= EH._edge_density(A) <= 1.0
    assert EH._edge_density(np.eye(5, dtype=np.float32)) == 0.0


def test_min_pairs_skips_all_sources():
    """When no source has >= min_pairs finite overlaps, every candidate is skipped and only the self-loop
    survives (identity graph, zero off-diagonal density)."""
    rng = np.random.default_rng(5)
    vshock = rng.standard_normal((60, 4))
    sqrt_pk = np.abs(rng.standard_normal((60, 4))) + 0.1
    A = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, 59, horizon=1, top_k=2, alpha=None, min_pairs=10_000)
    assert np.allclose(np.diag(A), 1.0)
    assert EH._edge_density(A) == 0.0


def test_constant_source_contributes_no_edge():
    """A source column with zero variance (std==0) is skipped, so no target receives an edge from it."""
    rng = np.random.default_rng(6)
    n, T = 4, 200
    vshock = rng.standard_normal((T, n)); vshock[:, 2] = 5.0   # constant source i=2
    sqrt_pk = np.abs(rng.standard_normal((T, n))) + 0.1
    A = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, T - 1, horizon=1, top_k=3, alpha=None)
    for j in range(n):
        if j != 2:
            assert A[j, 2] == 0.0                              # no incoming edge from the constant source


def test_pool_delegates_to_pred_dict(monkeypatch):
    """_pool is a thin adapter forwarding the model output + fold tensors to RMR._pred_dict."""
    import types
    seen = {}

    def fake_pred_dict(o, y, tm, d, N):
        seen["args"] = (o, y, tm, d, N)
        return {"pooled": True}

    monkeypatch.setattr(EH.RMR, "_pred_dict", fake_pred_dict)
    D = types.SimpleNamespace(y_te="Y", tmask_te="TM", d_te="D", N=9)
    out = EH._pool("OBJ", D)
    assert out == {"pooled": True}
    assert seen["args"] == ("OBJ", "Y", "TM", "D", 9)


def _bruteforce_hmatched(vshock, sqrt_pk, last_row, horizon, top_k, alpha, min_pairs):
    """Independent O(N^2) reference (per-pair np.corrcoef with NaN masking) for the vectorised builder."""
    import math
    from statistics import NormalDist
    v = vshock[:last_row + 1]; p = sqrt_pk[:last_row + 1]
    src = v[:-horizon]; tgt = p[horizon:]
    n = v.shape[1]
    A = np.zeros((n, n), dtype=np.float32)
    z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(n - 1, 1))) if alpha else 0.0
    for j in range(n):
        fj = tgt[:, j]; corrs = np.full(n, np.nan); thr = np.full(n, np.inf)
        for i in range(n):
            if i == j:
                continue
            m = np.isfinite(src[:, i]) & np.isfinite(fj); mm = int(m.sum())
            if mm < min_pairs:
                continue
            a, b = src[:, i][m], fj[m]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            corrs[i] = float(np.corrcoef(a, b)[0, 1])
            thr[i] = (z / math.sqrt(mm)) if alpha else 0.0
        sig = np.isfinite(corrs) & (np.abs(corrs) > thr); valid = np.flatnonzero(sig)
        if valid.size:
            k = valid[np.argsort(-np.abs(corrs[valid]))[:top_k]]
            A[j, k] = corrs[k]
    np.fill_diagonal(A, 1.0)
    return A


def test_vectorised_matches_bruteforce_with_nans():
    """The vectorised builder must equal an independent per-pair np.corrcoef loop, including NaN masking
    (pairwise-complete) and the Bonferroni floor -- proves the O(N^2)->BLAS rewrite preserves semantics."""
    rng = np.random.default_rng(7)
    n, T = 12, 260
    vshock = rng.standard_normal((T, n))
    sqrt_pk = np.abs(rng.standard_normal((T, n))) + 0.1
    vshock[rng.random((T, n)) < 0.05] = np.nan          # sprinkle missing values
    sqrt_pk[rng.random((T, n)) < 0.05] = np.nan
    for alpha in (None, 0.05):
        A_vec = EH.directed_vol2pk_hmatched(vshock, sqrt_pk, T - 1, horizon=3, top_k=4, alpha=alpha)
        A_ref = _bruteforce_hmatched(vshock, sqrt_pk, T - 1, 3, 4, alpha, MR._MIN_PAIRS)
        assert np.allclose(A_vec, A_ref, atol=1e-6, equal_nan=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
