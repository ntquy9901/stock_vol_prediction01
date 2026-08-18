"""Tests for the seq-lookback experiment scripts (dm_seq_compare + run_seq helpers).

Focus: the cross-seq DM must pair on the INTERSECTION of the two runs' test observations (a shorter
lookback produces extra early test windows the longer one lacks), align targets on that intersection,
and raise on genuine target mismatch; plus the run_seq 90/10 ratio patch and the basis smoke-assert.

Run (GPU venv, torch available):  python -m pytest scripts/seq_lookback/test -q
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_SEQDIR = _HERE.parents[1]
_ROOT = _HERE.parents[3]
for _p in (str(_SEQDIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dm_seq_compare  # noqa: E402


def _write_dump(results: Path, ts: str, h: int, seed: int, rung_dir: str, rows: list[dict]) -> None:
    d = results / f"volatility_retrain_h{h}_seed{seed}_{ts}" / rung_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions_test.json").write_text(json.dumps(rows), encoding="utf-8")


def _row(tid: int, date: str, target: float, pred: float) -> dict:
    return {"ticker_id": tid, "target_date": date, "target_raw": target, "prediction_raw": pred}


# ----------------------------- dm_seq_compare -----------------------------

def test_dm_seq_intersects_common_keys(tmp_path, monkeypatch):
    """seq5 has an extra early obs (d1) that seq10 lacks -> DM must pair only on the common {d2,d3}."""
    monkeypatch.setattr(dm_seq_compare, "RESULTS", tmp_path)
    # A = short lookback: 3 obs incl. early d1; B = ref: only d2,d3. Same targets on the overlap.
    _write_dump(tmp_path, "TSA", 1, 42, "FULL",
                [_row(0, "d1", 0.5, 0.4), _row(0, "d2", 0.6, 0.5), _row(0, "d3", 0.7, 0.6)])
    _write_dump(tmp_path, "TSB", 1, 42, "FULL",
                [_row(0, "d2", 0.6, 0.9), _row(0, "d3", 0.7, 0.9)])
    r = dm_seq_compare.dm_seq("TSA", "TSB", 1, "FULL", [42], "qlike")
    assert r["n"] == 2                      # intersection {d2,d3}, NOT 3
    assert r["favors"] == "A(short)"        # A preds closer to targets -> lower loss


def test_dm_seq_raises_on_target_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dm_seq_compare, "RESULTS", tmp_path)
    _write_dump(tmp_path, "TSA", 1, 42, "FULL", [_row(0, "d2", 0.6, 0.5)])
    _write_dump(tmp_path, "TSB", 1, 42, "FULL", [_row(0, "d2", 0.99, 0.5)])  # different target for d2
    with pytest.raises(ValueError, match="targets differ"):
        dm_seq_compare.dm_seq("TSA", "TSB", 1, "FULL", [42], "qlike")


def test_dm_seq_raises_on_no_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(dm_seq_compare, "RESULTS", tmp_path)
    _write_dump(tmp_path, "TSA", 1, 42, "FULL", [_row(0, "d1", 0.6, 0.5)])
    _write_dump(tmp_path, "TSB", 1, 42, "FULL", [_row(0, "d9", 0.6, 0.5)])
    with pytest.raises(ValueError, match="no common test observations"):
        dm_seq_compare.dm_seq("TSA", "TSB", 1, "FULL", [42], "qlike")


def test_load_run_seed_ensembles_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(dm_seq_compare, "RESULTS", tmp_path)
    _write_dump(tmp_path, "TS", 1, 42, "FULL", [_row(0, "d1", 0.6, 0.4)])
    _write_dump(tmp_path, "TS", 1, 7, "FULL", [_row(0, "d1", 0.6, 0.8)])
    run = dm_seq_compare._load_run("TS", 1, "FULL", [42, 7])
    assert run[(0, "d1")][0] == pytest.approx(0.6)         # target
    assert run[(0, "d1")][1] == pytest.approx(0.6)         # mean(0.4, 0.8)


def test_load_run_raises_on_seed_obs_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(dm_seq_compare, "RESULTS", tmp_path)
    _write_dump(tmp_path, "TS", 1, 42, "FULL", [_row(0, "d1", 0.6, 0.4)])
    _write_dump(tmp_path, "TS", 1, 7, "FULL", [_row(0, "d2", 0.6, 0.4)])  # different obs across seeds
    with pytest.raises(ValueError, match="different observations"):
        dm_seq_compare._load_run("TS", 1, "FULL", [42, 7])


# ----------------------------- run_seq helpers -----------------------------
# run_seq imports run_retrain_trainval (torch); skip these if torch is unavailable in the runner.

run_seq = pytest.importorskip("run_seq", reason="run_seq needs torch (run under the GPU venv)")


def test_split_9010_uses_correct_ratios(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_seq, "_ORIG_SPLIT", lambda p, ratios: captured.update(p=p, ratios=ratios))
    run_seq._split_9010("PROC")
    assert captured["ratios"] == (0.80, 0.10, 0.10)
    assert captured["p"] == "PROC"


def _basis_from_snap(x_price: np.ndarray):
    """Build a fake (pooled, graph, store, allowed, edges) whose graph.snapshots[0].x_price is given
    (the [nodes, seq, feats] tensor the model consumes, which _build_with_smoke inspects)."""
    snap = types.SimpleNamespace(x_price=np.asarray(x_price, dtype=np.float32))
    graph = types.SimpleNamespace(snapshots=[snap])
    return ("pooled", graph, "store", ["allowed_sample"], 165)


def _uniform(n_nodes: int, seq_len: int, price_dim: int, value: float) -> np.ndarray:
    return np.full((n_nodes, seq_len, price_dim), value, dtype=np.float32)


def test_build_smoke_passes_valid_basis(monkeypatch):
    import combo_ladder
    monkeypatch.setattr(combo_ladder, "SEQ", 5)
    monkeypatch.setattr(run_seq, "_ORIG_BUILD", lambda stamp: _basis_from_snap(_uniform(3, 5, 5, 0.3)))
    out = run_seq._build_with_smoke(Path("stamp"))
    assert out[4] == 165


def test_build_smoke_rejects_wrong_price_dim(monkeypatch):
    import combo_ladder
    monkeypatch.setattr(combo_ladder, "SEQ", 5)
    monkeypatch.setattr(run_seq, "_ORIG_BUILD", lambda stamp: _basis_from_snap(_uniform(3, 5, 4, 0.3)))
    with pytest.raises(AssertionError, match="price_dim=5"):
        run_seq._build_with_smoke(Path("stamp"))


def test_build_smoke_rejects_wrong_seq_dim(monkeypatch):
    import combo_ladder
    monkeypatch.setattr(combo_ladder, "SEQ", 5)
    monkeypatch.setattr(run_seq, "_ORIG_BUILD", lambda stamp: _basis_from_snap(_uniform(3, 7, 5, 0.3)))
    with pytest.raises(AssertionError, match="lookback dim"):
        run_seq._build_with_smoke(Path("stamp"))


def test_build_smoke_rejects_all_zero_block(monkeypatch):
    import combo_ladder
    monkeypatch.setattr(combo_ladder, "SEQ", 5)
    monkeypatch.setattr(run_seq, "_ORIG_BUILD", lambda stamp: _basis_from_snap(_uniform(3, 5, 5, 0.0)))
    with pytest.raises(AssertionError, match="all-zero"):
        run_seq._build_with_smoke(Path("stamp"))


def test_build_smoke_rejects_single_zeroed_column(monkeypatch):
    """The per-column guard must catch ONE feature column silently zeroed (others nonzero)."""
    import combo_ladder
    monkeypatch.setattr(combo_ladder, "SEQ", 5)
    x = _uniform(3, 5, 5, 0.3)
    x[:, :, 2] = 0.0                       # feature column 2 fully zeroed
    monkeypatch.setattr(run_seq, "_ORIG_BUILD", lambda stamp: _basis_from_snap(x))
    with pytest.raises(AssertionError, match="column 2 is all-zero"):
        run_seq._build_with_smoke(Path("stamp"))


def test_ablation_wrapper_sets_seq_and_injects_smoke_without_ratio_patch(monkeypatch):
    """run_seq_ablation.main must set combo_ladder.SEQ, inject the smoke at run_volatility.build_basis,
    and leave the split ratio (load_and_split_price_data) at the harness 70/15/15 default."""
    run_seq_ablation = pytest.importorskip("run_seq_ablation", reason="needs torch (GPU venv)")
    import combo_ladder
    import run_volatility
    called = {}
    # register globals main() mutates in place so monkeypatch restores them on teardown
    monkeypatch.setattr(combo_ladder, "SEQ", combo_ladder.SEQ)
    monkeypatch.setattr(run_volatility, "build_basis", run_volatility.build_basis)
    monkeypatch.setattr(run_seq_ablation.run_ablation, "main",
                        lambda *a, **k: called.update(args=a, kwargs=k))
    orig_split = combo_ladder.load_and_split_price_data
    monkeypatch.setattr(sys, "argv", ["run_seq_ablation.py", "5", "TSX", "cpu", "42", "10", "1", "5"])
    run_seq_ablation.main()
    assert combo_ladder.SEQ == 5
    assert run_volatility.build_basis is run_seq_ablation._build_with_smoke   # smoke injected
    assert combo_ladder.load_and_split_price_data is orig_split               # ratio NOT patched
    assert called["args"][0] == "TSX" and called["args"][4] == (1, 5)         # ts + horizons forwarded


@pytest.mark.smoke
def test_smoke_dm_seq_end_to_end(tmp_path, monkeypatch):
    """Boot the DM path on synthetic dumps across all loss families (happy path).

    Predictions are NEITHER purely additive NOR purely proportional to the target, so the per-obs
    loss differential has genuine variance for QLIKE, SE and AE alike (a degenerate constant
    differential would trip diebold_mariano's non-positive-variance guard)."""
    monkeypatch.setattr(dm_seq_compare, "RESULTS", tmp_path)
    def tgt(i): return 0.5 + 0.01 * i
    rows_a = [_row(0, f"d{i}", tgt(i), tgt(i) * 1.02 + 0.001 * (i % 3)) for i in range(30)]
    rows_b = [_row(0, f"d{i}", tgt(i), tgt(i) * 0.90 + 0.002 * (i % 4)) for i in range(5, 30)]
    _write_dump(tmp_path, "TSA", 5, 42, "FULL", rows_a)
    _write_dump(tmp_path, "TSB", 5, 42, "FULL", rows_b)
    for loss in ("qlike", "se", "ae"):
        r = dm_seq_compare.dm_seq("TSA", "TSB", 5, "FULL", [42], loss)
        assert r["n"] == 25 and np.isfinite(r["dm_hln"])
