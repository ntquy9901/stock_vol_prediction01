"""Coverage for the checkpointed DY-ablation driver: training stubbed, checkpoints written/skipped,
final JSON assembled from checkpoints. No GPU, no epochs.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_dy_incremental as RDI  # noqa: E402
from config import SMOKE, Config  # noqa: E402

PANEL = "hnx"


def _tiny_D_files():
    keep = RDI.RDA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:8])
    td = tempfile.mkdtemp()
    D, files = RDI.RDA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
    return D, files


def test_adj_for_routing():
    class _D:
        adj_vol2pk = np.eye(3, dtype=np.float32)
    dy = np.full((3, 3), 0.5, dtype=np.float32)
    ug, a = RDI._adj_for("dy_GAT", _D, dy)
    assert ug is True and a is dy
    ug, a = RDI._adj_for("stat_GAT_vol2pk", _D, dy)
    assert ug is True and a is _D.adj_vol2pk
    ug, a = RDI._adj_for("no_graph_LSTM", _D, dy)
    assert ug is False and a is _D.adj_vol2pk


def test_run_checkpoints_and_assembles(monkeypatch, tmp_path):
    D, files = _tiny_D_files()

    calls = {"n": 0}

    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        calls["n"] += 1
        off = 0.0 if not use_graph else (1e-4 if adj is Dd.adj_vol2pk else 2e-4)
        # full split output (test/val/train predictions + learning curves) so the assembled result
        # carries the mandated over/under-fit evidence
        return {"test": Dd.y_te + off + seed * 1e-9,
                "val": Dd.y_va + off, "train": Dd.y_tr + off,
                "train_curve": [1e-4, 8e-5], "val_curve": [1.2e-4, 1.0e-4], "best_epoch": 2}

    monkeypatch.setattr(RDI.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(RDI.RDA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, files))
    monkeypatch.setattr(RDI.RDA, "dy_adj_for",
                        lambda D_, files_, **k: (np.eye(D.N, dtype=np.float32),
                                                 {"total_connectedness_index": 10.0, "row_sum_mean": 1.0}))

    res = RDI.run(PANEL, 1, epochs=1, seeds=[42, 123], out_dir=str(tmp_path))
    # 3 variants x 2 seeds = 6 trainings, all checkpointed
    assert calls["n"] == 6
    assert set(res["metrics_ensemble"]) == set(RDI.VARIANTS)
    assert set(res["dm"]) == {"dy_vs_no_graph", "dy_vs_stat", "stat_vs_no_graph"}
    assert (tmp_path / "dy_ablation_hnx_h1.json").exists()
    assert (tmp_path / "ckpt" / "dy_stats.json").exists()
    for m in res["metrics_ensemble"].values():
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)
    # mandated over/under-fit evidence present for every variant
    assert set(res["train_metrics"]) == set(RDI.VARIANTS)
    assert set(res["val_metrics"]) == set(RDI.VARIANTS)
    for v in RDI.VARIANTS:
        assert "status" in res["fit_diagnostics"][v]
        assert len(res["learning_curves"][v]["train"]) == 2   # 2 seeds
        assert len(res["learning_curves"][v]["best_epoch"]) == 2

    # re-run: every checkpoint exists -> zero new trainings, same assembly
    calls["n"] = 0
    res2 = RDI.run(PANEL, 1, epochs=1, seeds=[42, 123], out_dir=str(tmp_path))
    assert calls["n"] == 0
    assert res2["num_nodes"] == res["num_nodes"]


def test_train_one_returns_split_output(monkeypatch):
    class _D:
        adj_vol2pk = np.eye(2, dtype=np.float32)
        N = 2
    captured = {}

    def fake(D, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        captured["return_splits"] = return_splits
        captured["use_graph"] = use_graph
        return {"test": np.zeros((1, 2)), "val": np.zeros((1, 2)), "train": np.zeros((1, 2)),
                "train_curve": [1.0], "val_curve": [1.0], "best_epoch": 1}

    monkeypatch.setattr(RDI.RMR, "train_masked_rich", fake)
    out = RDI.train_one(_D, Config(seeds=(42,)), "dy_GAT", 42, np.eye(2, dtype=np.float32))
    assert captured["return_splits"] is True and captured["use_graph"] is True
    assert set(out) == {"test", "val", "train", "train_curve", "val_curve", "best_epoch"}
