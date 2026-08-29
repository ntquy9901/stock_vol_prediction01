"""Coverage for the panel-agnostic MTGNN ablation runner WITHOUT GPU training.

``run_training`` is exercised on a tiny real HNX slice with BOTH ``train_masked_rich`` (fixed-edge
variants) and ``train_learned`` (the MTGNN variant) monkeypatched to deterministic stubs, so the
metric / DM / over-under-fit plumbing runs on CPU in a fraction of a second (no epochs, no GPU).
Mirrors 2026-08-29_sector_gat_ablation/test/test_runner_and_fetch.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("LEARNED_ABLATION_FORCE_CPU", "1")
_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_learned_ablation as R  # noqa: E402
from config import Config, SMOKE  # noqa: E402

PANEL = "hnx"
CONFIGS = {R.NO_GRAPH, R.STAT, R.SECTOR, R.LEARNED}


def _tiny_D():
    keep = R.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:8])
    td = tempfile.mkdtemp()
    return R.build_panel(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)


def test_default_sector_csv_routing():
    assert R.default_sector_csv("sp500").name == "sp500_gics_sectors.csv"
    assert R.default_sector_csv("hnx").name == "vn_sectors.csv"
    assert R.default_sector_csv("vn100").name == "vn_sectors.csv"


def _stub_split(y_te, y_va, y_tr, off):
    return {"test": y_te + off, "val": y_va + off, "train": y_tr + off,
            "train_curve": [1e-6], "val_curve": [1.1e-6], "best_epoch": 1}


def test_run_training_stubbed(monkeypatch, tmp_path):
    D = _tiny_D()

    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        off = 0.0 if not use_graph else (1e-4 if adj is Dd.adj_vol2pk else 2e-4)
        return _stub_split(Dd.y_te, Dd.y_va, Dd.y_tr, off) if return_splits else Dd.y_te + off

    def fake_learned(Dd, cfg, seed, subgraph_size=20, node_dim=40, alpha=3.0, return_splits=False):
        off = 3e-4
        return _stub_split(Dd.y_te, Dd.y_va, Dd.y_tr, off) if return_splits else Dd.y_te + off

    monkeypatch.setattr(R.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(R, "train_learned", fake_learned)
    monkeypatch.setattr(R, "build_panel",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: D)
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    res = R.run_training(PANEL, cfg, horizon=1, subgraph_size=4, node_dim=8, alpha=3.0,
                         out_dir=str(tmp_path))
    assert set(res["metrics"]) == CONFIGS
    assert res["num_nodes"] == D.N
    assert res["device"] == "cpu"
    assert res["mtgnn"]["subgraph_size_k"] == min(4, D.N)
    assert set(res["dm_date_clustered"]) == {
        "learned_vs_no_graph", "learned_vs_stat_vol2pk", "learned_vs_sector",
        "stat_vs_no_graph", "sector_vs_no_graph"}
    assert (tmp_path / "learned_graph_ablation_hnx_h1.json").exists()
    for m in res["metrics"].values():
        assert np.isfinite(m["qlike"])
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)
    # out_dir=None branch: results returned, nothing written
    res2 = R.run_training(PANEL, cfg, horizon=1, subgraph_size=4, node_dim=8, alpha=3.0, out_dir=None)
    assert set(res2["metrics"]) == CONFIGS


def test_run_training_emits_overfit_evidence(monkeypatch, tmp_path):
    """run_training must stamp train/val metrics + per-model fit verdict + per-seed learning curves
    (CLAUDE.md over/under-fit mandate). Split-DISTINGUISHABLE errors (train < val < test) pin that each
    split's metrics come from the CORRECT split array."""
    D = _tiny_D()

    def _split(Dd, goff):
        return {"test": Dd.y_te + 1e-5 + goff, "val": Dd.y_va + 5e-6 + goff,
                "train": Dd.y_tr + 1e-6 + goff,
                "train_curve": [1e-6, 5e-7], "val_curve": [1.1e-6, 6e-7], "best_epoch": 2}

    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        goff = 0.0 if not use_graph else (1e-7 if adj is Dd.adj_vol2pk else 2e-7)
        return _split(Dd, goff) if return_splits else Dd.y_te + 1e-5 + goff

    def fake_learned(Dd, cfg, seed, subgraph_size=20, node_dim=40, alpha=3.0, return_splits=False):
        return _split(Dd, 3e-7) if return_splits else Dd.y_te + 1e-5 + 3e-7

    monkeypatch.setattr(R.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(R, "train_learned", fake_learned)
    monkeypatch.setattr(R, "build_panel",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: D)
    cfg = Config(seeds=(42, 123), epochs=2, min_epochs=1, patience=1)
    res = R.run_training(PANEL, cfg, horizon=1, subgraph_size=4, node_dim=8, alpha=3.0,
                         out_dir=str(tmp_path))
    for block in ("train_metrics", "val_metrics", "fit_diagnostics", "learning_curves"):
        assert block in res, f"missing {block}"
        assert set(res[block]) == CONFIGS, block
    for k in CONFIGS:
        assert {"qlike", "r2"} <= set(res["train_metrics"][k])
        assert {"qlike", "r2"} <= set(res["val_metrics"][k])
        assert res["fit_diagnostics"][k]["status"] in {"ok", "overfit", "underfit", "unknown"}
        lc = res["learning_curves"][k]
        assert len(lc["train"]) == 2 and len(lc["val"]) == 2 and len(lc["best_epoch"]) == 2
        assert isinstance(lc["train"][0], list)   # per-epoch MSE list per seed
        assert res["train_metrics"][k]["mse"] < res["val_metrics"][k]["mse"] < res["metrics"][k]["mse"]


def test_run_dry_builds_and_forwards(monkeypatch):
    """run_dry: build the tiny panel + ONE CPU forward of the learned-graph net (no training)."""
    D = _tiny_D()
    monkeypatch.setattr(R, "build_panel",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: D)
    out = R.run_dry(PANEL, horizon=1, max_tickers=8, subgraph_size=4, node_dim=8, alpha=3.0)
    assert out["n_nodes"] == D.N
    assert out["forward_shape"][-1] == D.N
    # max_tickers falsy branch: no ticker cap applied
    out2 = R.run_dry(PANEL, horizon=1, max_tickers=0, subgraph_size=4, node_dim=8, alpha=3.0)
    assert out2["n_nodes"] == D.N
