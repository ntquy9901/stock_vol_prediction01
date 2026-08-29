"""Fast unit test for ``train_learned`` control flow on tiny SYNTHETIC data (no real panel, milliseconds).

Covers the epoch loop, the improve/​else (early-stop wait) branches, the break, best-state reload, and both
``return_splits`` paths -- so the training loop is exercised without the slow real-data smoke.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("LEARNED_ABLATION_FORCE_CPU", "1")
_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_learned_ablation as R  # noqa: E402
from config import Config  # noqa: E402
from dataclasses import replace  # noqa: E402


def _synthetic_D(n_nodes=4, seq=10, n_tr=16, n_va=6, n_te=5):
    rng = np.random.default_rng(0)
    def blk(n):
        return (rng.standard_normal((n, n_nodes, seq, 5)).astype(np.float32),
                np.ones((n, n_nodes), np.float32),
                np.ones((n, n_nodes), np.float32),
                np.abs(rng.standard_normal((n, n_nodes)).astype(np.float32)) * 1e-3 + 1e-4)
    Xtr, nmtr, tmtr, ytr = blk(n_tr)
    Xva, nmva, tmva, yva = blk(n_va)
    Xte, nmte, tmte, yte = blk(n_te)
    return SimpleNamespace(
        tickers=[f"T{i}" for i in range(n_nodes)], N=n_nodes,
        t_mean=(np.abs(rng.standard_normal(n_nodes)) * 1e-3 + 1e-4).astype(np.float32),
        t_std=(np.abs(rng.standard_normal(n_nodes)) * 1e-3 + 1e-4).astype(np.float32),
        X_tr=Xtr, nmask_tr=nmtr, tmask_tr=tmtr, y_tr=ytr,
        X_va=Xva, nmask_va=nmva, tmask_va=tmva, y_va=yva,
        X_te=Xte, nmask_te=nmte, tmask_te=tmte, y_te=yte)


@pytest.fixture
def cfg():
    return replace(Config(), epochs=8, min_epochs=1, patience=1, seeds=(42,),
                   batch_size=8, hidden=8, heads=2, dropout=0.1)


def test_train_learned_returns_finite_test_predictions(cfg):
    D = _synthetic_D()
    te = R.train_learned(D, cfg, 42, subgraph_size=2, node_dim=8, alpha=3.0)
    assert te.shape == (len(D.X_te), D.N)
    assert np.isfinite(te).all() and (te > 0).all()   # positivity floor holds


def test_train_learned_return_splits_has_curves_and_early_stop(cfg):
    D = _synthetic_D()
    out = R.train_learned(D, cfg, 42, subgraph_size=2, node_dim=8, alpha=3.0, return_splits=True)
    for k in ("test", "val", "train", "train_curve", "val_curve", "best_epoch"):
        assert k in out
    assert 1 <= len(out["train_curve"]) <= cfg.epochs      # early stopping may cut < epochs
    assert len(out["train_curve"]) == len(out["val_curve"])
    assert out["best_epoch"] >= 1
    assert np.isfinite(out["train"]).all() and np.isfinite(out["val"]).all()


def test_train_learned_early_stops_when_val_never_improves():
    """lr=0 -> the net never updates -> val MSE is constant -> after the first (best) epoch every later epoch
    is non-improving, deterministically exercising the wait-increment + early-stop break."""
    D = _synthetic_D()
    cfg0 = replace(Config(), lr=0.0, epochs=6, min_epochs=1, patience=1, seeds=(42,),
                   batch_size=8, hidden=8, heads=2, dropout=0.0)
    out = R.train_learned(D, cfg0, 42, subgraph_size=2, node_dim=8, alpha=3.0, return_splits=True)
    assert out["best_epoch"] == 1                    # first epoch is best; later epochs never improve
    assert len(out["val_curve"]) == 2                # stopped one epoch after the plateau (patience=1)
