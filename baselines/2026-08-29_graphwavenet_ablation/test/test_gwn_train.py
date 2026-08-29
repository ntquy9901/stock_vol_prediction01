"""Fast unit test for ``train_gwn`` control flow on tiny SYNTHETIC data (no real panel, milliseconds).

Covers the epoch loop, improve/​else (early-stop wait) branches, the break, best-state reload, both
``return_splits`` paths, and both ``adaptive`` variants -- so the GWN training loop is exercised without the
slow real-data smoke. Mirrors the sibling ``test_train_learned`` lr=0 early-stop trick.
"""
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("GWN_FORCE_CPU", "1")
_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_gwn_ablation as R  # noqa: E402
from config import Config  # noqa: E402


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
    return replace(Config(), epochs=6, min_epochs=1, patience=1, seeds=(42,), dropout=0.1)


@pytest.mark.parametrize("adaptive", [True, False])
def test_train_gwn_returns_finite_positive_predictions(cfg, adaptive):
    D = _synthetic_D()
    te = R.train_gwn(D, cfg, 42, adaptive, bs=8, skip_channels=16, end_channels=32)
    assert te.shape == (len(D.X_te), D.N)
    assert np.isfinite(te).all() and (te > 0).all()   # positivity floor holds


def test_train_gwn_return_splits_has_curves(cfg):
    D = _synthetic_D()
    out = R.train_gwn(D, cfg, 42, True, bs=8, skip_channels=16, end_channels=32, return_splits=True)
    for k in ("test", "val", "train", "train_curve", "val_curve", "best_epoch"):
        assert k in out
    assert 1 <= len(out["train_curve"]) <= cfg.epochs
    assert len(out["train_curve"]) == len(out["val_curve"])
    assert np.isfinite(out["train"]).all() and np.isfinite(out["val"]).all()


def test_train_gwn_early_stops_when_val_never_improves():
    """lr=0 -> weights never update -> val MSE constant -> every epoch after the first is non-improving,
    deterministically exercising the wait-increment + early-stop break."""
    D = _synthetic_D()
    cfg0 = replace(Config(), lr=0.0, epochs=6, min_epochs=1, patience=1, seeds=(42,), dropout=0.0)
    out = R.train_gwn(D, cfg0, 42, True, bs=8, skip_channels=16, end_channels=32, return_splits=True)
    assert out["best_epoch"] == 1                     # first epoch is best; later epochs never improve
    assert len(out["val_curve"]) == 2                 # stopped one epoch after the plateau (patience=1)
