"""CPU integration test: run train_deep end-to-end on tiny synthetic data for all (loss, anchor) configs.
This EXECUTES the trainer (no GPU needed -- torch falls back to CPU), so train_deep is covered rather than
pragma-excluded."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import qa_train as QT  # noqa: E402


def _synth(seed=0):
    rng = np.random.default_rng(seed)
    A, N, seq, F = 6, 3, 4, 5
    def X(a): return rng.random((a, N, seq, F)).astype(np.float32)
    def y(a): return (rng.random((a, N)).astype(np.float32) * 1e-3 + 1e-4)
    ones = lambda a: np.ones((a, N), dtype=np.float32)
    D = SimpleNamespace(
        X_tr=X(A), X_va=X(3), X_te=X(3),
        nmask_tr=ones(A), nmask_va=ones(3), nmask_te=ones(3),
        tmask_tr=ones(A), tmask_va=ones(3), tmask_te=ones(3),
        y_tr=y(A), y_va=y(3), y_te=y(3),
        t_mean=(rng.random(N).astype(np.float32) * 1e-3 + 1e-4),
        t_std=(rng.random(N).astype(np.float32) * 1e-3 + 1e-4))
    return D, A, N


def _cfg():
    return SimpleNamespace(hidden=8, heads=2, dropout=0.0, lr=1e-3, weight_decay=0.0, batch_size=3,
                           grad_clip=1.0, epochs=2, min_epochs=1, patience=1, qlike_floor=1e-8)


@pytest.mark.parametrize("loss", ["mse", "qlike"])
@pytest.mark.parametrize("anchor", ["none", "harx"])
def test_train_deep_runs_all_configs(loss, anchor):
    D, A, N = _synth()
    adj = np.eye(N, dtype=np.float32)
    harx = None
    if anchor == "harx":
        base = np.full((A, N), 5e-4, dtype=np.float32)
        oof = base.copy(); oof[:2] = np.nan                    # warm-up cells have no OOF -> masked out of loss
        harx = {"tr": oof, "va": np.full((3, N), 5e-4, np.float32),
                "te": np.full((3, N), 5e-4, np.float32), "tr_base": base}
    out = QT.train_deep(D, _cfg(), seed=0, use_graph=(anchor == "none"), adj=adj,
                        loss=loss, anchor=anchor, harx=harx, clip=0.5, return_splits=True)
    for sp in ("train", "val", "test"):
        assert out[sp].shape == getattr(D, f"y_{sp[:2] if sp != 'test' else 'te'}" if False else
                                        {"train": "y_tr", "val": "y_va", "test": "y_te"}[sp]).shape
        assert np.all(np.isfinite(out[sp])) and np.all(out[sp] > 0)   # positive floored, no NaN
    assert len(out["val_curve"]) == len(out["train_curve"]) >= 1
    assert 1 <= out["best_epoch"] <= 2
    te = QT.train_deep(D, _cfg(), seed=0, use_graph=(anchor == "none"), adj=adj,
                       loss=loss, anchor=anchor, harx=harx, clip=0.5, return_splits=False)
    assert te.shape == D.y_te.shape and np.all(te > 0)         # return_splits=False path
    # P2 fix: curves record the training OBJECTIVE (finite, on the correct scale) -- qlike curves are the
    # QLIKE loss (O(0.1-1)), mse curves are the tiny variance MSE (<1e-2); never NaN.
    assert np.all(np.isfinite(out["val_curve"])) and np.all(np.isfinite(out["train_curve"]))
    if loss == "qlike":
        assert out["val_curve"][-1] > 1e-3                     # QLIKE scale, not the ~1e-6 MSE scale
