"""Tests for the GNNHAR baseline: faithful-model forward shapes, causal (train-only) graph, the
learning-curve training loop, restart-on-collapse, QLIKE/DM wiring, and an integration ``run()`` smoke
on tiny synthetic frames asserting no-exception + the full over/under-fit evidence schema."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import gnnhar_config as cfg
import gnnhar_sp500 as G
import overfit_check as OF
import run_gnnhar as R
import vn_gbm_graph_stage1 as S1


def _synth_frames(n_tickers=6, seed=0):
    """Frames matching FM.load's output shape: per-ticker enriched dataframe over 2021..2023 business days
    with a positive AR(1) Parkinson-variance series + HAR lags + the _feat-derived own-history columns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", "2023-03-31")
    frames = {}
    for i in range(n_tickers):
        logv = np.zeros(len(dates))
        logv[0] = rng.normal(-9, 0.3)
        for t in range(1, len(dates)):
            logv[t] = 0.95 * logv[t - 1] + rng.normal(0, 0.25) - 0.45 * 9 * 0.05
        pk = np.exp(logv)
        d = pd.DataFrame({"date": dates, "parkinson_variance": pk})
        d = S1._feat(d)                                   # adds logpk, rq, mr_* (own-history)
        pks = pd.Series(pk)
        d["har_daily"] = pk
        d["har_weekly"] = pks.rolling(5, min_periods=1).mean().to_numpy()
        d["har_monthly"] = pks.rolling(22, min_periods=1).mean().to_numpy()
        d["ticker"] = f"T{i}"
        frames[f"T{i}"] = d
    return frames


def _load_fn(market):
    return _synth_frames(), {}, {}


# ---------------------------------------------------------------- config / faithful model
def test_config_own8_har3():
    assert "rq" not in cfg.OWN8
    assert len(cfg.OWN8) == 8
    assert set(cfg.HAR3) <= set(cfg.OWN8)
    assert cfg.HAR3 == ["har_daily", "har_weekly", "har_monthly"]


def test_model_forward_shapes_nonneg():
    model = G.GNNHAR(len(cfg.HAR3), cfg.N_HID, cfg.N_GCN)
    x = torch.randn(4, 6, len(cfg.HAR3))
    adj = torch.eye(6)
    out = model(x, adj)
    assert out.shape == (4, 6)
    assert torch.all(out >= 0)                            # relu output (variance forecast is non-negative)


def test_metrics5_keys():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 3.2])
    m = R._metrics5(y, p)
    assert set(m) == {"mse", "rmse", "mae", "r2", "qlike"}


# ---------------------------------------------------------------- training loop / curves / restart
def _tiny_tensors(n_dates=40, n_nodes=5, seed=1):
    rng = np.random.default_rng(seed)
    dev = R.DEVICE                                         # train loop's contract: inputs already on device
    X = torch.as_tensor(rng.normal(size=(n_dates, n_nodes, len(cfg.HAR3))).astype(np.float32), device=dev)
    Ys = torch.as_tensor(np.abs(rng.normal(1.0, 0.2, size=(n_dates, n_nodes))).astype(np.float32), device=dev)
    Mt = torch.ones((n_dates, n_nodes), dtype=torch.float32, device=dev)
    adj = torch.eye(n_nodes, device=dev)
    tr_idx = np.arange(n_dates - cfg.VALID_LEN)
    va_idx = np.arange(n_dates - cfg.VALID_LEN, n_dates)
    return X, Ys, Mt, adj, tr_idx, va_idx


def test_train_with_curves_records_curve():
    X, Ys, Mt, adj, tr_idx, va_idx = _tiny_tensors()
    model, best_val, curve = R.train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx,
                                                  len(cfg.HAR3), cfg.N_GCN, 0, max_epochs=4, patience=3)
    assert isinstance(model, G.GNNHAR)
    assert 1 <= len(curve) <= 4
    assert set(curve[0]) == {"epoch", "train", "val"}
    assert np.isfinite(best_val)


def test_train_with_curves_zero_epochs():
    X, Ys, Mt, adj, tr_idx, va_idx = _tiny_tensors()
    model, best_val, curve = R.train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx,
                                                  len(cfg.HAR3), cfg.N_GCN, 0, max_epochs=0, patience=3)
    assert curve == []                                    # no epoch ran -> best_state stayed None
    assert best_val == float("inf")


def test_fit_restarts_on_collapse(monkeypatch):
    calls = {"n": 0}
    good = G.GNNHAR(len(cfg.HAR3), cfg.N_HID, cfg.N_GCN)

    def fake(*a, seed, **k):
        calls["n"] += 1
        return (good, 9.9, []) if calls["n"] == 1 else (good, 0.5, [{"epoch": 0, "train": 1.0, "val": 0.5}])

    monkeypatch.setattr(R, "train_with_curves",
                        lambda X, Ys, Mt, adj, tr, va, inf, ng, seed, me, pa: fake(seed=seed))
    X, Ys, Mt, adj, tr_idx, va_idx = _tiny_tensors()
    model, best_val, curve = R.fit(X, Ys, Mt, adj, tr_idx, va_idx, len(cfg.HAR3), cfg.N_GCN, 0, 4, 3)
    assert calls["n"] == 2                                # collapsed once, restarted, took the good run
    assert best_val == 0.5 and len(curve) == 1


# ---------------------------------------------------------------- causal graph (train-only)
def test_build_graph_receives_only_train_rows(monkeypatch):
    seen = {}
    orig = S1.build_graph

    def spy(train, tickers, rng):
        seen["max_date"] = pd.Timestamp(train["date"].max())
        return orig(train, tickers, rng)

    monkeypatch.setattr(S1, "build_graph", spy)
    R.run("sp500", load_fn=_load_fn, smoke=True, min_train=50, max_epochs=2, seeds=(0,))
    embargo = pd.Timedelta(days=int(1 * 1.6) + 5)
    fold_start = pd.Timestamp(S1.FOLDS[0])
    assert seen["max_date"] < fold_start - embargo         # graph built strictly on pre-embargo train rows


# ---------------------------------------------------------------- integration run() smoke
def test_run_smoke_structure_sp500():
    out = R.run("sp500", load_fn=_load_fn, smoke=True, min_train=50, max_epochs=2, seeds=(0,))
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.MODELS:
            assert f"{m}_h1" in out[blk]
            assert set(out[blk][f"{m}_h1"]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert out["fit_diagnostics"]["GNNHAR_h1"]["status"] in {"ok", "overfit", "underfit", "unknown"}
    assert len(out["learning_curves"]["GNNHAR_h1"]) == 1           # one seed
    d = out["dm"]["GNNHAR_vs_HAR_h1"]
    assert 0.0 <= d["p_value"] <= 1.0 and "gain_pct" in d
    assert "GNNHAR_h1" in OF.learned_models(out)                   # gate will auto-detect + validate it
    ok, problems = OF.check_result_evidence(out)                   # runs without error (verdict data-dependent)
    assert isinstance(ok, bool) and isinstance(problems, list)
    assert "per_fold_qlike" not in out                             # sp500 branch


def test_run_smoke_hose_perfold_and_checkpoint(tmp_path):
    outp = tmp_path / "gnnhar_hose.json"
    out = R.run("hose", load_fn=_load_fn, out_path=outp, smoke=True, min_train=50, max_epochs=2, seeds=(0,))
    assert "per_fold_qlike" in out
    assert len(out["per_fold_qlike"]["GNNHAR_h1"]) == out["n_folds"]["h1"]
    saved = json.loads(outp.read_text())                          # atomic checkpoint wrote valid JSON
    assert saved["metrics"]["GNNHAR_h1"]["qlike"] == out["metrics"]["GNNHAR_h1"]["qlike"]


def test_run_no_qualifying_fold_returns_empty():
    # no explicit min_train -> falls back to cfg.MIN_TRAIN_ROWS["sp500"]=30000, which the tiny synthetic
    # panel cannot reach, so every fold is gated out and _horizon returns None.
    out = R.run("sp500", load_fn=_load_fn, smoke=True, max_epochs=2, seeds=(0,))
    assert out["metrics"] == {}                                   # every fold gated out -> _horizon None
    assert out["dm"] == {}


def test_train_with_curves_early_stops(monkeypatch):
    # constant validation loss (grad-carrying so backward works) -> no improvement after epoch 0 -> the
    # patience/min-epoch early-stop branch fires. MIN_EPOCHS lowered so the break is reachable quickly.
    monkeypatch.setattr(R.G, "_qlike_loss", lambda pred, y, mask: (pred * 0).sum() + 1.0)
    monkeypatch.setattr(cfg, "MIN_EPOCHS", 1)
    X, Ys, Mt, adj, tr_idx, va_idx = _tiny_tensors()
    _model, best_val, curve = R.train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx,
                                                   len(cfg.HAR3), cfg.N_GCN, 0, max_epochs=10, patience=2)
    assert len(curve) < 10                                        # stopped early via patience
    assert best_val == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
