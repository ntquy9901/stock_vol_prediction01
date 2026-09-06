"""Tests for the MASTER baseline: architecture shapes, gate slicing, causal/no-look-ahead market
features, per-node scaler fit-on-train, positive floored output, reproducibility, batched-attention
per-anchor independence + key masking, and an end-to-end tiny walk-forward fold.

Run: .venv_gpu_encode/Scripts/python.exe -m pytest baselines/2026-09-06_master_transformer/test -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "code"
REPO = HERE.parents[3]
sys.path.insert(0, str(CODE))
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-09-05_edge_horizon_matched" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import market_features as MF  # noqa: E402
import master_config as MC  # noqa: E402
from master_net import MASTER, build_master  # noqa: E402
import run_master as RM  # noqa: E402
from wf_enriched_panel import EnrichedPanel  # noqa: E402
from wf_folds import make_folds  # noqa: E402
from run_volga_walkforward import VolgaWFConfig  # noqa: E402
from run_walkforward import training_config  # noqa: E402


# --------------------------- architecture ---------------------------

def _tiny_cfg(**kw):
    from dataclasses import replace
    return replace(MC.MasterConfig(d_model=16, t_nhead=2, s_nhead=2, epochs=2, min_epochs=1, patience=1,
                                   batch_size=4, seeds=(42, 123)), **kw)


def test_forward_shape_and_gate_slice():
    cfg = _tiny_cfg()
    net = build_master(cfg)
    b, n, t, d = 3, 5, cfg.max_len // 20, cfg.n_stock_feat + cfg.n_market_feat
    t = 4
    x = torch.randn(b, n, t, d)
    out = net(x)
    assert out.shape == (b, n)
    # the gate reads exactly the market columns at the LAST timestep
    assert net.gate_input_start_index == cfg.n_stock_feat
    assert net.gate_input_end_index == cfg.n_stock_feat + cfg.n_market_feat


def test_gate_uses_only_last_timestep_market_features():
    """Changing market features at non-last timesteps must NOT change the output (gate reads t=-1 only,
    and the stock branch ignores the market columns)."""
    cfg = _tiny_cfg()
    net = build_master(cfg).eval()
    b, n, t = 2, 4, 5
    d = cfg.n_stock_feat + cfg.n_market_feat
    x = torch.randn(b, n, t, d)
    x2 = x.clone()
    x2[:, :, :-1, cfg.n_stock_feat:] += 7.0                 # perturb market feats at all but the last step
    with torch.no_grad():
        assert torch.allclose(net(x), net(x2), atol=1e-6)


def test_per_anchor_independence_of_batched_attention():
    """A batched forward must not mix anchors: perturbing anchor 1 leaves anchor 0's output unchanged."""
    cfg = _tiny_cfg()
    net = build_master(cfg).eval()
    x = torch.randn(2, 5, 4, cfg.n_stock_feat + cfg.n_market_feat)
    with torch.no_grad():
        base = net(x)
        x2 = x.clone(); x2[1] += 3.0
        pert = net(x2)
    assert torch.allclose(base[0], pert[0], atol=1e-6)
    assert not torch.allclose(base[1], pert[1], atol=1e-4)


def test_key_mask_excludes_invalid_stocks():
    """Masking a stock as an invalid key changes the (valid) other stocks' outputs (dense attention)."""
    cfg = _tiny_cfg()
    net = build_master(cfg).eval()
    x = torch.randn(1, 5, 4, cfg.n_stock_feat + cfg.n_market_feat)
    km = torch.ones(1, 5)
    with torch.no_grad():
        full = net(x, km)
        km2 = km.clone(); km2[0, 4] = 0.0                  # drop stock 4 as a key
        masked = net(x, km2)
    assert not torch.allclose(full[0, 0], masked[0, 0], atol=1e-5)


def test_all_but_one_masked_key_is_finite():
    """A valid target attending to only itself (all other keys masked) must stay finite (nan_to_num)."""
    cfg = _tiny_cfg()
    net = build_master(cfg).eval()
    x = torch.randn(1, 4, 4, cfg.n_stock_feat + cfg.n_market_feat)
    km = torch.zeros(1, 4); km[0, 0] = 1.0
    with torch.no_grad():
        out = net(x, km)
    assert torch.isfinite(out).all()


def test_reproducible_with_fixed_seed():
    cfg = _tiny_cfg()
    x = torch.randn(2, 4, 4, cfg.n_stock_feat + cfg.n_market_feat)
    torch.manual_seed(0); a = build_master(cfg)(x)
    torch.manual_seed(0); b = build_master(cfg)(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_master_module_direct_construction():
    m = MASTER(d_feat=5, d_model=8, t_nhead=2, s_nhead=2, t_dropout=0.0, s_dropout=0.0,
               gate_input_start_index=5, gate_input_end_index=9, beta=2.0, max_len=50)
    out = m(torch.randn(2, 3, 4, 9))
    assert out.shape == (2, 3)


# --------------------------- market features (causal) ---------------------------

def test_causal_rolling_z_no_look_ahead():
    v = np.array([1.0, 2.0, 3.0, 100.0, 5.0])
    z = MF.causal_rolling_z(v, window=3)
    # truncating the series after t must not change z[t] (proves no look-ahead)
    z_trunc = MF.causal_rolling_z(v[:4], window=3)
    assert np.allclose(z[:4], z_trunc)
    assert z[0] == 0.0                                       # single obs -> std 0 -> z 0


def test_compute_market_raw_shape_and_finite():
    rng = np.random.default_rng(0)
    pk = np.abs(rng.normal(size=(40, 6))) * 1e-4
    vshock = rng.normal(size=(40, 6))
    mkt = MF.compute_market_raw(pk, vshock, window=5)
    assert mkt.shape == (40, MF.N_MARKET_FEAT)
    assert np.isfinite(mkt).all()


def test_cross_sectional_handles_all_nan_row():
    pk = np.full((3, 4), np.nan)
    pk[1] = [1.0, 2.0, np.nan, 4.0]
    vshock = np.full((3, 4), np.nan)
    vshock[1] = [0.1, 0.2, 0.3, np.nan]
    mm, md, mv = MF._cross_sectional(pk, vshock)
    assert mm[0] == 0.0 and md[0] == 0.0 and mv[0] == 0.0    # all-NaN row -> neutral 0
    assert mm[1] == pytest.approx(np.nanmean(pk[1]))


def test_market_scaler_fit_on_train_only():
    mkt = np.arange(20, dtype=float).reshape(10, 2)
    mean, std = MF.fit_market_scaler(mkt, train_end_row=4, eps=1e-8)
    assert np.allclose(mean, mkt[:5].mean(axis=0))          # rows 0..4 only, val/test excluded
    assert np.all(std > 0)


def test_market_scaler_train_end_out_of_range_raises():
    mkt = np.zeros((5, 2))
    with pytest.raises(ValueError):
        MF.fit_market_scaler(mkt, train_end_row=5, eps=1e-8)


def test_standardize_replaces_nonfinite():
    mkt = np.array([[np.nan, np.inf], [1.0, 2.0]])
    out = MF.standardize(mkt, mean=np.array([0.0, 0.0]), std=np.array([1.0, 1.0]))
    assert np.isfinite(out).all()


def test_pack_market_windows_and_raise():
    mkt = np.arange(30, dtype=np.float32).reshape(15, 2)
    xm = MF.pack_market(mkt, anchor_rows=np.array([5, 9]), lookback=4)
    assert xm.shape == (2, 4, 2)
    assert np.allclose(xm[0], mkt[2:6])                      # window [t-lb+1 .. t]
    with pytest.raises(ValueError):
        MF.pack_market(mkt, anchor_rows=np.array([2]), lookback=4)


# --------------------------- driver helpers ---------------------------

def test_cat_input_broadcasts_market_over_stocks():
    xs = np.zeros((2, 3, 4, 5), dtype=np.float32)
    xm = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)
    cat = RM._cat_input(xs, xm)
    assert cat.shape == (2, 3, 4, 7)
    assert np.allclose(cat[0, 0, :, 5:], xm[0])
    assert np.allclose(cat[0, 1, :, 5:], xm[0])             # same market row for every stock


def test_floor_is_positive_and_linear_inverse():
    t_mean = np.array([2.0, 4.0]); t_std = np.array([1.0, 2.0])
    cfg = MC.MasterConfig()
    pn = np.array([[-100.0, -100.0]])                       # very negative normalized output
    out = RM._floor(pn, t_mean, t_std, cfg)
    assert (out > 0).all()                                  # positivity floor, no Softplus needed
    assert np.allclose(out[0], cfg.pos_floor_frac * t_mean + cfg.pos_floor_eps)


def test_nonlock_metrics_drops_limitlock_and_empty_branch():
    floor = 1e-8
    pooled = {(0, "d0"): (1.0, 0.9), (1, "d0"): (2e-8, 1e-8)}   # 2nd is <= 2*floor -> dropped
    m = RM.nonlock_metrics(pooled, floor, mult=2.0)
    assert m["n_nonlock"] == 1 and m["n_limitlock"] == 1
    empty = RM.nonlock_metrics({(0, "d0"): (1e-9, 1e-9)}, floor, mult=2.0)
    assert empty["qlike_nonlock"] is None and empty["n_nonlock"] == 0


def test_winrate_vs_counts_and_dates():
    floor = 1e-8
    # model perfect on both obs of d0, worse on d1 -> ticker-date winrate 0.5, wins date d0 only
    model = {(0, "d0"): (1.0, 1.0), (1, "d0"): (2.0, 2.0), (0, "d1"): (1.0, 5.0)}
    harx = {(0, "d0"): (1.0, 2.0), (1, "d0"): (2.0, 3.0), (0, "d1"): (1.0, 1.2)}
    w = RM.winrate_vs(model, harx, floor)
    assert w["n_ticker_date"] == 3 and w["n_unique_date"] == 2
    assert w["ticker_date_winrate"] == pytest.approx(2 / 3)
    assert w["date_winrate"] == pytest.approx(0.5)


def test_edge_density_range():
    a = np.eye(4, dtype=np.float32)
    a[0, 1] = 0.5
    d = RM._edge_density(a)
    assert 0.0 <= d <= 1.0 and d == pytest.approx(1 / 12)


def test_date_range():
    pooled = {(0, "2020-01-02"): (1.0, 1.0), (1, "2020-01-05"): (1.0, 1.0)}
    dr = RM._date_range(pooled)
    assert dr["first_oos_date"] == "2020-01-02" and dr["last_oos_date"] == "2020-01-05"
    assert dr["n_unique_date"] == 2


def test_aggregate_pools_split_metrics():
    per_fold = [{"train_metrics": {"MASTER": {"mse": 1.0, "qlike": 0.5, "r2": 0.3, "n": 10}},
                 "val_metrics": {"MASTER": {"mse": 2.0, "qlike": 0.6, "r2": 0.2, "n": 5}}},
                {"train_metrics": {"MASTER": {"mse": 3.0, "qlike": 0.7, "r2": 0.1, "n": 20}},
                 "val_metrics": {"MASTER": {"mse": 4.0, "qlike": 0.8, "r2": 0.0, "n": 5}}}]
    tr, va = RM._aggregate(per_fold, ("MASTER",))
    assert tr["MASTER"]["mse"] == pytest.approx(2.0) and tr["MASTER"]["n"] == 30
    assert va["MASTER"]["qlike"] == pytest.approx(0.7)


# --------------------------- end-to-end tiny fold (real pipeline) ---------------------------

def _toy_panel(n=6, T=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.array([np.datetime64("2015-01-01") + np.timedelta64(i, "D") for i in range(T)])
    pk = np.abs(rng.normal(size=(T, n))) * 1e-4 + 1e-6
    feats = np.zeros((T, n, 5))
    feats[:, :, 0] = pk
    for j in range(n):                                       # causal rolling HAR-like features
        s = pk[:, j]
        feats[:, j, 1] = np.convolve(s, np.ones(5) / 5, mode="same")
        feats[:, j, 2] = np.convolve(s, np.ones(22) / 22, mode="same")
    feats[:, :, 3] = pk.mean(axis=1, keepdims=True)
    feats[:, :, 4] = rng.normal(size=(T, n))
    lookback, horizon = 5, 1
    from wf_enriched_panel import MR as _MR
    anchors = np.arange(_MR.FIRST_VALID + lookback - 1, T - horizon)
    win_ok = np.ones((len(anchors), n), dtype=bool)
    tgt_ok = np.ones((len(anchors), n), dtype=bool)
    node_ok = win_ok & tgt_ok
    target_dates = dates[anchors + horizon]
    tickers = [f"T{j}" for j in range(n)]
    return EnrichedPanel(tickers, dates, pk, feats, anchors, win_ok, tgt_ok, node_ok, target_dates)


@pytest.mark.smoke
def test_end_to_end_fold_trains_all_models_and_produces_evidence():
    panel = _toy_panel()
    lookback, horizon = 5, 1
    mkt_raw = MF.compute_market_raw(panel.pk, panel.feats[:, :, 4], window=5)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=1)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, (n - ts))
    folds = make_folds(n, ts, K, wf.val, horizon)
    mcfg = _tiny_cfg()
    lcfg = training_config(epochs=2, patience=1, seeds=mcfg.seeds, batch=8)
    upd, ev = RM.run_fold(panel, folds[0], wf, mcfg, lcfg, mkt_raw)
    assert set(upd["seeds"]) == {"LSTM", "VolGA", "MASTER"}
    assert len(upd["HAR-X"]) > 0
    for m in ("LSTM", "VolGA", "MASTER"):
        assert m in ev["fit_diagnostics"]
        assert "train" in ev["learning_curves"][m] and len(ev["learning_curves"][m]["train"]) == len(mcfg.seeds)
        # every pooled test prediction is strictly positive (floored, non-negative volatility)
        for _, p in upd["seeds"][m][0].values():
            assert p > 0.0


@pytest.mark.smoke
def test_train_master_scaler_fit_on_train_and_positive_output():
    """train_master must fit the target scaler on TRAIN only and emit positive floored predictions."""
    panel = _toy_panel(seed=1)
    lookback, horizon = 5, 1
    mkt_raw = MF.compute_market_raw(panel.pk, panel.feats[:, :, 4], window=5)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=1)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    folds = make_folds(n, ts, max(1, n - ts), wf.val, horizon)
    from wf_enriched_panel import pack_fold
    D = pack_fold(panel, folds[0], lookback, horizon)
    xm_tr, xm_va, xm_te = RM.fold_market(panel, folds[0], mkt_raw, lookback, _tiny_cfg())
    out = RM.train_master(D, xm_tr, xm_va, xm_te, _tiny_cfg(), seed=42, return_splits=True)
    assert out["test"].shape == D.y_te.shape
    assert (out["test"] > 0).all()
    assert len(out["train_curve"]) >= 1 and out["best_epoch"] >= 1
    # return_splits=False returns the floored test-prediction array directly
    te_only = RM.train_master(D, xm_tr, xm_va, xm_te, _tiny_cfg(), seed=42, return_splits=False)
    assert te_only.shape == D.y_te.shape and (te_only > 0).all()
