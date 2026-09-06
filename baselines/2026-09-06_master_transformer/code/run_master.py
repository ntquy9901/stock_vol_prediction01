"""Walk-forward evaluation of MASTER vs HAR-X / LSTM / VolGA for daily Parkinson-variance forecasting.

Each expanding-window fold refits HAR/HAR-X (OLS), a no-graph LSTM, VolGA (LSTM + sparse horizon-matched
vol->PK GAT) and MASTER (market-guided dense-attention transformer) on the SAME train window with the
SAME per-node train-only scalers and the SAME shared QLIKE positivity floor, then forecasts the next K
days. Predictions are pooled over the whole OOS region and compared with date-clustered Diebold-Mariano
tests (MASTER vs HAR-X / LSTM / VolGA). All the fold/panel/metric/DM/evidence machinery is reused
READ-ONLY from the delivered baselines; only the MASTER module + market features + this driver are new.

Run:   .venv_gpu_encode/Scripts/python.exe baselines/2026-09-06_master_transformer/code/run_master.py --market vn30 --horizon 1
Smoke: ... --smoke   (1 fold, 2 epochs, 2 seeds)
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-09-05_edge_horizon_matched" / "code",
           REPO / "submission" / "soict_lstm_gat",
           REPO / "scripts" / "quality_gate"):
    sys.path.insert(0, str(_p))

import masked_rich as MR  # noqa: E402
import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import _har_ols_preds, training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_edge_hmatched import directed_vol2pk_hmatched  # noqa: E402

import market_features as MF  # noqa: E402
import master_config as MC  # noqa: E402
from master_net import build_master  # noqa: E402

_MODELS = ("HAR", "HAR-X", "LSTM", "VolGA", "MASTER")   # MASTER name -> flagged learned by overfit gate


# ----------------------------- MASTER training (mirrors train_masked_rich zscore_floor) -----------------------------

def _cat_input(xs: np.ndarray, xm: np.ndarray) -> np.ndarray:
    """Concatenate per-node stock features ``[A,N,T,5]`` with broadcast market features ``[A,T,M]``.

    Market features are the same for every stock at a given (anchor, timestep) -> broadcast over N.
    """
    a, n, t, _ = xs.shape
    xm_b = np.broadcast_to(xm[:, None, :, :], (a, n, t, xm.shape[-1]))
    return np.concatenate([xs.astype(np.float32), xm_b.astype(np.float32)], axis=-1)


def _floor(pn: np.ndarray, t_mean: np.ndarray, t_std: np.ndarray, cfg) -> np.ndarray:
    """Inverse per-node z-score then apply the shared positivity floor (identical basis to LSTM/VolGA)."""
    return np.maximum(pn * t_std + t_mean, cfg.pos_floor_frac * t_mean + cfg.pos_floor_eps)


def train_master(D, xm_tr: np.ndarray, xm_va: np.ndarray, xm_te: np.ndarray, cfg, seed: int,
                 return_splits: bool = False):
    """Train MASTER on one fold: per-node z-score target, LINEAR output, inverse-transform + positivity
    floor at eval (no Softplus/ReLU). Early-stop on masked val MSE. Returns floored test predictions
    ``[A,N]`` (and, if ``return_splits``, val/train predictions + per-epoch learning curves)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_master(cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    t_mean, t_std = D.t_mean, D.t_std
    tmean = torch.from_numpy(t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(t_std.astype(np.float32)).to(dev)
    # Build the concatenated [A,N,T,D] inputs + key masks ONCE per split (not per epoch): the market
    # broadcast + concatenate is a large host allocation; rebuilding it every epoch/inference call was the
    # bottleneck. Tensors live on the GPU for the whole fold; only minibatch views are indexed in the loops.
    xtr = torch.from_numpy(_cat_input(D.X_tr, xm_tr)).to(dev)
    xva = torch.from_numpy(_cat_input(D.X_va, xm_va)).to(dev)
    xte = torch.from_numpy(_cat_input(D.X_te, xm_te)).to(dev)
    kmtr = torch.from_numpy(D.nmask_tr.astype(np.float32)).to(dev)
    kmva = torch.from_numpy(D.nmask_va.astype(np.float32)).to(dev)
    kmte = torch.from_numpy(D.nmask_te.astype(np.float32)).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr.astype(np.float32)).to(dev)
    ytr_n = (torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd
    bs = cfg.batch_size

    def infer(xt, km):                                       # xt, km already on device (prebuilt per split)
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(xt), bs):
                outs.append(net(xt[i:i + bs], km[i:i + bs]).cpu().numpy())
        return _floor(np.concatenate(outs), t_mean, t_std, cfg)

    best = np.inf
    best_state = None
    wait = 0
    best_ep = 0
    train_curve = []
    val_curve = []
    rng = np.random.default_rng(seed)
    for ep in range(cfg.epochs):
        net.train()
        idx = rng.permutation(len(xtr))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb, kmb, tmb, yb = xtr[b], kmtr[b], tmtr[b], ytr_n[b]
            opt.zero_grad()
            pred = net(xb, kmb)
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        pva = infer(xva, kmva)
        mva = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[mva] - D.y_va[mva]) ** 2))
        ptr = infer(xtr, kmtr)
        mtr = D.tmask_tr.astype(bool)
        train_curve.append(float(np.mean((ptr[mtr] - D.y_tr[mtr]) ** 2)))
        val_curve.append(vmse)
        sched.step(vmse)
        if vmse < best - 1e-12:
            best = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
            best_ep = ep + 1
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    te = infer(xte, kmte)
    if return_splits:
        return {"test": te, "val": infer(xva, kmva), "train": infer(xtr, kmtr),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te


# ----------------------------- market features per fold -----------------------------

def fold_market(panel, fold, mkt_raw: np.ndarray, lookback: int, cfg):
    """Per-fold TRAIN-only-standardized market windows for the train/val/test anchors."""
    train_end = int(panel.anchors[fold.train][-1])
    mean, std = MF.fit_market_scaler(mkt_raw, train_end, cfg.scaler_eps)
    mkt_std = MF.standardize(mkt_raw, mean, std)
    return (MF.pack_market(mkt_std, panel.anchors[fold.train], lookback),
            MF.pack_market(mkt_std, panel.anchors[fold.val], lookback),
            MF.pack_market(mkt_std, panel.anchors[fold.forecast], lookback))


# ----------------------------- evaluation helpers -----------------------------

def _pool(o, D):
    return RMR._pred_dict(o, D.y_te, D.tmask_te, D.d_te, D.N)


def nonlock_metrics(pooled: dict, floor: float, mult: float) -> dict:
    """Conditional QLIKE after dropping limit-lock / zero-range obs (target <= mult*floor), where the
    floored ~0 target makes QLIKE's y/f ratio explode. ``pooled`` maps key -> (y_true, y_pred)."""
    kept = {k: v for k, v in pooled.items() if v[0] > mult * floor}
    if not kept:
        return {"qlike_nonlock": None, "n_nonlock": 0, "n_limitlock": len(pooled)}
    return {"qlike_nonlock": RMR._metrics(kept, floor)["qlike"], "n_nonlock": len(kept),
            "n_limitlock": len(pooled) - len(kept)}


def winrate_vs(model_pool: dict, harx_pool: dict, floor: float) -> dict:
    """Ticker-date and unique-date win-rate of a model's per-obs QLIKE vs HAR-X on the shared obs."""
    keys = sorted(set(model_pool) & set(harx_pool))
    y = np.array([model_pool[k][0] for k in keys])
    pm = np.array([model_pool[k][1] for k in keys])
    ph = np.array([harx_pool[k][1] for k in keys])
    qm = MR_perobs(y, pm, floor)
    qh = MR_perobs(y, ph, floor)
    wins = qm < qh
    dates = np.array([k[1] for k in keys])
    uniq = sorted(set(dates))
    date_wins = 0
    for d in uniq:
        sel = dates == d
        if qm[sel].mean() < qh[sel].mean():
            date_wins += 1
    return {"ticker_date_winrate": float(wins.mean()), "n_ticker_date": int(len(keys)),
            "date_winrate": float(date_wins / len(uniq)), "n_unique_date": int(len(uniq))}


def MR_perobs(y, p, floor):
    """Per-observation QLIKE via the delivered metrics module (shared floor / clamp)."""
    import metrics as M
    return M.per_obs_qlike(y, p, floor)


def _date_range(pooled: dict) -> dict:
    dates = sorted({k[1] for k in pooled})
    return {"first_oos_date": dates[0], "last_oos_date": dates[-1], "n_unique_date": len(dates)}


def _git_commit() -> str:  # pragma: no cover - environment glue
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _versions() -> dict:  # pragma: no cover - environment glue
    return {"torch": torch.__version__, "numpy": np.__version__,
            "device": ("cuda" if torch.cuda.is_available() else "cpu"),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}


# ----------------------------- one fold -----------------------------

def run_fold(panel, fold, wf, mcfg, lcfg, mkt_raw):
    """Train HAR/HAR-X/LSTM/VolGA/MASTER on one fold; return pooled test dicts + per-fold fit evidence."""
    D = pack_fold(panel, fold, wf.lookback, wf.horizon)
    fl = lcfg.qlike_floor
    nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
    har, harx = _har_ols_preds(D, fl, nfloor)
    eye = np.eye(D.N, dtype=np.float32)
    last_tr_row = int(panel.anchors[fold.train][-1]) + wf.horizon
    adj = directed_vol2pk_hmatched(panel.feats[:, :, 4], np.sqrt(panel.pk), last_tr_row, wf.horizon, MR.EDGE_TOP_K)
    xm_tr, xm_va, xm_te = fold_market(panel, fold, mkt_raw, wf.lookback, mcfg)

    lstm_outs = [RMR.train_masked_rich(D, lcfg, s, False, eye, return_splits=True) for s in lcfg.seeds]
    volga_outs = [RMR.train_masked_rich(D, lcfg, s, True, adj, return_splits=True) for s in lcfg.seeds]
    master_outs = [train_master(D, xm_tr, xm_va, xm_te, mcfg, s, return_splits=True) for s in mcfg.seeds]

    seed_te = {
        "LSTM": [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in lstm_outs],
        "VolGA": [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in volga_outs],
        "MASTER": [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in master_outs],
    }
    upd = {"HAR": _pool(har["te"], D), "HAR-X": _pool(harx["te"], D), "seeds": seed_te}

    outs = {"LSTM": lstm_outs, "VolGA": volga_outs, "MASTER": master_outs}
    tr_pred = {"HAR": har["tr"], "HAR-X": harx["tr"]}
    va_pred = {"HAR": har["va"], "HAR-X": harx["va"]}
    te_pred = {"HAR": har["te"], "HAR-X": harx["te"]}
    for m in ("LSTM", "VolGA", "MASTER"):
        tr_pred[m] = RMR._ens_split(outs[m], "train")
        va_pred[m] = RMR._ens_split(outs[m], "val")
        te_pred[m] = RMR._ens_split(outs[m], "test")
    train_m = {m: RMR._split_metrics(tr_pred[m], D.y_tr, D.tmask_tr, fl) for m in _MODELS}
    val_m = {m: RMR._split_metrics(va_pred[m], D.y_va, D.tmask_va, fl) for m in _MODELS}
    test_m = {m: RMR._split_metrics(te_pred[m], D.y_te, D.tmask_te, fl) for m in _MODELS}
    fit = {m: RMR.OF.classify_fit(train_m[m], val_m[m], test_m[m]) for m in _MODELS}
    curves = {m: {"train": [o["train_curve"] for o in outs[m]], "val": [o["val_curve"] for o in outs[m]],
                  "best_epoch": [o["best_epoch"] for o in outs[m]]} for m in ("LSTM", "VolGA", "MASTER")}
    evidence = {"idx": fold.idx, "n_train": int(D.tmask_tr.sum()), "n_val": int(D.tmask_va.sum()),
                "n_forecast": int(D.tmask_te.sum()), "edge_density": _edge_density(adj),
                "train_metrics": train_m, "val_metrics": val_m, "test_metrics": test_m,
                "fit_diagnostics": fit, "learning_curves": curves}
    return upd, evidence


def _edge_density(a: np.ndarray) -> float:
    n = a.shape[0]
    off = a.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.count_nonzero(off)) / (n * (n - 1))


def _aggregate(per_fold, learned):
    """Roll per-fold seed-ensembled train/val split metrics into pooled means (over/under-fit evidence)."""
    def agg(block, m):
        ds = [f[block][m] for f in per_fold]
        keys = ("mse", "qlike", "r2")
        out = {k: float(np.mean([d[k] for d in ds])) for k in keys}
        out["n"] = int(sum(d["n"] for d in ds))
        return out
    train_metrics = {m: agg("train_metrics", m) for m in learned}
    val_metrics = {m: agg("val_metrics", m) for m in learned}
    return train_metrics, val_metrics


# ----------------------------- pooled run -----------------------------

def run(market, horizon, mcfg, lcfg, folds_target, lookback, out=None):  # pragma: no cover - driver glue
    t0 = time.time()
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    mkt_raw = MF.compute_market_raw(panel.pk, panel.feats[:, :, 4], mcfg.market_z_window)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)

    pooled = {"HAR": {}, "HAR-X": {}}
    seed_pool = {m: [{} for _ in mcfg.seeds] for m in ("LSTM", "VolGA", "MASTER")}
    per_fold = []
    print(f"[master] {market} h{horizon}: {panel.N} nodes, {len(folds)} folds, K={K}, "
          f"{len(mcfg.seeds)} seeds, lb{lookback}", flush=True)
    for fi, fold in enumerate(folds):
        ft0 = time.time()
        upd, ev = run_fold(panel, fold, wf, mcfg, lcfg, mkt_raw)
        pooled["HAR"].update(upd["HAR"])
        pooled["HAR-X"].update(upd["HAR-X"])
        for m in ("LSTM", "VolGA", "MASTER"):
            for si, d in enumerate(upd["seeds"][m]):
                seed_pool[m][si].update(d)
        per_fold.append(ev)
        print(f"[master] fold {fi + 1}/{len(folds)} done in {time.time() - ft0:.0f}s "
              f"(fit MASTER={ev['fit_diagnostics']['MASTER']['status']}, "
              f"elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)

    fl = lcfg.qlike_floor
    ens = {m: RMR._ens(seed_pool[m]) for m in ("LSTM", "VolGA", "MASTER")}
    preds = {"HAR": pooled["HAR"], "HAR-X": pooled["HAR-X"], **ens}
    metrics = {m: {**RMR._metrics(preds[m], fl), **nonlock_metrics(preds[m], fl, mcfg.limit_lock_mult),
                   **winrate_vs(preds[m], pooled["HAR-X"], fl)} for m in _MODELS}
    per_seed = {m: RMR.seed_metric_stats(seed_pool[m], fl) for m in ("LSTM", "VolGA", "MASTER")}
    train_metrics, val_metrics = _aggregate(per_fold, ("LSTM", "VolGA", "MASTER"))
    fit_diagnostics = {m: RMR.OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m])
                       for m in ("LSTM", "VolGA", "MASTER")}
    dm = {"MASTER_vs_HARX": RMR._dm_all(ens["MASTER"], pooled["HAR-X"], horizon, fl),
          "MASTER_vs_LSTM": RMR._dm_all(ens["MASTER"], ens["LSTM"], horizon, fl),
          "MASTER_vs_VolGA": RMR._dm_all(ens["MASTER"], ens["VolGA"], horizon, fl),
          "VolGA_vs_HARX": RMR._dm_all(ens["VolGA"], pooled["HAR-X"], horizon, fl),
          "LSTM_vs_HARX": RMR._dm_all(ens["LSTM"], pooled["HAR-X"], horizon, fl)}
    result = {
        "experiment": "master_transformer", "model": "MASTER (arXiv:2312.15235, adapted for variance)",
        "market": market, "horizon": horizon, "num_nodes": int(panel.N), "n_folds": len(folds),
        "retrain_cadence_K": K, "folds_target": folds_target, "lookback": lookback,
        "seeds": list(mcfg.seeds), "n_test_obs": metrics["HAR"]["n"],
        "date_range": _date_range(pooled["HAR"]),
        "git_commit": _git_commit(), "versions": _versions(),
        "master_config": {"d_model": mcfg.d_model, "t_nhead": mcfg.t_nhead, "s_nhead": mcfg.s_nhead,
                          "dropout": mcfg.dropout, "beta": mcfg.beta, "lr": mcfg.lr, "epochs": mcfg.epochs,
                          "patience": mcfg.patience, "batch_size": mcfg.batch_size,
                          "n_market_feat": mcfg.n_market_feat},
        "baseline_config": {"epochs": lcfg.epochs, "patience": lcfg.patience, "batch_size": lcfg.batch_size,
                            "qlike_floor": fl, "edge_top_k": MR.EDGE_TOP_K},
        "metrics": metrics, "metrics_per_seed": per_seed, "dm_date_clustered": dm,
        "train_metrics": train_metrics, "val_metrics": val_metrics, "fit_diagnostics": fit_diagnostics,
        "learning_curves": {m: [f["learning_curves"][m] for f in per_fold] for m in ("LSTM", "VolGA", "MASTER")},
        "edge_density_mean": float(np.mean([f["edge_density"] for f in per_fold])),
        "per_fold": [{k: f[k] for k in ("idx", "n_train", "n_val", "n_forecast", "edge_density",
                                        "fit_diagnostics")} for f in per_fold],
        "seconds": round(time.time() - t0, 1)}
    print(f"[master] QLIKE h{horizon}: " + ", ".join(f"{m}={metrics[m]['qlike']:.4f}" for m in _MODELS), flush=True)
    print("[master] non-lock QLIKE: " + ", ".join(
        f"{m}={metrics[m]['qlike_nonlock']:.4f}" for m in _MODELS if metrics[m]['qlike_nonlock'] is not None), flush=True)
    for name, dd in dm.items():
        print(f"[master] DM {name} qlike p={dd['qlike'].get('p_value')} favors={dd['qlike'].get('favors')}", flush=True)
    out = Path(out) if out else REPO / "results" / "master_transformer" / f"master_{market}_h{horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"[master] wrote {out} ({result['seconds']}s)", flush=True)
    return result


def main():  # pragma: no cover - entry driver (GPU run)
    ap = argparse.ArgumentParser(description="MASTER walk-forward vs HAR-X/LSTM/VolGA (Parkinson variance).")
    ap.add_argument("--market", default="vn30", choices=["vn30", "vn100"])
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=MC.FOLDS_TARGET)
    ap.add_argument("--lookback", type=int, default=MC.LOOKBACK)
    ap.add_argument("--n-seeds", type=int, default=len(MC.SEEDS))
    ap.add_argument("--batch", type=int, default=MC.BATCH_SIZE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from dataclasses import replace
    if a.smoke:
        mcfg = MC.SMOKE
        lcfg = training_config(epochs=2, patience=1, seeds=mcfg.seeds, batch=32)
        ft = 1
    else:
        seeds = MC.SEEDS[:a.n_seeds]
        mcfg = replace(MC.MasterConfig(), seeds=seeds, batch_size=a.batch)
        lcfg = training_config(epochs=MC.EPOCHS, patience=MC.PATIENCE, seeds=seeds, batch=32)
        ft = a.folds_target
    run(a.market, a.horizon, mcfg, lcfg, ft, a.lookback, a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
