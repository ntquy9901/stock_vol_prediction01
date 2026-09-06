"""Configurable deep trainer: loss in {mse, qlike} x anchor in {none, harx}.

Vendors the delivered MaskedRichNet + scaling/early-stop scaffolding (run_masked_rich) and adds two knobs to
test whether they let the deep model beat HAR-X on QLIKE and what it costs the squared/absolute metrics:

- loss=mse  : masked MSE (the delivered objective).
- loss=qlike: masked QLIKE (y/f - log(y/f) - 1) on the floored, positive-scale forecast (penalizes the
  spike under-forecast that MSE-training leaves — see docs/reports/2026-09-07_sp500_issues_master.md).
- anchor=none: forecast = clamp(z_out * std + mean, floor)   (predict the level, as delivered).
- anchor=harx: forecast = HAR-X * exp(clip(z_out, -c, c))     (predict a bounded residual on top of HAR-X, so
  a spike day cannot collapse far below HAR-X; z=0 falls back to HAR-X). Train residual uses OOF HAR-X.

Pure forecast/loss math is factored into tested helpers; ``train_deep`` (GPU loop) is smoke-tested via the
baseline run, mirroring the delivered ``train_masked_rich`` which is likewise not unit-tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[3]
for _p in (HERE, _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           _REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import pipeline_config as pc  # noqa: E402


def node_floor(t_mean):
    """Shared per-node positivity floor = POS_FLOOR_FRAC*train_mean + POS_FLOOR_EPS (identical to delivered)."""
    return pc.POS_FLOOR_FRAC * np.asarray(t_mean, dtype=float) + pc.POS_FLOOR_EPS


def zscore_forecast(z_out, t_mean, t_std, floor):
    """anchor=none positive forecast: max(z_out*std + mean, floor). Arrays broadcast over [A,N]."""
    return np.maximum(np.asarray(z_out) * t_std + t_mean, floor)


def anchor_forecast(z_out, harx, clip, floor=None):
    """anchor=harx positive forecast: HAR-X * exp(clip(z_out, -clip, clip)); optional positivity floor.
    z=0 -> exactly HAR-X (no collapse); the clip bounds the correction so a spike cannot fall far below."""
    f = np.asarray(harx, dtype=float) * np.exp(np.clip(np.asarray(z_out, dtype=float), -clip, clip))
    return f if floor is None else np.maximum(f, floor)


def qlike_np(y, f, floor=1e-12):
    """Elementwise QLIKE = y/f - log(y/f) - 1 with a positivity floor on y and f (matches RMR._metrics)."""
    y = np.maximum(np.asarray(y, dtype=float), floor)
    f = np.maximum(np.asarray(f, dtype=float), floor)
    r = y / f
    return r - np.log(r) - 1.0


def residual_target(y, harx, eps=1e-12):
    """anchor=harx MSE training target: z_true = log((y+eps)/(harx+eps))."""
    return np.log((np.maximum(np.asarray(y, dtype=float), 0.0) + eps) / (np.asarray(harx, dtype=float) + eps))


def split_objective(y, f, mask, loss, floor):
    """Masked value of the TRAINING objective on one split: mean QLIKE (loss='qlike') or mean MSE else,
    over cells where ``mask`` is true. Used for early-stopping AND the learning curve so both reflect the
    same objective and the same QLIKE floor as the final metric (fixes floor/curve inconsistencies)."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return float("nan")
    yv = np.asarray(y)[m]; fv = np.asarray(f)[m]
    return float(qlike_np(yv, fv, floor).mean()) if loss == "qlike" else float(np.mean((fv - yv) ** 2))


def train_deep(D, cfg, seed, use_graph, adj, loss, anchor, harx=None, clip=0.5, return_splits=True):
    """Train MaskedRichNet under (loss, anchor). ``harx`` = dict with 'tr'/'va'/'te' positive HAR-X forecasts
    [A,N] (OOF for 'tr'); required when anchor=='harx'. Returns floored positive predictions per split + curves.
    Early-stops on the validation value of the training loss."""
    import torch
    import torch.nn as nn
    import run_masked_rich as RMR

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = RMR.MaskedRichNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph).to(dev)
    base = torch.from_numpy(adj).to(dev)
    f32 = np.float32
    tmean = torch.from_numpy(D.t_mean.astype(f32)).to(dev); tstd = torch.from_numpy(D.t_std.astype(f32)).to(dev)
    floor_np = node_floor(D.t_mean); floor = torch.from_numpy(floor_np.astype(f32)).to(dev)
    fl = cfg.qlike_floor                                    # shared QLIKE floor: train == early-stop == eval
    assert D.tmask_va.astype(bool).any(), "empty validation mask (cannot early-stop)"   # fail loud
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    ytr = torch.from_numpy(D.y_tr.astype(f32)).to(dev)
    ytr_z = (ytr - tmean) / tstd
    # anchor=harx: harx['tr']=OOF (NaN warm-up) is the leakage-safe residual base for TRAINING; warm-up cells
    # (no OOF) are masked out of the loss so the residual target is never the in-sample fit. harx['tr_base']
    # (in-sample, finite) is the forecast base for TRAIN inference/evidence only.
    if anchor == "harx":
        oof_np = np.asarray(harx["tr"], dtype=f32); oof_ok = np.isfinite(oof_np)
        hx_tr = torch.from_numpy(np.where(oof_ok, oof_np, 1.0)).to(dev)          # NaN -> 1 (masked out anyway)
        valid_tr = torch.from_numpy(oof_ok.astype(f32)).to(dev)
        ztr_true = torch.log((ytr.clamp(min=0) + 1e-12) / (hx_tr + 1e-12))
    else:
        hx_tr = valid_tr = ztr_true = None
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bs = cfg.batch_size

    def adj_batch(nm):
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def fpos(pn, hxb):                                   # positive forecast in torch (for qlike loss)
        if anchor == "harx":
            return torch.maximum(hxb * torch.exp(pn.clamp(-clip, clip)), floor)
        return torch.maximum(pn * tstd + tmean, floor)

    def batch_loss(pn, yb, ybz, tmb, hxb, ztrue):
        if loss == "qlike":
            f = fpos(pn, hxb); r = yb.clamp(min=fl) / f      # same QLIKE floor as early-stop + final metric
            per = r - torch.log(r) - 1.0
        elif anchor == "harx":
            per = (pn.clamp(-clip, clip) - ztrue) ** 2
        else:
            per = (pn - ybz) ** 2
        return (per * tmb).sum() / tmb.sum().clamp(min=1)

    def infer(sp):
        X = getattr(D, f"X_{sp}"); nm = getattr(D, f"nmask_{sp}")
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs]).to(dev); nmb = torch.from_numpy(nm[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs)
        if anchor == "harx":
            base_hx = harx["tr_base"] if sp == "tr" else harx[sp]   # train base = in-sample HAR-X (evidence only)
            return anchor_forecast(pn, base_hx, clip, floor_np)
        return zscore_forecast(pn, D.t_mean, D.t_std, floor_np)

    def val_score(pva):                                 # early-stop == training objective, same QLIKE floor
        return split_objective(D.y_va, pva, D.tmask_va, loss, fl)

    best = np.inf; best_state = None; wait = 0; best_ep = 0; train_curve = []; val_curve = []
    for ep in range(cfg.epochs):
        net.train()
        for idx in RMR._batches(len(Xtr), bs, True, seed + ep):
            pn = net(Xtr[idx], adj_batch(nmtr[idx]))
            tmb = tmtr[idx] * valid_tr[idx] if anchor == "harx" else tmtr[idx]   # warm-up (no OOF) excluded
            l = batch_loss(pn, ytr[idx], ytr_z[idx], tmb,
                           hx_tr[idx] if anchor == "harx" else None,
                           ztr_true[idx] if anchor == "harx" else None)
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer("va"); vs = val_score(pva)
        # in-sample-base train fit-evidence over ALL train cells; for anchor=harx this is NOT the OOF training
        # objective (loss uses OOF-valid cells only) -- see result 'train_metrics_note'. val_curve IS the objective.
        train_curve.append(split_objective(D.y_tr, infer("tr"), D.tmask_tr, loss, fl))
        val_curve.append(vs)                             # val objective (== early-stop, matches the training loss)
        sched.step(vs)
        if vs < best - 1e-12:
            best = vs; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}; wait = 0; best_ep = ep + 1
        else:
            wait += 1                                        # pragma: no cover - no-improve epoch (nondeterministic)
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break                                            # pragma: no cover - early-stop trigger (nondeterministic)
    if best_state:                                           # pragma: no cover - always set after >=1 finite-val epoch
        net.load_state_dict(best_state)
    te = infer("te")
    if return_splits:
        return {"test": te, "val": infer("va"), "train": infer("tr"),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te
