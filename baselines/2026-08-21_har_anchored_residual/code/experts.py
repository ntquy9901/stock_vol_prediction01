"""Data build + unified neural expert trainer for the HAR-anchored ladder (E0-E8).

Design (see design.md): the full ladder runs on the common-date SNAPSHOT design (so the GAT graph is
well-defined and E5/E6/E7 are same-fold comparable), with a target-overlap PURGE of the last ``horizon``
train and val snapshots (fixes leakage F1). HAR is a pooled per-horizon OLS anchor; residual training
targets use expanding-window cross-fitted HAR predictions (no in-sample leakage). One trainer serves all
neural experts via a ``framing`` switch:
  full     -> predict normalized target        (E1 LSTM-only, E2 LSTM+GAT)
  additive -> predict normalized HAR residual  (E5 LSTM, E6 GAT, E7 LSTM+GAT)
  mult     -> predict normalized log-residual  (E8 multiplicative, HAR-anchored, positive)
Early stopping is on validation QLIKE of the reconstructed prediction (the predeclared primary metric).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B  # noqa: E402
import edges as EG  # noqa: E402
import metrics as M  # noqa: E402
from snapshots import build_snapshots  # noqa: E402

import models as md  # noqa: E402


def _purge_snapshots(snap, horizon):
    """Drop the last ``horizon`` train and val snapshots (their target date lands in the next split)."""
    if horizon < len(snap.train):
        snap.train = snap.train[:-horizon]
    if horizon < len(snap.val):
        snap.val = snap.val[:-horizon]
    return snap


def _stack(snaps, key):
    return np.stack([s[key] for s in snaps])           # [n, ...]


def _crossfit_har_pooled(har_tr, y_tr, n_folds, floor):
    """Expanding-window pooled OOS HAR preds over train date-blocks. har_tr[n,N,3], y_tr[n,N]."""
    n = len(har_tr)
    oos = np.full_like(y_tr, np.nan, dtype=np.float64)
    blocks = np.array_split(np.arange(n), n_folds)
    for k in range(1, len(blocks)):
        fit = np.concatenate(blocks[:k])
        coef = B.har_fit(har_tr[fit].reshape(-1, 3), y_tr[fit].reshape(-1))
        pr = B.har_predict(har_tr[blocks[k]].reshape(-1, 3), coef, floor=floor)
        oos[blocks[k]] = pr.reshape(len(blocks[k]), -1)
    return oos                                          # NaN on first block


def build_data(files, lookback, horizon, cfg, n_cvfolds=5, eps=1e-8, min_common=300):
    snap = _purge_snapshots(build_snapshots(files, lookback, horizon,
                                            cfg.train_frac, cfg.val_frac, min_common=min_common), horizon)
    N = snap.num_nodes
    adj = EG.glasso_adjacency(snap.adj_pk_train, top_k=cfg.top_k)
    adj_t = torch.from_numpy(np.asarray(adj, dtype=np.float32))

    X_tr, X_va, X_te = _stack(snap.train, "x"), _stack(snap.val, "x"), _stack(snap.test, "x")
    y_tr = _stack(snap.train, "y_raw"); y_va = _stack(snap.val, "y_raw"); y_te = _stack(snap.test, "y_raw")
    har_tr = _stack(snap.train, "har_raw"); har_va = _stack(snap.val, "har_raw"); har_te = _stack(snap.test, "har_raw")
    d_va = [s["date"] for s in snap.val]; d_te = [s["date"] for s in snap.test]

    # pooled full-train HAR anchor (deployment forecast on val/test)
    coef = B.har_fit(har_tr.reshape(-1, 3), y_tr.reshape(-1))
    harp_tr = B.har_predict(har_tr.reshape(-1, 3), coef, floor=cfg.qlike_floor).reshape(y_tr.shape)
    harp_va = B.har_predict(har_va.reshape(-1, 3), coef, floor=cfg.qlike_floor).reshape(y_va.shape)
    harp_te = B.har_predict(har_te.reshape(-1, 3), coef, floor=cfg.qlike_floor).reshape(y_te.shape)

    # cross-fitted OOS HAR on train -> residual targets (drop first fold with NaN)
    oos = _crossfit_har_pooled(har_tr, y_tr, n_cvfolds, cfg.qlike_floor)
    mask = np.isfinite(oos).all(1)                      # keep dates where every node has an OOS pred
    r_add = y_tr - oos                                  # additive residual [n,N]
    r_mul = np.log((y_tr + eps)) - np.log((oos + eps))  # log residual
    add_scale = np.nanstd(r_add[mask], axis=0) + 1e-8   # per-node [N]
    mul_scale = np.nanstd(r_mul[mask], axis=0) + 1e-8

    # full-target per-node standardization (from train targets)
    t_mean = y_tr.mean(0); t_std = y_tr.std(0) + 1e-8
    # train-derived POSITIVE output floor (plan section 10): the additive residual can drive a prediction
    # below zero; clipping to 0 then makes QLIKE (y/pred) blow up through the metric floor. Flooring the
    # reconstruction at a small fraction of each node's mean train variance is a positive output mapping
    # bounded from training data only; it is far below any normal HAR-scale prediction (so it changes
    # nothing on panels where predictions stay sane) and only bounds pathological near-zero corrections.
    pred_floor = np.maximum(1e-3 * t_mean, cfg.qlike_floor)

    return {
        "N": N, "adj": adj_t, "horizon": horizon, "eps": eps, "pred_floor": pred_floor,
        "X_tr": X_tr.astype(np.float32), "X_va": X_va.astype(np.float32), "X_te": X_te.astype(np.float32),
        "y_tr": y_tr, "y_va": y_va, "y_te": y_te,
        "harp_tr": harp_tr, "harp_va": harp_va, "harp_te": harp_te,
        "oos_add": r_add, "oos_mul": r_mul, "cf_mask": mask,
        "add_scale": add_scale, "mul_scale": mul_scale, "t_mean": t_mean, "t_std": t_std,
        "d_va": d_va, "d_te": d_te, "tickers": snap.tickers,
        "coef": coef,
    }


def har_pred_dict(D, split):
    """E0 HAR prediction dict {(node,date):(y_raw,pred_raw)} on val or test."""
    sfx = "te" if split == "test" else "va"
    y = D[f"y_{sfx}"]; hp = D[f"harp_{sfx}"]; dates = D[f"d_{sfx}"]
    return {(j, dates[i]): (float(y[i, j]), float(hp[i, j]))
            for i in range(len(dates)) for j in range(D["N"])}


def _reconstruct(framing, D, c, harp):
    """c [n,N] normalized net output -> raw prediction [n,N], floored at the train-derived pred_floor."""
    if framing == "full":
        pred = c * D["t_std"] + D["t_mean"]
    elif framing == "additive":
        pred = md.additive_pred(harp, c, res_scale=D["add_scale"], lam=1.0)
    else:
        pred = md.multiplicative_pred(harp, c, eps=D["eps"], lam=1.0, res_scale=D["mul_scale"])
    return np.maximum(pred, D["pred_floor"])


def train_neural(D, cfg, seed, use_lstm, use_graph, framing):
    """Train one expert; return (test pred dict, val pred dict, best_val_qlike, param_count)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = md.ResidualNet(hidden=cfg.hidden, heads=cfg.heads, dropout=cfg.dropout,
                         use_lstm=use_lstm, use_graph=use_graph).to(device)
    adj = D["adj"].to(device)

    # training target (normalized) + row mask
    if framing == "full":
        tgt = (D["y_tr"] - D["t_mean"]) / D["t_std"]; rows = np.ones(len(tgt), bool)
    elif framing == "additive":
        tgt = D["oos_add"] / D["add_scale"]; rows = D["cf_mask"]
    else:
        tgt = D["oos_mul"] / D["mul_scale"]; rows = D["cf_mask"]
    Xtr = torch.from_numpy(D["X_tr"][rows]).to(device)
    Ttr = torch.from_numpy(tgt[rows].astype(np.float32)).to(device)
    Xva = torch.from_numpy(D["X_va"]).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lossf = nn.MSELoss(); n = len(Xtr); bs = cfg.batch_size
    best = np.inf; best_state = None; wait = 0
    for ep in range(cfg.epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); out = net(Xtr[idx], adj); loss = lossf(out, Ttr[idx])
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        # val QLIKE of reconstructed prediction (predeclared primary metric)
        net.eval()
        with torch.no_grad():
            cva = net(Xva, adj).cpu().numpy()
        pva = np.maximum(_reconstruct(framing, D, cva, D["harp_va"]), 0.0)
        vq = M.qlike(D["y_va"].reshape(-1), pva.reshape(-1), floor=cfg.qlike_floor)
        sched.step(vq)
        if vq < best - 1e-9:
            best = vq; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}; wait = 0
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        cte = net(torch.from_numpy(D["X_te"]).to(device), adj).cpu().numpy()
        cva = net(Xva, adj).cpu().numpy()
        ctr = net(torch.from_numpy(D["X_tr"]).to(device), adj).cpu().numpy()
    pte = np.maximum(_reconstruct(framing, D, cte, D["harp_te"]), 0.0)
    pva = np.maximum(_reconstruct(framing, D, cva, D["harp_va"]), 0.0)
    te = {(j, D["d_te"][i]): (float(D["y_te"][i, j]), float(pte[i, j]))
          for i in range(len(D["d_te"])) for j in range(D["N"])}
    va = {(j, D["d_va"][i]): (float(D["y_va"][i, j]), float(pva[i, j]))
          for i in range(len(D["d_va"])) for j in range(D["N"])}
    corr = {"c_tr": ctr, "c_va": cva, "c_te": cte, "framing": framing}
    return te, va, float(best), sum(p.numel() for p in net.parameters()), corr
