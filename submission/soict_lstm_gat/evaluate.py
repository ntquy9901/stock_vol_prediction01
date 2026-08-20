"""Test-set prediction + metrics + Diebold-Mariano for the snapshot graph models and baselines."""
from __future__ import annotations

import numpy as np
import torch

import metrics as M


def predict_test(train_result, snap) -> dict:
    """Run a trained model over the test snapshots -> {(ticker_id, date): (y_raw, pred_raw)}.

    Predictions are inverse-transformed by each node's per-ticker scaler and floored (>= qlike_floor
    handled by the caller's metrics floor; here we clip to >= 0)."""
    model = train_result["model"]; adj_t = train_result["adj_t"]; device = train_result["device"]
    model.eval()
    t_mean = np.array([snap.scalers[j][0] for j in range(snap.num_nodes)])
    t_std = np.array([snap.scalers[j][1] for j in range(snap.num_nodes)])
    out = {}
    with torch.no_grad():
        for s in snap.test:
            x = torch.from_numpy(s["x"][None]).float().to(device)   # [1,N,seq,3]
            pred_norm = model(x, adj_t).cpu().numpy()[0]            # [N]
            pred_raw = np.maximum(pred_norm * t_std + t_mean, 0.0)
            for j in range(snap.num_nodes):
                out[(j, s["date"])] = (float(s["y_raw"][j]), float(pred_raw[j]))
    return out


def har_baseline(snap, floor: float) -> dict:
    """Pooled HAR OLS on raw HAR features -> {(ticker_id, date): (y_raw, pred_raw)} on the test obs."""
    import baselines as B
    Xtr = np.concatenate([s["har_raw"] for s in snap.train])       # [ntr*N, 3]
    ytr = np.concatenate([s["y_raw"] for s in snap.train])         # [ntr*N]
    coef = B.har_fit(Xtr, ytr)
    out = {}
    for s in snap.test:
        pred = B.har_predict(s["har_raw"], coef, floor=floor)      # [N]
        for j in range(snap.num_nodes):
            out[(j, s["date"])] = (float(s["y_raw"][j]), float(pred[j]))
    return out


def garch_baseline(snap, floor: float) -> dict:
    """Per-ticker GARCH(1,1) forecast on the test obs -> {(ticker_id, date): (y_raw, pred_raw)}."""
    import baselines as B
    # per-ticker train variance series (from snapshot targets, train split) in date order
    tr_series = {j: [] for j in range(snap.num_nodes)}
    for s in snap.train:
        for j in range(snap.num_nodes):
            tr_series[j].append(s["y_raw"][j])
    test_by_ticker = {j: [] for j in range(snap.num_nodes)}
    for s in snap.test:
        for j in range(snap.num_nodes):
            test_by_ticker[j].append((s["date"], float(s["y_raw"][j])))
    out = {}
    for j in range(snap.num_nodes):
        series = np.asarray(tr_series[j], dtype=float)
        n_test = len(test_by_ticker[j])
        try:
            fc = B.garch_forecast(series, n_test=n_test, horizon=snap.horizon, floor=floor)
        except Exception:
            fc = np.full(n_test, max(float(np.var(series)), floor))
        for i, (date, yraw) in enumerate(test_by_ticker[j]):
            out[(j, date)] = (yraw, float(fc[i]))
    return out


def ensemble(pred_dicts: list) -> dict:
    """Seed-ensemble a list of {(key): (y_raw, pred_raw)} -> mean prediction on shared keys."""
    keys = set(pred_dicts[0])
    for d in pred_dicts[1:]:
        keys &= set(d)
    keys = sorted(keys)
    out = {}
    for k in keys:
        y = pred_dicts[0][k][0]
        p = float(np.mean([d[k][1] for d in pred_dicts]))
        out[k] = (y, p)
    return out


def all_metrics(pred: dict, floor: float) -> dict:
    keys = sorted(pred)
    y = np.array([pred[k][0] for k in keys]); p = np.array([pred[k][1] for k in keys])
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "qlike": M.qlike(y, p, floor=floor), "r2": M.r2(y, p), "n": len(keys)}


def dm_compare(pred_a: dict, pred_b: dict, horizon: int, loss: str, floor: float):
    keys = sorted(set(pred_a) & set(pred_b))
    y = np.array([pred_a[k][0] for k in keys])
    pa = np.array([pred_a[k][1] for k in keys]); pb = np.array([pred_b[k][1] for k in keys])
    la = M.per_obs_qlike(y, pa, floor=floor) if loss == "qlike" else M.per_obs_se(y, pa)
    lb = M.per_obs_qlike(y, pb, floor=floor) if loss == "qlike" else M.per_obs_se(y, pb)
    r = M.diebold_mariano(la, lb, h=horizon)
    return {"dm_hln": r.dm_hln, "p_value": r.p_value, "mean_diff": r.mean_diff, "n": r.n,
            "favors": "A" if r.mean_diff < 0 else "B"}
