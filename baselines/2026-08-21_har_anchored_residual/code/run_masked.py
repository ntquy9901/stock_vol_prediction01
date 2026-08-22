"""Masked-panel graph experiment: HAR vs LSTM (no graph) vs LSTM+GAT (mask-aware), on the UNION-of-dates
masked panel (fixes common-date selection bias / low power). Reports ALL metrics (MSE/RMSE/MAE/QLIKE/R2)
and per-metric date-clustered Diebold-Mariano (QLIKE, squared-error, absolute-error) so the strengths of
LSTM and of the graph are judged fairly on each metric, not only QLIKE.

CLI: python run_masked.py <dataset> <horizon> [--data-root DIR] [--batch B] [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.backends.cudnn.benchmark = True   # fixed-length LSTM -> faster cuDNN kernels (safe, no numeric change)

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B  # noqa: E402
import metrics as M  # noqa: E402
import model as _submodel  # noqa: E402  (reuse GATLayer)
import stats as ST  # noqa: E402
from config import Config, SMOKE  # noqa: E402
import masked_snapshots as MS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


class MaskedNet(nn.Module):
    def __init__(self, hidden=64, heads=4, dropout=0.2, use_graph=True):
        super().__init__()
        self.use_graph, self.hidden = use_graph, hidden
        self.lstm = nn.LSTM(3, hidden, num_layers=2, batch_first=True, dropout=dropout)
        gdim = hidden * heads if use_graph else 0
        if use_graph:
            self.gat = _submodel.GATLayer(3, hidden, heads)
        self.head = nn.Sequential(nn.Linear(hidden + gdim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, adj_b):                    # x [B,N,seq,3]; adj_b [B,N,N] (invalid cols zeroed)
        b, n, seq, d = x.shape
        out, _ = self.lstm(x.reshape(b * n, seq, d))
        h = out[:, -1].reshape(b, n, self.hidden)
        parts = [h]
        if self.use_graph:
            parts.append(self.gat(x[:, :, -1, :], adj_b))
        return self.head(torch.cat(parts, -1)).squeeze(-1)   # [B,N]


def _batches(nrow, bs, shuffle, seed=0):
    idx = np.arange(nrow)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, nrow, bs):
        yield idx[i:i + bs]


def train_masked(D, cfg, seed, use_graph):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = MaskedNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph).to(dev)
    base = torch.from_numpy(D.adj).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    ytr_n = ((torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd)
    # large panels (sp500): the driver passes a conservative --batch; bump it so there are fewer,
    # larger steps (the kernel-launch/Python overhead — not compute — is the bottleneck per the repo
    # perf audit). Small panels (VN) keep the given batch.
    bs = max(cfg.batch_size, 32) if D.N > 200 else cfg.batch_size

    def adj_batch(nm):                              # [b,N,N] = base * valid-neighbour mask
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def infer(X_np, nm_np):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                p = net(xb, adj_batch(nmb)).cpu().numpy()
                outs.append(p)
        pn = np.concatenate(outs)
        return np.maximum(pn * D.t_std + D.t_mean, 1e-3 * D.t_mean + 1e-12)   # raw, floored

    best = np.inf; best_state = None; wait = 0
    for ep in range(cfg.epochs):
        net.train()
        for idx in _batches(len(Xtr), bs, True, seed + ep):
            xb = Xtr[idx]; nmb = nmtr[idx]; tmb = tmtr[idx]; yb = ytr_n[idx]
            opt.zero_grad(); pn = net(xb, adj_batch(nmb))
            loss = (((pn - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)   # MSE on valid targets only
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer(D.X_va, D.nmask_va)
        m = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[m] - D.y_va[m]) ** 2))                     # val MSE (metric-neutral)
        sched.step(vmse)
        if vmse < best - 1e-12:
            best = vmse; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}; wait = 0
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    pte = infer(D.X_te, D.nmask_te)
    return pte


def _pred_dict(pred, y, tmask, dates, N):
    out = {}
    for i in range(len(dates)):
        for j in range(N):
            if tmask[i, j]:
                out[(j, dates[i])] = (float(y[i, j]), float(pred[i, j]))
    return out


def _ens(dicts):
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    keys = sorted(keys)
    return {k: (dicts[0][k][0], float(np.mean([d[k][1] for d in dicts]))) for k in keys}


def _metrics(pred, floor):
    ks = sorted(pred); y = np.array([pred[k][0] for k in ks]); p = np.array([pred[k][1] for k in ks])
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "qlike": M.qlike(y, p, floor), "r2": M.r2(y, p), "n": len(ks)}


def _dm_all(a, b, horizon, floor):                 # date-clustered DM on QLIKE / SE / AE
    ks = sorted(set(a) & set(b)); y = np.array([a[k][0] for k in ks])
    pa = np.array([a[k][1] for k in ks]); pb = np.array([b[k][1] for k in ks])
    dates = np.array([k[1] for k in ks])
    fams = {"qlike": (M.per_obs_qlike(y, pa, floor), M.per_obs_qlike(y, pb, floor)),
            "se": ((y - pa) ** 2, (y - pb) ** 2),
            "ae": (np.abs(y - pa), np.abs(y - pb))}
    out = {}
    for name, (la, lb) in fams.items():
        try:
            r = ST.date_clustered_dm(la, lb, dates, horizon)
            out[name] = {"p_value": r["p_value"], "mean_diff": r["mean_diff"],
                         "favors": "A" if r["mean_diff"] < 0 else "B"}
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
    return out


def run(dataset, files, horizon, cfg, lookback=10):
    t0 = time.time()
    D = MS.build_masked(files, lookback, horizon)
    N = D.N
    # HAR (pooled OLS on train valid rows)
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    hp = B.har_predict(D.har_te.reshape(-1, 3), coef, floor=cfg.qlike_floor).reshape(D.y_te.shape)  # [n,N]
    HAR = _pred_dict(hp, D.y_te, D.tmask_te, D.d_te, N)
    # LSTM (no graph) + LSTM+GAT, seed-ensembled
    lstm = _ens([_pred_dict(train_masked(D, cfg, s, False), D.y_te, D.tmask_te, D.d_te, N) for s in cfg.seeds])
    gat = _ens([_pred_dict(train_masked(D, cfg, s, True), D.y_te, D.tmask_te, D.d_te, N) for s in cfg.seeds])
    preds = {"HAR": HAR, "LSTM": lstm, "LSTM_GAT": gat}
    metrics = {k: _metrics(v, cfg.qlike_floor) for k, v in preds.items()}
    dm = {"LSTM_vs_HAR": _dm_all(lstm, HAR, horizon, cfg.qlike_floor),
          "LSTM_GAT_vs_HAR": _dm_all(gat, HAR, horizon, cfg.qlike_floor),
          "LSTM_GAT_vs_LSTM": _dm_all(gat, lstm, horizon, cfg.qlike_floor)}
    n_dates = len(set(k[1] for k in HAR))
    res = {"dataset": dataset, "horizon": horizon, "design": "masked-union-panel", "num_nodes": N,
           "n_test_obs": metrics["HAR"]["n"], "n_test_dates": n_dates, "seeds": list(cfg.seeds),
           "metrics": metrics, "dm_date_clustered": dm, "seconds": round(time.time() - t0, 1)}
    outp = REPO / "results" / "masked_panel" / f"{dataset}_h{horizon}"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "result.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"[masked] {dataset} h{horizon} N={N} test_dates={n_dates} obs={metrics['HAR']['n']} "
          f"QLIKE HAR={metrics['HAR']['qlike']:.4f} LSTM={metrics['LSTM']['qlike']:.4f} "
          f"GAT={metrics['LSTM_GAT']['qlike']:.4f} | MAE HAR={metrics['HAR']['mae']:.6f} "
          f"GAT={metrics['LSTM_GAT']['mae']:.6f} {res['seconds']}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("horizon", type=int)
    ap.add_argument("--data-root", default=str(_SUB / "data"))
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = SMOKE if a.smoke else Config()
    if a.batch:
        from dataclasses import replace
        cfg = replace(cfg, batch_size=a.batch)
    root = Path(a.data_root)
    mp = {"vn30": [root / "vn30" / "*_processed.csv", root / "*_processed.csv"],
          "vn100": [root / "vn100" / "*_processed.csv", root / "vn100_vnstock" / "*_processed.csv"],
          "sp500": [root / "sp500" / "*_processed.csv"]}
    files = next((glob.glob(str(p)) for p in mp[a.dataset] if glob.glob(str(p))), [])
    run(a.dataset, files, a.horizon, cfg)


if __name__ == "__main__":
    main()
