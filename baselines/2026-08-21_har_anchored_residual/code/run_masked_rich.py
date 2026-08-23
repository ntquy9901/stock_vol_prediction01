"""Fairest graph test: masked union-of-dates panel + 5 deliverable node features + a DIRECTED
volume->PK edge consumed by a WEIGHT-AWARE, 2-HOP GAT (edge weight AND sign change the output, unlike
the binary-mask GATLayer; two stacked hops replicate the deliverable's ``gat_layers=2`` default).
Compares HAR, LSTM(5-feat, no graph), LSTM+wGAT-2hop(vol->PK) and LSTM+wGAT-2hop(corr) on the SAME
masked folds, reporting all metrics (MSE/RMSE/MAE/QLIKE/R2) plus per-metric date-clustered
Diebold-Mariano for graph-vs-HAR and graph-vs-no-graph-LSTM.

CLI: python run_masked_rich.py <dataset> <horizon> [--price-dir DIR] [--data-root DIR] [--batch B]
     [--single] [--no-corr] [--smoke]
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

torch.backends.cudnn.benchmark = True

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines as B  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
from config import Config, SMOKE  # noqa: E402
import masked_rich as MR  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


class WeightedGATLayer(nn.Module):
    """Multi-head GAT whose attention AND messages consume the signed, weighted adjacency.

    Attention logit ``e_ij = LeakyReLU(a_dst.Wh_i + a_src.Wh_j + b * A_ij)`` (edge-conditioned), and
    the aggregated message multiplies each neighbour's transformed value by the edge weight:
    ``out_i = ELU( sum_j softmax_j(e_ij) * (A_ij * Wh_j) )``. Flipping an edge's sign or changing its
    weight therefore changes the output (the binary GATLayer ignored both). Mask-aware: only ``A_ij != 0``
    neighbours enter the softmax; invalid-neighbour columns are zeroed by the caller (node_mask), and the
    self-loop ``A_ii = 1`` keeps every node connected. ``A[i, j]`` is the edge from source ``j`` into
    target ``i`` (same convention as the correlation / vol->PK adjacencies).
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_dim))
        self.edge_bias = nn.Parameter(torch.ones(heads))   # init 1.0 -> edge weight consumed from the start
        self.leaky = nn.LeakyReLU(negative_slope)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        wh = self.W(h).view(b, n, self.heads, self.out_dim)
        e_src = (wh * self.a_src).sum(-1)                       # [b,n,heads]
        e_dst = (wh * self.a_dst).sum(-1)                       # [b,n,heads]
        # edge-conditioned logit: add b * A_ij (per head) so sign/weight shift attention
        a_term = adjacency.unsqueeze(-1) * self.edge_bias.view(1, 1, 1, self.heads)   # [b,i,j,heads]
        e = self.leaky(e_dst.unsqueeze(2) + e_src.unsqueeze(1) + a_term)
        mask = (adjacency != 0).unsqueeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        # weighted message: multiply attention by the signed edge weight -> messages carry sign/magnitude
        msg = alpha * adjacency.unsqueeze(-1)                   # [b,i,j,heads]
        out = torch.einsum("bijh,bjho->biho", msg, wh)
        return torch.nn.functional.elu(out.reshape(b, n, self.heads * self.out_dim))


class MaskedRichNet(nn.Module):
    """LSTM(5-feat) temporal branch + optional 2-HOP weighted-GAT spatial branch.

    The GAT is stacked TWO hops (``gat1: in_dim -> hidden*heads``, ``gat2: hidden*heads -> hidden*heads``)
    exactly like the old deliverable (``deliverables_20260817/.../2026-08-15_volatility/code/model.py``
    ``gat_layers=2``, whose P2 report kept 2-hop over 1-hop on VN). WeightedGATLayer ends in ELU, so that
    is the between-hop nonlinearity (matching the deliverable's GATLayer stacking); only the last hop's
    output feeds the head (``gnn_dim = hidden*heads``). The same masked adjacency is applied at BOTH hops.
    """

    def __init__(self, hidden=64, heads=4, dropout=0.2, use_graph=True, in_dim=MR.N_FEAT, gat_layers=2):
        super().__init__()
        self.use_graph, self.hidden, self.gat_layers = use_graph, hidden, gat_layers
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        gdim = hidden * heads if use_graph else 0
        if use_graph:
            self.gat1 = WeightedGATLayer(in_dim, hidden, heads)               # in_dim -> hidden*heads
            if gat_layers == 2:
                self.gat2 = WeightedGATLayer(hidden * heads, hidden, heads)   # hidden*heads -> hidden*heads
        self.head = nn.Sequential(nn.Linear(hidden + gdim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _gat(self, node_raw, adj_b):
        out = self.gat1(node_raw, adj_b)
        if self.gat_layers == 2:
            out = self.gat2(out, adj_b)          # 2nd hop, same masked adjacency
        return out

    def forward(self, x, adj_b):                    # x [B,N,seq,5]; adj_b [B,N,N] (invalid cols zeroed)
        b, n, seq, d = x.shape
        out, _ = self.lstm(x.reshape(b * n, seq, d))
        h = out[:, -1].reshape(b, n, self.hidden)
        parts = [h]
        if self.use_graph:
            parts.append(self._gat(x[:, :, -1, :], adj_b))
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def _batches(nrow, bs, shuffle, seed=0):
    idx = np.arange(nrow)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, nrow, bs):
        yield idx[i:i + bs]


def train_masked_rich(D, cfg, seed, use_graph, adj, output_param="zscore_floor"):
    """output_param: 'zscore_floor' (delivered default: standardized target, linear denorm, 1e-2*mean
    floor) or 'ratio_exp' (node-scaled ratio target y/mean + exp output, no relative floor; the
    review-validated parameterization that removes QLIKE seed-instability -- see
    docs/reports/2026-08-22_output_parameterization_robustness.md). Only the deep model changes."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = MaskedRichNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph).to(dev)
    if output_param == "ratio_exp":                 # bias-match: exp(0)=1 starts at the mean ratio
        with torch.no_grad():
            net.head[-1].bias.fill_(0.0)
    base = torch.from_numpy(adj).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    if output_param == "ratio_exp":
        ytr_n = torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) / (tmean + 1e-12)
    else:
        ytr_n = ((torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd)
    bs = cfg.batch_size
    tiny = float(np.finfo(np.float32).tiny)

    def adj_batch(nm):                              # [b,N,N] = base * valid-neighbour (source) mask
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def _apply(pn):                                 # network output -> prediction in the training space
        return torch.exp(pn.clamp(max=15.0)) if output_param == "ratio_exp" else pn

    def infer(X_np, nm_np):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs)
        if output_param == "ratio_exp":             # exp(pn)*mean: positive by construction, machine-eps only
            return np.maximum(np.exp(np.minimum(pn, 15.0)) * D.t_mean, tiny)
        return np.maximum(pn * D.t_std + D.t_mean, 1e-2 * D.t_mean + 1e-12)

    best = np.inf; best_state = None; wait = 0
    for ep in range(cfg.epochs):
        net.train()
        for idx in _batches(len(Xtr), bs, True, seed + ep):
            xb = Xtr[idx]; nmb = nmtr[idx]; tmb = tmtr[idx]; yb = ytr_n[idx]
            opt.zero_grad(); pred = _apply(net(xb, adj_batch(nmb)))
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer(D.X_va, D.nmask_va)
        m = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[m] - D.y_va[m]) ** 2))
        sched.step(vmse)
        if vmse < best - 1e-12:
            best = vmse; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}; wait = 0
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    return infer(D.X_te, D.nmask_te)


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


def seed_metric_stats(seed_dicts, floor):
    """Per-seed metric summary for a learned model: the MEAN (and std/min/max) of the seed-level metrics
    across the per-seed prediction dicts, plus the raw per-seed values. This is the faithful '5-seed mean'
    -- the average of seed-level metrics -- as opposed to ``_metrics(_ens(...))``, which is the metric of
    the seed-averaged (ensemble) prediction and is generally lower. ``n`` = scored obs (same every seed)."""
    ms = [_metrics(d, floor) for d in seed_dicts]
    keys = ("mse", "rmse", "mae", "qlike", "r2")
    out = {"n": ms[0]["n"], "n_seeds": len(ms)}
    for k in keys:
        vals = [float(m[k]) for m in ms]
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
        out[k + "_min"] = float(min(vals))
        out[k + "_max"] = float(max(vals))
    out["per_seed"] = {k: [float(m[k]) for m in ms] for k in keys}
    return out


def _dm_all(a, b, horizon, floor):                 # date-clustered DM on QLIKE / SE / AE; favors A if <0
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


def run(dataset, files, price_dir, horizon, cfg, lookback=10, with_corr=True, output_param="zscore_floor"):
    t0 = time.time()
    D = MR.build_masked_rich(files, price_dir, lookback, horizon)
    N = D.N
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    hp = B.har_predict(D.har_te.reshape(-1, 3), coef, floor=cfg.qlike_floor).reshape(D.y_te.shape)
    hp = np.maximum(hp, 1e-2 * D.t_mean + 1e-12)   # M1: shared per-node positivity floor (1e-2*mean; the value that removes the QLIKE tail-collapse -- HIGHER than the earlier 1e-3, i.e. clamps more). Under output_param='ratio_exp' the DEEP model is positive-by-construction (machine-eps only) while HAR/HAR-X keep this relative floor; QLIKE compares all models at the shared cfg.qlike_floor.
    HAR = _pred_dict(hp, D.y_te, D.tmask_te, D.d_te, N)
    # HAR-X: 5-feature LINEAR OLS — the fair baseline isolating the extra-feature contribution (so a deep
    # or graph win over HAR is not just the 2 extra node features market_pk/volume_zscore).
    _xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    _cx = np.linalg.lstsq(_xtr, D.y_tr[mtr], rcond=None)[0]
    _hx = (np.column_stack([np.ones(len(D.har5_te.reshape(-1, 5))), D.har5_te.reshape(-1, 5)]) @ _cx).reshape(D.y_te.shape)
    HARX = _pred_dict(np.maximum(_hx, 1e-2 * D.t_mean + 1e-12), D.y_te, D.tmask_te, D.d_te, N)
    # keep the per-seed prediction dicts so we can report the MEAN (+-std) of seed-level metrics, not just
    # the metric of the seed-averaged (ensemble) prediction (which is generally lower -- see seed_metric_stats)
    lstm_seeds = [_pred_dict(train_masked_rich(D, cfg, s, False, D.adj_vol2pk, output_param), D.y_te, D.tmask_te, D.d_te, N) for s in cfg.seeds]
    gat_seeds = [_pred_dict(train_masked_rich(D, cfg, s, True, D.adj_vol2pk, output_param), D.y_te, D.tmask_te, D.d_te, N) for s in cfg.seeds]
    lstm = _ens(lstm_seeds); gat_v = _ens(gat_seeds)
    preds = {"HAR": HAR, "HAR-X": HARX, "LSTM": lstm, "LSTM_wGAT_vol2pk": gat_v}
    per_seed = {"LSTM": seed_metric_stats(lstm_seeds, cfg.qlike_floor),
                "LSTM_wGAT_vol2pk": seed_metric_stats(gat_seeds, cfg.qlike_floor)}
    if with_corr:
        gat_c_seeds = [_pred_dict(train_masked_rich(D, cfg, s, True, D.adj_corr, output_param), D.y_te, D.tmask_te, D.d_te, N) for s in cfg.seeds]
        gat_c = _ens(gat_c_seeds)
        preds["LSTM_wGAT_corr"] = gat_c
        per_seed["LSTM_wGAT_corr"] = seed_metric_stats(gat_c_seeds, cfg.qlike_floor)
    metrics = {k: _metrics(v, cfg.qlike_floor) for k, v in preds.items()}
    dm = {"HARX_vs_HAR": _dm_all(HARX, HAR, horizon, cfg.qlike_floor),
          "LSTM_vs_HAR": _dm_all(lstm, HAR, horizon, cfg.qlike_floor),
          "LSTM_vs_HARX": _dm_all(lstm, HARX, horizon, cfg.qlike_floor),
          "wGAT_vol2pk_vs_HAR": _dm_all(gat_v, HAR, horizon, cfg.qlike_floor),
          "wGAT_vol2pk_vs_HARX": _dm_all(gat_v, HARX, horizon, cfg.qlike_floor),
          "wGAT_vol2pk_vs_LSTM": _dm_all(gat_v, lstm, horizon, cfg.qlike_floor)}
    if with_corr:
        dm["wGAT_corr_vs_HAR"] = _dm_all(preds["LSTM_wGAT_corr"], HAR, horizon, cfg.qlike_floor)
        dm["wGAT_corr_vs_LSTM"] = _dm_all(preds["LSTM_wGAT_corr"], lstm, horizon, cfg.qlike_floor)
    n_dates = len(set(k[1] for k in HAR))
    res = {"dataset": dataset, "horizon": horizon, "design": "masked-union-panel-rich-5feat-weighted-gat",
           "output_param": output_param,
           "num_nodes": N, "n_test_obs": metrics["HAR"]["n"], "n_test_dates": n_dates,
           "seeds": list(cfg.seeds), "lookback": lookback,
           "config": {"batch_size": cfg.batch_size, "epochs": cfg.epochs, "patience": cfg.patience,
                      "min_epochs": cfg.min_epochs},
           "metrics": metrics, "metrics_per_seed": per_seed, "dm_date_clustered": dm,
           "seconds": round(time.time() - t0, 1)}
    _subdir = "masked_rich_floor1e2" if output_param == "zscore_floor" else f"masked_rich_{output_param}"
    outp = REPO / "results" / _subdir / f"{dataset}_h{horizon}"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / "result.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"[rich] {dataset} h{horizon} N={N} dates={n_dates} obs={metrics['HAR']['n']} | "
          f"QLIKE HAR={metrics['HAR']['qlike']:.4f} LSTM={metrics['LSTM']['qlike']:.4f} "
          f"wGATv2p={metrics['LSTM_wGAT_vol2pk']['qlike']:.4f}"
          + (f" wGATcorr={metrics['LSTM_wGAT_corr']['qlike']:.4f}" if with_corr else "")
          + f" | MAE HAR={metrics['HAR']['mae']:.6f} wGATv2p={metrics['LSTM_wGAT_vol2pk']['mae']:.6f} "
          f"{res['seconds']}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["vn30", "vn100", "sp500"])
    ap.add_argument("horizon", type=int)
    ap.add_argument("--data-root", default=str(_SUB / "data"))
    ap.add_argument("--price-dir", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-corr", action="store_true")
    ap.add_argument("--single", action="store_true", help="1 seed (42) quick gate, full epochs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-param", choices=["zscore_floor", "ratio_exp"], default="zscore_floor",
                    help="deep-model output: zscore_floor (delivered default) or ratio_exp (review-validated, no floor)")
    a = ap.parse_args()
    cfg = SMOKE if a.smoke else Config()
    from dataclasses import replace
    cfg = replace(cfg, batch_size=a.batch)
    if a.single:
        cfg = replace(cfg, seeds=(42,))
    root = Path(a.data_root)
    mp = {"vn30": [root / "vn30" / "*_processed.csv", root / "*_processed.csv"],
          "vn100": [root / "vn100" / "*_processed.csv", root / "vn100_vnstock" / "*_processed.csv"],
          "sp500": [root / "sp500" / "*_processed.csv"]}
    files = next((glob.glob(str(p)) for p in mp[a.dataset] if glob.glob(str(p))), [])
    pd_map = {"vn30": REPO / "data" / "raw" / "prices",
              "vn100": REPO / "data" / "raw" / "prices" / "vn100_vnstock",
              "sp500": REPO / "data" / "raw" / "prices" / "sp500"}
    price_dir = a.price_dir or str(pd_map[a.dataset])
    run(a.dataset, files, price_dir, a.horizon, cfg, with_corr=not a.no_corr, output_param=a.output_param)


if __name__ == "__main__":
    main()
