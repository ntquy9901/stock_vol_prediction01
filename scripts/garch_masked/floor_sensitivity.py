"""Floor-sensitivity under a COMMON evaluation floor (per reviewer 2026-08-22).

Fairness requires every forecast to enter the metric under the SAME floor, applied to RAW forecasts:
    y_eval[m,i,t] = max(y_raw[m,i,t], eps_i)   with identical eps_i for HAR, LSTM(C), ...
The delivered runner instead PRE-floors HAR/HAR-X at 1e-2*mean before the metric, so a later shared 1e-8
QLIKE clamp does not restore fairness (HAR was already modified). This script re-scores from RAW.

For config C (ratio_exp deep model), on a (dataset, horizon):
  - raw HAR-X  = 5-feature OLS, NO floor (can be <=0);
  - raw deep-C = 5-seed ensemble of exp(pn)*mean, NO relative floor (positive by construction; TINY only
    guards fp underflow, it is not a relative floor).
Then re-score both models under two COMMON floor policies applied IDENTICALLY to both:
  P_eps  : eps_i = 1e-8            (fixed small; protects the log, does not move economic forecasts)
  P_econ : eps_i = 1e-2 * mean_i   (per-node)
Report per policy: QLIKE(HAR), QLIKE(deep), clip-% per model, DM deep-vs-HAR (date-clustered). If the
deep-vs-HAR conclusion is stable across both policies, it does not depend on the clipping choice.

Usage: python floor_sensitivity.py <vn30|vn100> <horizon>
"""
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
SUB = REPO / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(CODE))

import run_masked_rich as RM  # noqa: E402
import masked_rich as MR  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
from config import Config  # noqa: E402

EPS = 1e-12
QF = 1e-8
TINY = float(np.finfo(np.float32).tiny)
SEEDS = (42, 123, 2026, 7, 2024)


def _train_raw_ratio_exp(D, cfg, seed, adj):
    """Config C deep model; returns RAW test forecasts exp(pn)*mean (no relative floor, TINY guard only)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = RM.MaskedRichNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph=False).to(dev)
    with torch.no_grad():
        net.head[-1].bias.fill_(0.0)                 # bias-match: exp(0)=1
    base = torch.from_numpy(adj).to(dev)
    tmean = D.t_mean.astype(np.float32)
    ytr_n = torch.from_numpy((D.y_tr.astype(np.float32) / (tmean + EPS))).to(dev)
    Xtr = torch.from_numpy(D.X_tr).to(dev)
    nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bs = cfg.batch_size

    def adj_batch(nm):
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def raw_pred(X_np, nm_np):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs).astype(np.float64)
        return np.maximum(np.exp(np.minimum(pn, 15.0)) * tmean, TINY)   # positive; TINY = fp guard only

    best, best_state, wait = np.inf, None, 0
    for ep in range(cfg.epochs):
        net.train()
        for idx in RM._batches(len(Xtr), bs, True, seed + ep):
            xb, nmb, tmb, yb = Xtr[idx], nmtr[idx], tmtr[idx], ytr_n[idx]
            opt.zero_grad()
            pred = torch.exp(net(xb, adj_batch(nmb)).clamp(max=15.0))
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        pva = raw_pred(D.X_va, D.nmask_va)
        mv = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[mv] - D.y_va[mv]) ** 2))
        sched.step(vmse)
        if np.isfinite(vmse) and vmse < best - 1e-12:
            best, wait = vmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    return raw_pred(D.X_te, D.nmask_te)


def score_common_floor(y, raw, floor_vec):
    """Apply an identical common floor to RAW forecasts, then metrics. Returns (metrics, clip_fraction)."""
    ev = np.maximum(raw, floor_vec)
    clip = float(np.mean(raw < floor_vec))
    m = dict(mse=M.mse(y, ev), rmse=M.rmse(y, ev), mae=M.mae(y, ev),
             qlike=M.qlike(y, ev, QF), r2=M.r2(y, ev))
    return m, clip


def _flat(D):
    m = D.tmask_te.astype(bool)
    rows, cols = np.where(m)
    dates = np.array([D.d_te[r] for r in rows])
    return m, D.y_te[m], dates, cols


def screen_files(files, min_rows=250, max_zero_frac=0.5):
    """Liquidity + history screen for a HOSE/HNX universe (data-quality audit 2026-08-23): keep a ticker
    only if it has >= min_rows processed rows AND its fraction of zero Parkinson-variance days (H==L,
    illiquid) is <= max_zero_frac. Illiquid tickers make QLIKE/point metrics uninformative. Returns the
    kept file list; deterministic (sorted)."""
    import pandas as pd
    kept = []
    for f in sorted(files):
        try:
            v = pd.read_csv(f)["parkinson_volatility"].to_numpy(float)
        except Exception:
            continue
        if len(v) >= min_rows and float(np.mean(v == 0.0)) <= max_zero_frac:
            kept.append(f)
    return kept


def main():
    ds, h = sys.argv[1], int(sys.argv[2])
    cfg = replace(Config(), batch_size=32)
    fmap = {"vn30": REPO / "submission/soict_lstm_gat/data/vn30/*_processed.csv",
            "vn100": REPO / "submission/soict_lstm_gat/data/vn100/*_processed.csv"}
    pmap = {"vn30": REPO / "data/raw/prices", "vn100": REPO / "data/raw/prices/vn100_vnstock"}
    files = glob.glob(str(fmap[ds]))
    D = MR.build_masked_rich(files, str(pmap[ds]), cfg.lookback, h)
    m, y, dates, node = _flat(D)
    print(f"{ds} h{h}: N={D.N} obs={len(y)} dates={len(set(dates))}", flush=True)

    # raw HAR-X (5-feature OLS), NO floor (can be <= 0)
    mtr = D.tmask_tr.astype(bool)
    xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    cx = np.linalg.lstsq(xtr, D.y_tr[mtr], rcond=None)[0]
    har_raw = (np.column_stack([np.ones(len(D.har5_te.reshape(-1, 5))), D.har5_te.reshape(-1, 5)]) @ cx
               ).reshape(D.y_te.shape)[m]

    # raw deep-C (ratio_exp), 5-seed ensemble, NO relative floor
    deep_raw = np.mean([_train_raw_ratio_exp(D, cfg, s, D.adj_vol2pk)[m] for s in SEEDS], axis=0)

    floors = {"P_eps_1e-8": np.full(len(y), 1e-8),
              "P_econ_1e-2mean": (1e-2 * D.t_mean[node] + 1e-12)}
    out = {"dataset": ds, "horizon": h, "n_obs": int(len(y)), "seeds": list(SEEDS), "policies": {}}
    for name, fv in floors.items():
        hm, hc = score_common_floor(y, har_raw, fv)
        dm_, dc = score_common_floor(y, deep_raw, fv)
        # DM deep-vs-HAR on per-obs QLIKE of the COMMON-floored forecasts
        la = M.per_obs_qlike(y, np.maximum(deep_raw, fv), QF)
        lb = M.per_obs_qlike(y, np.maximum(har_raw, fv), QF)
        r = ST.date_clustered_dm(la, lb, dates, h)
        out["policies"][name] = {
            "HAR": {**hm, "clip_frac": hc}, "deep_C": {**dm_, "clip_frac": dc},
            "DM_deepC_vs_HAR": {"p_value": r["p_value"], "mean_diff": r["mean_diff"],
                                "favors": "deep_C" if r["mean_diff"] < 0 else "HAR"}}
        print(f"  [{name}] HAR QLIKE={hm['qlike']:.4f}(clip {hc:.4%}) deepC QLIKE={dm_['qlike']:.4f}"
              f"(clip {dc:.4%}) | DM deepC-vs-HAR p={r['p_value']:.4f} "
              f"favors={out['policies'][name]['DM_deepC_vs_HAR']['favors']}", flush=True)

    outp = REPO / "results" / "floor_sensitivity"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / f"{ds}_h{h}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {outp / f'{ds}_h{h}.json'}", flush=True)


if __name__ == "__main__":
    main()
