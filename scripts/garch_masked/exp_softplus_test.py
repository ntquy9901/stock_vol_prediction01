"""Experiment (1b): softplus output (positive-by-construction, but linear-tailed, no exp amplification)
vs the current linear-denorm + floor, on S&P500 h5 (worst QLIKE blow-up).

Same deterministic panel, same MaskedRichNet, same masked-MSE objective. Output mappings:
  - LINEAR  (current): train (y-mean)/std; infer max(pn*std+mean, 1e-2*mean)                 [floor]
  - SOFTPLUS(new):     train softplus(pn) toward ratio y/mean; infer softplus(pn)*mean       [no floor]
softplus(x)=log(1+e^x) is >0 everywhere and grows ~linearly (unlike exp), so it cannot blow up the way
log-variance+exp did (QLIKE ~126 in the previous test). Runs LSTM (no graph), seeds {42,123}; reports all
metrics + the QLIKE seed-spread (the stability we care about).
"""
import glob
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"
SUB = REPO / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(CODE))

import run_masked_rich as RM  # noqa: E402
import masked_rich as MR  # noqa: E402
import metrics as M  # noqa: E402
import baselines as B  # noqa: E402
from config import Config  # noqa: E402

EPS = 1e-12
QF = 1e-8


def _train(D, cfg, seed, adj, mode):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = RM.MaskedRichNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph=False).to(dev)
    base = torch.from_numpy(adj).to(dev)
    tmean = D.t_mean.astype(np.float32)

    if mode == "linear":
        loc, scale = tmean, D.t_std.astype(np.float32)
        ytr_np = (D.y_tr.astype(np.float32) - loc) / scale
    else:  # softplus: target is the ratio y / per-node mean
        loc, scale = tmean, np.ones_like(tmean)
        ytr_np = D.y_tr.astype(np.float32) / (loc + EPS)

    ytr_n = torch.from_numpy(ytr_np.astype(np.float32)).to(dev)
    loc_np = loc
    scale_np = scale
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
        pn = np.concatenate(outs)
        if mode == "linear":
            return np.maximum(pn * scale_np + loc_np, 1e-2 * D.t_mean + 1e-12)
        return np.logaddexp(0.0, pn) * loc_np                  # softplus(pn) * mean, positive, no floor

    best, best_state, wait = np.inf, None, 0
    for ep in range(cfg.epochs):
        net.train()
        for idx in RM._batches(len(Xtr), bs, True, seed + ep):
            xb, nmb, tmb, yb = Xtr[idx], nmtr[idx], tmtr[idx], ytr_n[idx]
            opt.zero_grad()
            pn = net(xb, adj_batch(nmb))
            pred = F.softplus(pn) if mode == "softplus" else pn
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        pva = raw_pred(D.X_va, D.nmask_va)
        mv = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[mv] - D.y_va[mv]) ** 2))
        sched.step(vmse)
        if vmse < best - 1e-12:
            best, wait = vmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    return raw_pred(D.X_te, D.nmask_te)


def _metrics(pred, D):
    m = D.tmask_te.astype(bool)
    y, p = D.y_te[m], pred[m]
    return dict(mse=M.mse(y, p), rmse=M.rmse(y, p), mae=M.mae(y, p), qlike=M.qlike(y, p, QF), r2=M.r2(y, p))


def main():
    cfg = replace(Config(), batch_size=32)
    files = glob.glob(str(REPO / "data" / "processed" / "sp500" / "*_processed.csv"))
    price_dir = str(REPO / "data" / "raw" / "prices" / "sp500")
    print("building sp500 h5 panel ...", flush=True)
    D = MR.build_masked_rich(files, price_dir, cfg.lookback, 5)
    print(f"N={D.N} test_obs={int(D.tmask_te.sum())}", flush=True)
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    hp = np.maximum(B.har_predict(D.har_te.reshape(-1, 3), coef, QF).reshape(D.y_te.shape), 1e-2 * D.t_mean + 1e-12)
    print("HAR (linear) ref  :", {k: round(v, 4) for k, v in _metrics(hp, D).items()}, flush=True)

    for mode in ("linear", "softplus"):
        res = {}
        for seed in (42, 123):
            res[seed] = _metrics(_train(D, cfg, seed, D.adj_vol2pk, mode), D)
            print(f"LSTM {mode:<8} seed{seed}: " + str({k: round(v, 4) for k, v in res[seed].items()}), flush=True)
        ql = [res[s]["qlike"] for s in (42, 123)]
        mean = {k: (res[42][k] + res[123][k]) / 2 for k in res[42]}
        print(f"  -> {mode} 2-seed MEAN: " + str({k: round(v, 4) for k, v in mean.items()}))
        print(f"  -> {mode} QLIKE seeds={[round(q, 4) for q in ql]} spread={abs(ql[0] - ql[1]):.4f}\n", flush=True)


if __name__ == "__main__":
    main()
