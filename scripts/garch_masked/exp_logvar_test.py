"""Experiment (1): does positive-by-construction output (log-variance + exp) stabilise QLIKE vs the
current linear-denorm + floor output, on the worst blow-up cell (S&P500 h5)?

Same deterministic masked-rich panel, same MaskedRichNet, same masked-MSE training. Only the TARGET
transform + inference mapping differ:
  - LINEAR (current): train on (y-mean)/std; infer = max(pn*std + mean, 1e-2*mean)   [floor]
  - LOGVAR  (new):    train on (log y - lmean)/lstd; infer = exp(pn*lstd + lmean + 0.5*sigma2)  [no floor]
sigma2 = validation residual variance on the log scale (log-normal retransformation / Jensen correction).

Runs LSTM (no graph) for seeds {42,123}, reports QLIKE + MSE/RMSE/MAE/R2 per seed, the 2-seed mean, and
the seed spread of QLIKE (the stability we care about). HAR (linear) and log-HAR are printed for reference.
"""
import sys
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
from config import Config  # noqa: E402

EPS = 1e-12
QF = 1e-8  # qlike floor (same as cfg.qlike_floor)


def _train(D, cfg, seed, adj, mode):
    """mode in {'linear','logvar'}. Returns test predictions on the raw variance scale."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = RM.MaskedRichNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph=False).to(dev)
    base = torch.from_numpy(adj).to(dev)
    m = D.tmask_tr.astype(bool)
    N = D.N

    if mode == "linear":
        loc = D.t_mean.astype(np.float32)
        scale = D.t_std.astype(np.float32)
        ytr = D.y_tr.astype(np.float32)
    else:  # logvar
        ly = np.log(D.y_tr + EPS)
        loc = np.array([ly[m[:, j], j].mean() if m[:, j].any() else 0.0 for j in range(N)], np.float32)
        scale = np.array([ly[m[:, j], j].std() if m[:, j].any() else 1.0 for j in range(N)], np.float32) + 1e-6
        ytr = ly.astype(np.float32)

    loc_t = torch.from_numpy(loc).to(dev)
    scale_t = torch.from_numpy(scale).to(dev)
    ytr_n = (torch.from_numpy(ytr).to(dev) - loc_t) / scale_t
    Xtr = torch.from_numpy(D.X_tr).to(dev)
    nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bs = cfg.batch_size

    def adj_batch(nm):
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def raw_pred(X_np, nm_np, sigma2=0.0):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs)
        if mode == "linear":
            return np.maximum(pn * scale + loc, 1e-2 * D.t_mean + 1e-12)
        return np.exp(pn * scale + loc + 0.5 * sigma2)   # positive by construction, NO floor

    best = np.inf
    best_state = None
    wait = 0
    for ep in range(cfg.epochs):
        net.train()
        for idx in RM._batches(len(Xtr), bs, True, seed + ep):
            xb, nmb, tmb, yb = Xtr[idx], nmtr[idx], tmtr[idx], ytr_n[idx]
            opt.zero_grad()
            pn = net(xb, adj_batch(nmb))
            loss = (((pn - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
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

    # log-normal retransformation variance from validation residuals (log scale)
    sigma2 = 0.0
    if mode == "logvar":
        net.eval()
        pv = []
        with torch.no_grad():
            for i in range(0, len(D.X_va), bs):
                xb = torch.from_numpy(D.X_va[i:i + bs]).to(dev)
                nmb = torch.from_numpy(D.nmask_va[i:i + bs]).to(dev)
                pv.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pv = np.concatenate(pv) * scale + loc            # predicted log-variance
        mv = D.tmask_va.astype(bool)
        resid = np.log(D.y_va[mv] + EPS) - pv[mv]
        sigma2 = float(np.var(resid))
    return raw_pred(D.X_te, D.nmask_te, sigma2)


def _metrics(pred, D):
    m = D.tmask_te.astype(bool)
    y = D.y_te[m]
    p = pred[m]
    return dict(mse=M.mse(y, p), rmse=M.rmse(y, p), mae=M.mae(y, p), qlike=M.qlike(y, p, QF), r2=M.r2(y, p))


def main():
    from dataclasses import replace
    cfg = replace(Config(), batch_size=32)   # 512 OOMs on 442-node sp500 @ 8GB GPU
    files = __import__("glob").glob(str(REPO / "data" / "processed" / "sp500" / "*_processed.csv"))
    price_dir = str(REPO / "data" / "raw" / "prices" / "sp500")
    print("building sp500 h5 masked-rich panel ...", flush=True)
    D = MR.build_masked_rich(files, price_dir, cfg.lookback, 5)
    print(f"N={D.N} test_obs={int(D.tmask_te.sum())}", flush=True)

    # HAR (linear) + log-HAR reference
    mtr = D.tmask_tr.astype(bool)
    import baselines as B
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    hp = np.maximum(B.har_predict(D.har_te.reshape(-1, 3), coef, QF).reshape(D.y_te.shape), 1e-2 * D.t_mean + 1e-12)
    print("HAR (linear)      :", {k: round(v, 4) for k, v in _metrics(hp, D).items()}, flush=True)
    lc, ls = _loghar_fit(D.har_tr[mtr], D.y_tr[mtr])
    lhp = _loghar_pred(D.har_te.reshape(-1, 3), lc, ls).reshape(D.y_te.shape)
    print("log-HAR (exp)     :", {k: round(v, 4) for k, v in _metrics(lhp, D).items()}, flush=True)

    for mode in ("linear", "logvar"):
        res = {}
        for seed in (42, 123):
            pred = _train(D, cfg, seed, D.adj_vol2pk, mode)
            res[seed] = _metrics(pred, D)
            print(f"LSTM {mode:<7} seed{seed}: " + str({k: round(v, 4) for k, v in res[seed].items()}), flush=True)
        ql = [res[s]["qlike"] for s in (42, 123)]
        mean = {k: (res[42][k] + res[123][k]) / 2 for k in res[42]}
        print(f"  -> {mode} 2-seed MEAN: " + str({k: round(v, 4) for k, v in mean.items()}))
        print(f"  -> {mode} QLIKE seeds={[round(q,4) for q in ql]} spread={abs(ql[0]-ql[1]):.4f}\n")


def _loghar_fit(X, y):
    lX = np.log(X + EPS)
    ly = np.log(y + EPS)
    design = np.column_stack([np.ones(len(lX)), lX])
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    sigma2 = float(np.var(ly - design @ coef))
    return coef, sigma2


def _loghar_pred(X, coef, sigma2):
    lX = np.log(X + EPS)
    lpred = coef[0] + lX @ coef[1:]
    return np.exp(lpred + 0.5 * sigma2)


if __name__ == "__main__":
    main()
