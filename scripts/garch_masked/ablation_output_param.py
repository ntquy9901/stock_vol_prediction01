"""Ablation (per reviewer): separate the effects of (i) ratio vs z-score target normalization,
(ii) positivity enforcement, (iii) softplus vs exp, (iv) removing the post-hoc floor.

Four configs on S&P500 h5 (worst QLIKE blow-up), LSTM (no graph), seeds {42,123}, same deterministic panel:
  A z-score + linear + floor   (CURRENT)  : target (y-mu)/sigma; infer max(pn*sigma+mu, 1e-2*mu)
  B ratio   + linear + floor              : target y/mu;         infer max(pn*mu, 1e-2*mu)
  C ratio   + exp    (no floor)           : target y/mu;         infer exp(pn)*mu
  D ratio   + softplus (no floor)         : target y/mu;         infer softplus(pn)*mu

A machine-epsilon (np.finfo.tiny) is applied to every forecast purely as a numerical safeguard for the
QLIKE division/log -- NOT the economic 1e-2*mu floor. QLIKE clamps target and forecast to a shared 1e-8
(the existing protocol; y==0 Parkinson-variance days are thereby clamped consistently across all configs).
Reports all metrics per seed, 2-seed mean, and the QLIKE seed-spread (the stability of interest).
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
TINY = float(np.finfo(np.float32).tiny)

CONFIGS = {
    "A_zscore_linear_floor": dict(norm="zscore", act="linear", floor=True),
    "B_ratio_linear_floor":  dict(norm="ratio",  act="linear", floor=True),
    "C_ratio_exp":           dict(norm="ratio",  act="exp",    floor=False),
    "D_ratio_softplus":      dict(norm="ratio",  act="softplus", floor=False),
}


def _train(D, cfg, seed, adj, spec):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = RM.MaskedRichNet(cfg.hidden, cfg.heads, cfg.dropout, use_graph=False).to(dev)
    base = torch.from_numpy(adj).to(dev)
    tmean = D.t_mean.astype(np.float32)
    econ_floor = (1e-2 * D.t_mean + 1e-12).astype(np.float64)

    if spec["norm"] == "zscore":
        loc, scale = tmean, D.t_std.astype(np.float32)
        ytr_np = (D.y_tr.astype(np.float32) - loc) / scale
    else:  # ratio
        loc, scale = tmean, np.ones_like(tmean)
        ytr_np = D.y_tr.astype(np.float32) / (loc + EPS)

    ytr_n = torch.from_numpy(ytr_np.astype(np.float32)).to(dev)
    Xtr = torch.from_numpy(D.X_tr).to(dev)
    nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bs = cfg.batch_size

    def adj_batch(nm):
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def act_t(pn):
        if spec["act"] == "exp":
            return torch.exp(pn.clamp(max=15.0))     # clamp only to avoid fp overflow crash
        if spec["act"] == "softplus":
            return F.softplus(pn)
        return pn

    def raw_pred(X_np, nm_np):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs).astype(np.float64)
        if spec["act"] == "exp":
            a = np.exp(np.minimum(pn, 15.0))
        elif spec["act"] == "softplus":
            a = np.logaddexp(0.0, pn)
        else:
            a = pn
        var = (a * scale.astype(np.float64) + loc.astype(np.float64)) if spec["norm"] == "zscore" \
            else (a * loc.astype(np.float64))
        if spec["floor"]:
            var = np.maximum(var, econ_floor)          # economic floor (configs A,B)
        return np.maximum(var, TINY)                   # machine-eps safeguard (all configs)

    best, best_state, wait = np.inf, None, 0
    for ep in range(cfg.epochs):
        net.train()
        for idx in RM._batches(len(Xtr), bs, True, seed + ep):
            xb, nmb, tmb, yb = Xtr[idx], nmtr[idx], tmtr[idx], ytr_n[idx]
            opt.zero_grad()
            pred = act_t(net(xb, adj_batch(nmb)))
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

    for name, spec in CONFIGS.items():
        res = {}
        for seed in (42, 123):
            res[seed] = _metrics(_train(D, cfg, seed, D.adj_vol2pk, spec), D)
            print(f"{name:<24} seed{seed}: " + str({k: round(v, 4) for k, v in res[seed].items()}), flush=True)
        ql = [res[s]["qlike"] for s in (42, 123)]
        mean = {k: (res[42][k] + res[123][k]) / 2 for k in res[42]}
        print(f"  -> {name} MEAN: " + str({k: round(v, 4) for k, v in mean.items()}))
        print(f"  -> {name} QLIKE seeds={[round(q, 4) for q in ql]} spread={abs(ql[0] - ql[1]):.4f}\n", flush=True)


if __name__ == "__main__":
    main()
