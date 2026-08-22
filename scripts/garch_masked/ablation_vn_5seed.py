"""Rigorous cross-market/multi-seed validation of the output-parameterization ablation (per reviewer).

A/B/C/D x 5 seeds on VN30 and VN100 (h5), same panel/seed/init/split/mask across configs.
Reports full-precision MSE/RMSE/MAE/QLIKE/R2 (scientific), per-seed mean +/- std AND the 5-seed
ensemble (mean of predictions), plus date-clustered Diebold-Mariano on per-obs QLIKE for the ensembles:
D-vs-HAR and C-vs-D. Config is FIXED here (not selected on the comparison cell). Saves JSON per dataset.

Usage: python ablation_vn_5seed.py <vn30|vn100> [horizon]
"""
import glob
import json
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
import stats as ST  # noqa: E402
import baselines as B  # noqa: E402
from config import Config  # noqa: E402

EPS = 1e-12
QF = 1e-8
TINY = float(np.finfo(np.float32).tiny)
SEEDS = (42, 123, 2026, 7, 2024)

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
    # bias-match: initial prediction starts at the mean ratio (~1) / mean level for a FAIR link comparison.
    # exp: exp(0)=1 -> bias 0.  softplus: softplus(0.5413)=1 -> bias log(e-1).  ratio-linear: pred=bias=1.
    if spec["act"] == "exp":
        _ib = 0.0
    elif spec["act"] == "softplus":
        _ib = float(np.log(np.expm1(1.0)))          # ~0.5413
    else:  # linear
        _ib = 1.0 if spec["norm"] == "ratio" else 0.0
    with torch.no_grad():
        net.head[-1].bias.fill_(_ib)
    base = torch.from_numpy(adj).to(dev)
    tmean = D.t_mean.astype(np.float32)
    econ_floor = (1e-2 * D.t_mean + 1e-12).astype(np.float64)
    if spec["norm"] == "zscore":
        loc, scale = tmean, D.t_std.astype(np.float32)
        ytr_np = (D.y_tr.astype(np.float32) - loc) / scale
    else:
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
            return torch.exp(pn.clamp(max=15.0))
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
            var = np.maximum(var, econ_floor)
        return np.maximum(var, TINY)

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
    return raw_pred(D.X_te, D.nmask_te)   # [n_rows, N] full grid


def _flat(D):
    m = D.tmask_te.astype(bool)
    rows, _ = np.where(m)
    dates = np.array([D.d_te[r] for r in rows])
    return m, D.y_te[m], dates


def _metrics_vec(y, p):
    return dict(mse=M.mse(y, p), rmse=M.rmse(y, p), mae=M.mae(y, p), qlike=M.qlike(y, p, QF), r2=M.r2(y, p))


def main():
    ds = sys.argv[1]
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    cfg = replace(Config(), batch_size=32)
    fmap = {"vn30": [REPO / "submission/soict_lstm_gat/data/vn30/*_processed.csv"],
            "vn100": [REPO / "submission/soict_lstm_gat/data/vn100/*_processed.csv"]}
    pmap = {"vn30": REPO / "data/raw/prices", "vn100": REPO / "data/raw/prices/vn100_vnstock"}
    files = next((glob.glob(str(p)) for p in fmap[ds] if glob.glob(str(p))), [])
    price_dir = str(pmap[ds])
    print(f"building {ds} h{h} panel ...", flush=True)
    D = MR.build_masked_rich(files, price_dir, cfg.lookback, h)
    m, y, dates = _flat(D)
    print(f"N={D.N} test_obs={len(y)} dates={len(set(dates))}", flush=True)

    # HAR reference (deterministic)
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    har = np.maximum(B.har_predict(D.har_te.reshape(-1, 3), coef, QF).reshape(D.y_te.shape), 1e-2 * D.t_mean + 1e-12)
    har_v = har[m]
    print("HAR ref: " + " ".join(f"{k}={v:.6e}" if k != "qlike" and k != "r2" else f"{k}={v:.4f}"
                                 for k, v in _metrics_vec(y, har_v).items()), flush=True)

    out = {"dataset": ds, "horizon": h, "n_obs": int(len(y)), "n_dates": int(len(set(dates))),
           "seeds": list(SEEDS), "har": _metrics_vec(y, har_v), "configs": {}}
    sel = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CONFIGS)
    ens = {}
    for name in sel:
        spec = CONFIGS[name]
        per_seed, preds = [], []
        for s in SEEDS:
            pv = _train(D, cfg, s, D.adj_vol2pk, spec)[m]
            preds.append(pv)
            per_seed.append(_metrics_vec(y, pv))
            print(f"{ds} {name} seed{s}: mse={per_seed[-1]['mse']:.6e} qlike={per_seed[-1]['qlike']:.4f} "
                  f"r2={per_seed[-1]['r2']:.4f}", flush=True)
        ensemble = np.mean(preds, axis=0)
        ens[name] = ensemble
        em = _metrics_vec(y, ensemble)
        agg = {k: {"mean": float(np.mean([d[k] for d in per_seed])),
                   "std": float(np.std([d[k] for d in per_seed]))} for k in per_seed[0]}
        out["configs"][name] = {"per_seed_mean_std": agg, "ensemble": em}
        print(f"  -> {name} ENSEMBLE: mse={em['mse']:.6e} rmse={em['rmse']:.6e} mae={em['mae']:.6e} "
              f"qlike={em['qlike']:.4f} r2={em['r2']:.4f}")
        print(f"  -> {name} per-seed QLIKE mean={agg['qlike']['mean']:.4f} std={agg['qlike']['std']:.4f} | "
              f"MSE mean={agg['mse']['mean']:.6e} std={agg['mse']['std']:.6e}\n", flush=True)

    # date-clustered DM on per-obs QLIKE (ensembles): D vs HAR, C vs D
    def dm(a, b):
        la = M.per_obs_qlike(y, a, QF)
        lb = M.per_obs_qlike(y, b, QF)
        r = ST.date_clustered_dm(la, lb, dates, h)
        return {"p_value": r["p_value"], "mean_diff": r["mean_diff"],
                "favors": "A" if r["mean_diff"] < 0 else "B"}
    out["dm"] = {"D_vs_HAR": dm(ens["D_ratio_softplus"], har_v),
                 "C_vs_D": dm(ens["C_ratio_exp"], ens["D_ratio_softplus"])}
    print("DM (date-clustered, QLIKE, ensembles):")
    print(f"  D vs HAR : p={out['dm']['D_vs_HAR']['p_value']:.4f} favors={out['dm']['D_vs_HAR']['favors']} "
          f"(A=D better, B=HAR better)")
    print(f"  C vs D   : p={out['dm']['C_vs_D']['p_value']:.4f} favors={out['dm']['C_vs_D']['favors']} "
          f"(A=C better, B=D better)")

    outp = REPO / "results" / "ablation_vn_5seed"
    outp.mkdir(parents=True, exist_ok=True)
    (outp / f"{ds}_h{h}_bm.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {outp / f'{ds}_h{h}_bm.json'}", flush=True)


if __name__ == "__main__":
    main()
