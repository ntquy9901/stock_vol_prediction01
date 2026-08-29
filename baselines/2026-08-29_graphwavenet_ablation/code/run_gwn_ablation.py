"""Graph WaveNet ablation on the masked volatility panel (HNX primary; panel-agnostic).

Trains, on the SAME masked-union folds / seeds, six models so ONLY the model differs:
  * HAR / HAR-X          : deterministic linear anchors (delivered basis)
  * LSTM                 : no-graph LSTM (delivered ``train_masked_rich`` use_graph=False) -- DM reference
  * LSTM_wGAT_vol2pk     : LSTM + weighted-GAT on the directed vol->PK edge (prior-art null)
  * GWN_adaptive         : Graph WaveNet WITH the self-adaptive adjacency (arXiv:1906.00121)
  * GWN_no_adaptive      : Graph WaveNet WITHOUT graph conv (pure TCN; adaptive graph removed)

``GWN_adaptive vs GWN_no_adaptive`` is the clean in-family adaptive-graph ablation (TCN held fixed);
``GWN_* vs LSTM/HAR`` is a TEMPORAL-BACKBONE comparison (GWN replaces the LSTM), reported as such.

All new code is in this baseline folder; live-training-path files are IMPORTED read-only. ``train_gwn``
mirrors the delivered ``train_masked_rich`` ``zscore_floor`` path (same optimizer, ReduceLROnPlateau,
per-node scaler, masked-MSE loss, early stopping, split/curve capture) -- only the network differs.

GPU is ALLOWED by default; set ``GWN_FORCE_CPU=1`` to force CPU.

Dry:   python run_gwn_ablation.py --panel hnx --horizon 1 --max-tickers 12
Train: python run_gwn_ablation.py --panel hnx --horizon 1 --train-epochs 10 --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if os.environ.get("GWN_FORCE_CPU", "0") == "1":  # pragma: no branch - honoured before torch import
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only)
import masked_rich as MR                   # noqa: E402  (read-only)
import run_masked_rich as RMR              # noqa: E402  (read-only)
import volatility_estimators as VE         # noqa: E402  (read-only)
from config import Config, SMOKE           # noqa: E402  (read-only)

from gwn_model import GraphWaveNet         # noqa: E402  (this baseline)

NO_GRAPH, STAT = "LSTM", "LSTM_wGAT_vol2pk"        # gate keys (OF.LEARNED) -- present with real evidence
GWN_A, GWN_N = "GWN_adaptive", "GWN_no_adaptive"
LEARNED_KEYS = (NO_GRAPH, STAT, GWN_A, GWN_N)


def build_panel(panel, cfg, horizon, out_dir, keep_tickers=None):
    files = EFA._write_estimator_processed(panel, "parkinson", out_dir, keep_tickers=keep_tickers)
    if len(files) < 2:  # pragma: no cover - defensive guard on an empty/broken panel build
        raise RuntimeError(f"{panel} processed build produced {len(files)} files (<2)")
    return MR.build_masked_rich(files, str(VE.PRICE[panel]), cfg.lookback, horizon,
                                edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)


def train_gwn(D, cfg, seed, adaptive, bs, skip_channels=64, end_channels=128,
              residual_channels=32, node_dim=10, return_splits=False):
    """Train Graph WaveNet (``adaptive`` toggles the self-adaptive graph). Mirrors ``train_masked_rich``
    zscore_floor: standardized target, masked-MSE, Adam+plateau, early stop, denorm with the 1e-2*mean floor."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = GraphWaveNet(D.N, in_dim=MR.N_FEAT, out_dim=1, residual_channels=residual_channels,
                       dilation_channels=residual_channels, skip_channels=skip_channels,
                       end_channels=end_channels, dropout=cfg.dropout, adaptive=adaptive,
                       node_dim=node_dim).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).float().to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    ytr_n = (torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd

    def infer(X_np, nm_np):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).float().to(dev)
                outs.append(net(xb, nmb).cpu().numpy())
        pn = np.concatenate(outs)
        return np.maximum(pn * D.t_std + D.t_mean, 1e-2 * D.t_mean + 1e-12)

    best = np.inf; best_state = None; wait = 0; best_ep = 0
    train_curve = []; val_curve = []
    for ep in range(cfg.epochs):
        net.train()
        for idx in RMR._batches(len(Xtr), bs, True, seed + ep):
            xb = Xtr[idx]; nmb = nmtr[idx]; tmb = tmtr[idx]; yb = ytr_n[idx]
            opt.zero_grad(); pred = net(xb, nmb)
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer(D.X_va, D.nmask_va); m = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[m] - D.y_va[m]) ** 2))
        ptr = infer(D.X_tr, D.nmask_tr); mtr = D.tmask_tr.astype(bool)
        train_curve.append(float(np.mean((ptr[mtr] - D.y_tr[mtr]) ** 2)))
        val_curve.append(vmse); sched.step(vmse)
        if vmse < best - 1e-12:
            best = vmse; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0; best_ep = ep + 1
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:  # pragma: no branch - epoch 0 always improves from inf
        net.load_state_dict(best_state)
    te = infer(D.X_te, D.nmask_te)
    if return_splits:
        return {"test": te, "val": infer(D.X_va, D.nmask_va), "train": infer(D.X_tr, D.nmask_tr),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te


def _har_context(D, cfg):
    """Deterministic HAR + HAR-X anchors on the SAME folds (test pred dicts + train/val split metrics)."""
    B = RMR.B; fl = cfg.qlike_floor; nfloor = 1e-2 * D.t_mean + 1e-12
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])

    def _har(block):
        return np.maximum(B.har_predict(block.reshape(-1, 3), coef, floor=fl).reshape(-1, D.N), nfloor)
    hp_te = _har(D.har_te).reshape(D.y_te.shape)
    _xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    cx = np.linalg.lstsq(_xtr, D.y_tr[mtr], rcond=None)[0]

    def _harx(block):
        flat = block.reshape(-1, 5)
        return np.maximum((np.column_stack([np.ones(len(flat)), flat]) @ cx).reshape(-1, D.N), nfloor)
    hx_te = _harx(D.har5_te).reshape(D.y_te.shape)
    HAR = RMR._pred_dict(hp_te, D.y_te, D.tmask_te, D.d_te, D.N)
    HARX = RMR._pred_dict(hx_te, D.y_te, D.tmask_te, D.d_te, D.N)
    tr = {"HAR": _har(D.har_tr).reshape(D.y_tr.shape), "HAR-X": _harx(D.har5_tr).reshape(D.y_tr.shape)}
    va = {"HAR": _har(D.har_va).reshape(D.y_va.shape), "HAR-X": _harx(D.har5_va).reshape(D.y_va.shape)}
    return {"HAR": HAR, "HAR-X": HARX}, tr, va


def run_training(panel, cfg, horizon, gwn_batch=64, skip_channels=64, end_channels=128, out_dir=None):
    fl = cfg.qlike_floor
    with tempfile.TemporaryDirectory() as td:
        D = build_panel(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        splits = {NO_GRAPH: [], STAT: [], GWN_A: [], GWN_N: []}
        for s in cfg.seeds:
            splits[NO_GRAPH].append(RMR.train_masked_rich(D, cfg, s, False, D.adj_vol2pk,
                                                          output_param="zscore_floor", return_splits=True))
            splits[STAT].append(RMR.train_masked_rich(D, cfg, s, True, D.adj_vol2pk,
                                                      output_param="zscore_floor", return_splits=True))
            splits[GWN_A].append(train_gwn(D, cfg, s, True, gwn_batch, skip_channels, end_channels,
                                           return_splits=True))
            splits[GWN_N].append(train_gwn(D, cfg, s, False, gwn_batch, skip_channels, end_channels,
                                           return_splits=True))
            print(f"  [seed {s}] done (N={D.N})", flush=True)  # pragma: no cover - progress log only
        seed_dicts = {k: [RMR._pred_dict(sp["test"], D.y_te, D.tmask_te, D.d_te, D.N) for sp in v]
                      for k, v in splits.items()}
        ens = {k: RMR._ens(v) for k, v in seed_dicts.items()}
        har_preds, har_tr, har_va = _har_context(D, cfg)
        ens.update(har_preds)                                   # HAR/HAR-X are deterministic -> single dict
        metrics = {k: RMR._metrics(v, fl) for k, v in ens.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seed_dicts.items()}
        # over/under-fit evidence: seed-ensembled train/val/test metrics + verdict + learning curves
        tr = {k: RMR._split_metrics(RMR._ens_split(v, "train"), D.y_tr, D.tmask_tr, fl) for k, v in splits.items()}
        va = {k: RMR._split_metrics(RMR._ens_split(v, "val"), D.y_va, D.tmask_va, fl) for k, v in splits.items()}
        for k in ("HAR", "HAR-X"):
            tr[k] = RMR._split_metrics(har_tr[k], D.y_tr, D.tmask_tr, fl)
            va[k] = RMR._split_metrics(har_va[k], D.y_va, D.tmask_va, fl)
        fit = {k: RMR.OF.classify_fit(tr[k], va[k], metrics[k]) for k in tr}
        curves = {k: {"train": [o["train_curve"] for o in v], "val": [o["val_curve"] for o in v],
                      "best_epoch": [o["best_epoch"] for o in v]} for k, v in splits.items()}
        dm = {
            "GWN_adaptive_vs_GWN_no_adaptive": RMR._dm_all(ens[GWN_A], ens[GWN_N], horizon, fl),
            "GWN_adaptive_vs_LSTM": RMR._dm_all(ens[GWN_A], ens[NO_GRAPH], horizon, fl),
            "GWN_no_adaptive_vs_LSTM": RMR._dm_all(ens[GWN_N], ens[NO_GRAPH], horizon, fl),
            "GWN_adaptive_vs_HAR": RMR._dm_all(ens[GWN_A], ens["HAR"], horizon, fl),
            "GWN_adaptive_vs_HARX": RMR._dm_all(ens[GWN_A], ens["HAR-X"], horizon, fl),
            "GWN_no_adaptive_vs_HAR": RMR._dm_all(ens[GWN_N], ens["HAR"], horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon,
            "design": "masked-union-panel Graph-WaveNet backbone ablation (TCN + self-adaptive graph)",
            "device": "gpu" if torch.cuda.is_available() else "cpu",
            "num_nodes": D.N, "n_test_obs": metrics[NO_GRAPH]["n"],
            "n_test_dates": len(set(k[1] for k in ens[NO_GRAPH])),
            # valid-node fraction bounds the BatchNorm-over-zero-padded-nodes caveat (see gwn_model docstring):
            # scored obs / (N * test dates). Common-mode across GWN variants; reported for the cross-family DM.
            "valid_node_fraction_test": round(
                metrics[NO_GRAPH]["n"] / (D.N * max(len(set(k[1] for k in ens[NO_GRAPH])), 1)), 4),
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "graphwavenet": {"blocks": 4, "layers": 2, "kernel_size": 2, "order": 2, "node_dim": 10,
                             "residual_channels": 32, "dilation_channels": 32,
                             "skip_channels": skip_channels, "end_channels": end_channels,
                             "gwn_batch": gwn_batch, "receptive_field": 13,
                             "reference": "Wu et al. IJCAI 2019 arXiv:1906.00121; nnzhan/Graph-WaveNet model.py",
                             "note": "adaptive-only graph (no predefined support); w/o-adaptive = no graph conv"},
            "metrics": metrics, "metrics_per_seed": per_seed,
            "train_metrics": tr, "val_metrics": va, "fit_diagnostics": fit, "learning_curves": curves,
            "dm_date_clustered": dm,
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"graphwavenet_ablation_{panel}_h{horizon}.json").write_text(
                json.dumps(res, indent=2, default=float), encoding="utf-8")
        return res


def run_dry(panel, horizon, max_tickers, gwn_batch=2):
    """Build the panel + ONE CPU forward of both GWN variants. No training."""
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D = build_panel(panel, cfg, horizon, td, keep_tickers=keep)
        b = int(min(gwn_batch, len(D.X_te)))
        xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))
        nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float()
        for adaptive in (True, False):
            net = GraphWaveNet(D.N, in_dim=MR.N_FEAT, adaptive=adaptive).eval()
            with torch.no_grad():
                out = net(xb, nmb).numpy()
            if not np.isfinite(out).all():  # pragma: no cover - defensive; a finite forward is asserted by tests
                raise RuntimeError(f"GraphWaveNet(adaptive={adaptive}) produced non-finite output")
            tag = "adaptive" if adaptive else "no_adaptive"
            extra = ""
            if adaptive:
                A = net.adaptive_adjacency().detach().numpy()
                extra = f" | A rowsum~1={np.allclose(A.sum(1), 1.0, atol=1e-5)} asym={not np.allclose(A, A.T)}"
            print(f"[dry] {panel} nodes={D.N} GWN_{tag} forward={out.shape} finite=True{extra}")
    return {"n_nodes": D.N}


def main():  # pragma: no cover - CLI entry driver
    ap = argparse.ArgumentParser(description="Graph WaveNet ablation (GPU-allowed; GWN_FORCE_CPU=1 forces CPU).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | hose | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12)
    ap.add_argument("--train-epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--gwn-batch", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16, help="LSTM/GAT batch (delivered scale; avoids VRAM thrash)")
    ap.add_argument("--skip-channels", type=int, default=64)
    ap.add_argument("--end-channels", type=int, default=128)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "graphwavenet_ablation"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers)
        return
    base = Config()
    from dataclasses import replace
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = replace(base, epochs=a.train_epochs, patience=base.patience,
                  min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds, batch_size=a.batch)
    dev = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[train:{dev}] {a.panel} Graph WaveNet ablation h{a.horizon}, {cfg.epochs} epochs, "
          f"seeds={cfg.seeds}, gwn_batch={a.gwn_batch}, skip={a.skip_channels}, end={a.end_channels}")
    res = run_training(a.panel, cfg, a.horizon, a.gwn_batch, a.skip_channels, a.end_channels, out_dir=a.out_dir)
    print(f"  {'model':22} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics"].items():
        print(f"  {k:22} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm_date_clustered"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:34} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
