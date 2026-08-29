"""MTGNN learned-adjacency ablation for the LSTM+GAT volatility model (panel-agnostic; HNX primary).

Compares FOUR edge choices under the SAME MaskedRichNet / HAR-X pipeline and folds, so ONLY the edge
mechanism differs:
  * no_graph_LSTM            : use_graph=False                       (temporal branch only)
  * stat_GAT_vol2pk          : use_graph=True, adj = D.adj_vol2pk    (shipped directed volume-shock->PK)
  * sector_GAT               : use_graph=True, adj = static sector   (2026-08-29_sector_gat_ablation)
  * learned_GAT_mtgnn        : adjacency BUILT each forward by the MTGNN GraphConstructor (trainable)

The three fixed-edge variants reuse the delivered ``run_masked_rich.train_masked_rich`` unchanged; the
learned variant needs its own loop (its adjacency is a trainable module inside the net, not a numpy
base) -- ``train_learned`` mirrors the delivered ``zscore_floor`` training path exactly (same optimizer,
ReduceLROnPlateau, per-node scaler, masked-MSE loss, early stopping, split/​curve capture) so the only
difference is where the adjacency comes from.

All new code lives in this baseline folder; live-training-path files are IMPORTED read-only.

CPU is FORCED by default (``LEARNED_ABLATION_FORCE_CPU=1``) so a run never contends with the live GPU
job. Set ``LEARNED_ABLATION_FORCE_CPU=0`` to allow GPU once it is free.

Dry:   python run_learned_ablation.py --panel hnx --horizon 1 --max-tickers 12
Train: python run_learned_ablation.py --panel hnx --horizon 1 --train-epochs 10 --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if os.environ.get("LEARNED_ABLATION_FORCE_CPU", "1") == "1":  # pragma: no branch - always CPU under tests
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # HARD-set BEFORE torch import -> stays off the live GPU

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-08-29_sector_gat_ablation" / "code",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only)
import masked_rich as MR                   # noqa: E402  (read-only)
import run_masked_rich as RMR              # noqa: E402  (read-only)
import volatility_estimators as VE         # noqa: E402  (read-only)
from config import Config, SMOKE           # noqa: E402  (read-only)
from sector_adjacency import build_sector_adjacency, load_sector_map  # noqa: E402  (read-only)

from mtgnn_graph import LearnedGraphNet     # noqa: E402  (this baseline)

VN_SECTOR_CSV = REPO / "baselines" / "2026-08-29_sector_gat_ablation" / "vn_sectors.csv"
SP500_SECTOR_CSV = REPO / "baselines" / "2026-08-29_sector_gat_ablation" / "sp500_gics_sectors.csv"
# gate keys: LEARNED = ("LSTM", "LSTM_wGAT_vol2pk") MUST be present with fit evidence
NO_GRAPH, STAT, SECTOR, LEARNED = "LSTM", "LSTM_wGAT_vol2pk", "LSTM_wGAT_sector", "LSTM_wGAT_learned_mtgnn"


def default_sector_csv(panel: str) -> Path:
    return SP500_SECTOR_CSV if panel == "sp500" else VN_SECTOR_CSV


def build_panel(panel, cfg, horizon, out_dir, keep_tickers=None):
    files = EFA._write_estimator_processed(panel, "parkinson", out_dir, keep_tickers=keep_tickers)
    if len(files) < 2:  # pragma: no cover - defensive guard on an empty/broken panel build
        raise RuntimeError(f"{panel} processed build produced {len(files)} files (<2)")
    D = MR.build_masked_rich(files, str(VE.PRICE[panel]), cfg.lookback, horizon,
                             edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)
    return D


def sector_adj(tickers, sector_csv):
    return build_sector_adjacency(list(tickers), load_sector_map(sector_csv)).astype(np.float32)


def train_learned(D, cfg, seed, subgraph_size=20, node_dim=40, alpha=3.0, return_splits=False):
    """Train the MTGNN-learned-adjacency net; mirrors ``train_masked_rich``'s ``zscore_floor`` path.

    Difference from the delivered loop: the net is ``LearnedGraphNet`` and its ``forward(x, nmask)`` builds
    the adjacency internally from the trainable ``GraphConstructor`` (so no numpy ``adj`` / ``adj_batch``).
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = LearnedGraphNet(D.N, cfg.hidden, cfg.heads, cfg.dropout, subgraph_size, node_dim, alpha).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).float().to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    ytr_n = (torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd
    bs = cfg.batch_size

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
    if best_state:  # pragma: no branch - epoch 0 always improves from inf, so best_state is always set
        net.load_state_dict(best_state)
    te = infer(D.X_te, D.nmask_te)
    if return_splits:
        return {"test": te, "val": infer(D.X_va, D.nmask_va), "train": infer(D.X_tr, D.nmask_tr),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te


def _splits_for(D, cfg, seed, use_graph, adj):
    """Fixed-edge variant via the delivered loop, returning the split dict (train/val/test + curves)."""
    return RMR.train_masked_rich(D, cfg, seed, use_graph, adj, output_param="zscore_floor",
                                 return_splits=True)


def _pred(D, split):
    return RMR._pred_dict(split["test"], D.y_te, D.tmask_te, D.d_te, D.N)


def run_training(panel, cfg, horizon, subgraph_size, node_dim, alpha, sector_csv=None, out_dir=None):
    sector_csv = sector_csv or default_sector_csv(panel)
    fl = cfg.qlike_floor
    with tempfile.TemporaryDirectory() as td:
        D = build_panel(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        adj_sec = sector_adj(D.tickers, sector_csv)
        # per-seed split dicts for every variant (same seeds, same folds)
        splits = {NO_GRAPH: [], STAT: [], SECTOR: [], LEARNED: []}
        for s in cfg.seeds:
            splits[NO_GRAPH].append(_splits_for(D, cfg, s, False, D.adj_vol2pk))
            splits[STAT].append(_splits_for(D, cfg, s, True, D.adj_vol2pk))
            splits[SECTOR].append(_splits_for(D, cfg, s, True, adj_sec))
            splits[LEARNED].append(train_learned(D, cfg, s, subgraph_size, node_dim, alpha,
                                                 return_splits=True))
            print(f"  [seed {s}] done (N={D.N})", flush=True)  # pragma: no cover - progress log only
        seed_dicts = {k: [_pred(D, sp) for sp in v] for k, v in splits.items()}
        ens = {k: RMR._ens(v) for k, v in seed_dicts.items()}
        metrics = {k: RMR._metrics(v, fl) for k, v in ens.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seed_dicts.items()}
        # over/under-fit evidence: seed-ensembled train/val/test metrics + verdict + learning curves
        tr = {k: RMR._split_metrics(RMR._ens_split(v, "train"), D.y_tr, D.tmask_tr, fl) for k, v in splits.items()}
        va = {k: RMR._split_metrics(RMR._ens_split(v, "val"), D.y_va, D.tmask_va, fl) for k, v in splits.items()}
        fit = {k: RMR.OF.classify_fit(tr[k], va[k], metrics[k]) for k in splits}
        curves = {k: {"train": [o["train_curve"] for o in v], "val": [o["val_curve"] for o in v],
                      "best_epoch": [o["best_epoch"] for o in v]} for k, v in splits.items()}
        dm = {
            "learned_vs_no_graph": RMR._dm_all(ens[LEARNED], ens[NO_GRAPH], horizon, fl),
            "learned_vs_stat_vol2pk": RMR._dm_all(ens[LEARNED], ens[STAT], horizon, fl),
            "learned_vs_sector": RMR._dm_all(ens[LEARNED], ens[SECTOR], horizon, fl),
            "stat_vs_no_graph": RMR._dm_all(ens[STAT], ens[NO_GRAPH], horizon, fl),
            "sector_vs_no_graph": RMR._dm_all(ens[SECTOR], ens[NO_GRAPH], horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon, "design": "masked-union-panel learned-graph (MTGNN) ablation",
            "device": "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "gpu",
            "num_nodes": D.N, "n_test_obs": metrics[NO_GRAPH]["n"],
            "n_test_dates": len(set(k[1] for k in ens[NO_GRAPH])),
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "mtgnn": {"subgraph_size_k": min(subgraph_size, D.N), "node_dim": node_dim, "alpha": alpha,
                      "reference": "Wu et al. 2020 arXiv:2005.11650 Eq.(1-3),(5-6); nnzhan/MTGNN graph_constructor"},
            "metrics": metrics, "metrics_per_seed": per_seed,
            "train_metrics": tr, "val_metrics": va, "fit_diagnostics": fit, "learning_curves": curves,
            "dm_date_clustered": dm,
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"learned_graph_ablation_{panel}_h{horizon}.json").write_text(
                json.dumps(res, indent=2, default=float), encoding="utf-8")
        return res


def run_dry(panel, horizon, max_tickers, subgraph_size, node_dim, alpha):
    """Build the panel + ONE CPU forward pass of the learned-graph net. No training."""
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D = build_panel(panel, cfg, horizon, td, keep_tickers=keep)
        net = LearnedGraphNet(D.N, cfg.hidden, cfg.heads, cfg.dropout, subgraph_size, node_dim, alpha).eval()
        b = int(min(2, len(D.X_te)))
        xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))
        nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float()
        with torch.no_grad():
            out = net(xb, nmb).numpy()
            A = net.learned_adjacency().numpy()
    off = A.copy(); np.fill_diagonal(off, 0.0)
    if not np.isfinite(out).all():  # pragma: no cover - defensive: a finite forward is asserted by tests
        raise RuntimeError("LearnedGraphNet produced non-finite output")
    print(f"[dry] {panel} nodes={D.N} k={min(subgraph_size, D.N)} forward={out.shape} finite={np.isfinite(out).all()} "
          f"| A: max_off_out_degree={int((off > 0).sum(1).max())} asymmetric={not np.allclose(A, A.T)} "
          f"selfloop_diag_min={A.diagonal().min():.3f}")
    return {"n_nodes": D.N, "forward_shape": list(out.shape)}


def main():  # pragma: no cover - CLI entry driver
    ap = argparse.ArgumentParser(description="MTGNN learned-graph ablation (CPU-forced by default).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12)
    ap.add_argument("--subgraph-size", type=int, default=20, help="MTGNN top-k neighbours per node")
    ap.add_argument("--node-dim", type=int, default=40, help="MTGNN node-embedding dim d")
    ap.add_argument("--alpha", type=float, default=3.0, help="MTGNN saturation rate (paper default 3)")
    ap.add_argument("--sector-csv", default=None)
    ap.add_argument("--train-epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "learned_graph_ablation"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers, a.subgraph_size, a.node_dim, a.alpha)
        return
    base = Config()
    from dataclasses import replace
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = replace(base, epochs=a.train_epochs, patience=base.patience,
                  min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds,
                  batch_size=a.batch or base.batch_size)
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "gpu"
    print(f"[train:{dev}] {a.panel} learned-graph ablation h{a.horizon}, {cfg.epochs} epochs, "
          f"seeds={cfg.seeds}, k={a.subgraph_size}, d={a.node_dim}, bs={cfg.batch_size}")
    res = run_training(a.panel, cfg, a.horizon, a.subgraph_size, a.node_dim, a.alpha,
                       a.sector_csv, out_dir=a.out_dir)
    print(f"  {'model':26} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics"].items():
        print(f"  {k:26} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm_date_clustered"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:26} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
