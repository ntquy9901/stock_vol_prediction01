"""DY (2014) spillover-edge ablation for the LSTM+GAT volatility model (HNX primary).

Compares FOUR edge choices under the SAME MaskedRichNet / HAR-X pipeline on a panel:
  * dy_GAT        : use_graph=True, adj = DY (2014) generalized-FEVD connectedness (this contribution)
  * stat_GAT_vol2pk: use_graph=True, adj = D.adj_vol2pk (shipped directed volume-shock->PK edge)
  * no_graph_LSTM : use_graph=False
  * (sector-GAT is compared from results/sector_gat_ablation/ if that run exists -- not re-trained here)

The DY connectedness matrix is a FIXED (frozen-on-train) directed weighted adjacency estimated on the
TRAIN Parkinson-variance panel only (elastic-net VAR(1) -> generalized FEVD -> row-normalise -> Top-K
sources + self-loop=1.0). It plugs into WeightedGATLayer exactly like the sector / vol2pk adjacency, so
only the EDGE differs (LSTM branch, 5 node features, masked panel, HAR-X anchor, per-ticker scalers and
QLIKE evaluation are identical).

All new code lives here; live-training-path files are IMPORTED read-only (never edited):
  estimator_forecast_ablation (processed writer + screened universe), masked_rich (panel build),
  run_masked_rich (MaskedRichNet + train_masked_rich), config (Config/SMOKE), stats (date-clustered DM).

CPU IS FORCED by default (``DY_ABLATION_FORCE_CPU=1``): CUDA is hidden BEFORE torch import so a run here
never contends with the live GPU jobs. Set ``DY_ABLATION_FORCE_CPU=0`` to allow GPU once it is free.
Default mode is DRY/SMOKE (build adj + one forward pass); ``--train-epochs N`` runs the comparison.

Dry run:   python run_dy_ablation.py --panel hnx --horizon 1 --max-tickers 12
CPU train: python run_dy_ablation.py --panel hnx --horizon 1 --train-epochs 10 --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Force CPU BEFORE importing torch so torch.cuda.is_available() (inside train_masked_rich, a file we must
# not edit) returns False -> the ablation stays off the live-GPU jobs.
if os.environ.get("DY_ABLATION_FORCE_CPU", "1") == "1":  # pragma: no cover (import-time GPU guard)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "quality_gate",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only import)
import masked_rich as MR                   # noqa: E402  (read-only import)
import run_masked_rich as RMR              # noqa: E402  (read-only import)
import volatility_estimators as VE         # noqa: E402  (read-only import)
from config import Config, SMOKE           # noqa: E402  (read-only import)

import dy_connectedness as DY              # noqa: E402

SECTOR_RESULT = REPO / "results" / "sector_gat_ablation" / "sector_ablation_hnx_h1.json"


def build_panel_masked(panel: str, cfg: Config, horizon: int, out_dir: str, keep_tickers=None):
    """Write the panel's Parkinson processed CSVs (read-only estimator writer) and build the masked panel."""
    files = EFA._write_estimator_processed(panel, "parkinson", out_dir, keep_tickers=keep_tickers)
    if len(files) < 2:
        raise RuntimeError(f"{panel} processed build produced {len(files)} files (<2)")
    price_dir = VE.PRICE[panel]
    D = MR.build_masked_rich(files, str(price_dir), cfg.lookback, horizon,
                             edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)
    return D, files


def dy_adj_for(D, files, p=1, H=10, alpha=0.05, l1_ratio=0.5, top_k=MR.EDGE_TOP_K):
    """DY (2014) connectedness adjacency aligned to ``D.tickers`` + the connectedness-stats dict.

    Uses the TRAIN volatility panel only (rows strictly before the first validation target ``D.d_va[0]``).
    """
    train_panel = DY.train_vol_panel(files, D.tickers, D.d_va[0])
    adj, _theta, stats = DY.build_dy_adjacency(train_panel, p=p, H=H, alpha=alpha,
                                               l1_ratio=l1_ratio, top_k=top_k)
    if adj.shape != (D.N, D.N):
        raise RuntimeError(f"DY adjacency {adj.shape} != (N,N)=({D.N},{D.N})")
    return adj, stats


def forward_pass_smoke(D, adj, batch=2):
    """ONE CPU forward pass of MaskedRichNet(use_graph=True) given the DY adjacency. Returns [b,N]."""
    b = int(min(batch, len(D.X_te)))
    if b < 1:
        raise RuntimeError("empty test split -- cannot run forward-pass smoke")
    net = RMR.MaskedRichNet(64, 4, 0.2, use_graph=True).eval()   # CPU
    xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))
    nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float()
    base = torch.from_numpy(np.ascontiguousarray(adj)).float()
    adj_b = base.unsqueeze(0) * nmb.unsqueeze(1)
    with torch.no_grad():
        out = net(xb, adj_b)
    out_np = out.numpy()
    if not np.isfinite(out_np).all():
        raise RuntimeError("MaskedRichNet produced non-finite output on the DY adjacency")
    return out_np


def _train_variant(D, cfg, use_graph, adj):
    """Seed-ensembled TEST prediction dict for one edge choice (run_masked_rich conventions)."""
    seed_dicts = [RMR._pred_dict(RMR.train_masked_rich(D, cfg, s, use_graph, adj),
                                 D.y_te, D.tmask_te, D.d_te, D.N) for s in cfg.seeds]
    return RMR._ens(seed_dicts), seed_dicts


def _load_sector_dm(dy_pred, horizon, floor):
    """If the sector-GAT ablation result exists, DM DY-GAT vs the RECORDED sector-GAT is not possible
    (different training run / no stored per-obs preds), so we only surface the sector-GAT ensemble QLIKE
    for context. Returns None when the file is absent."""
    if not SECTOR_RESULT.exists():
        return None
    d = json.loads(SECTOR_RESULT.read_text())
    return {"sector_GAT_qlike": d["metrics_ensemble"]["sector_GAT"]["qlike"],
            "sector_GAT_metrics": d["metrics_ensemble"]["sector_GAT"],
            "note": "sector-GAT is from results/sector_gat_ablation (separate run); QLIKE shown for "
                    "context, no per-obs DM (predictions not stored)."}


def run_training(panel, cfg, horizon, out_dir=None, p=1, H=10, alpha=0.05, l1_ratio=0.5):
    """10-epoch comparison: dy-GAT vs stat-GAT vs no-graph LSTM on all 5 metrics + date-clustered DM.

    CPU-forced by default. A directional viability check on HNX h1."""
    with tempfile.TemporaryDirectory() as td:
        D, files = build_panel_masked(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        adj_dy, dy_stats = dy_adj_for(D, files, p=p, H=H, alpha=alpha, l1_ratio=l1_ratio)
        dy_pred, dy_seeds = _train_variant(D, cfg, True, adj_dy)
        stat_pred, stat_seeds = _train_variant(D, cfg, True, D.adj_vol2pk)
        lstm_pred, lstm_seeds = _train_variant(D, cfg, False, D.adj_vol2pk)
        fl = cfg.qlike_floor
        preds = {"dy_GAT": dy_pred, "stat_GAT_vol2pk": stat_pred, "no_graph_LSTM": lstm_pred}
        seedmap = {"dy_GAT": dy_seeds, "stat_GAT_vol2pk": stat_seeds, "no_graph_LSTM": lstm_seeds}
        metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seedmap.items()}
        dm = {
            "dy_vs_no_graph": RMR._dm_all(dy_pred, lstm_pred, horizon, fl),
            "dy_vs_stat": RMR._dm_all(dy_pred, stat_pred, horizon, fl),
            "stat_vs_no_graph": RMR._dm_all(stat_pred, lstm_pred, horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon, "design": "dy-spillover-graph-ablation",
            "edge": "DY-2014 generalized-FEVD connectedness (elastic-net VAR)",
            "device": "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "gpu",
            "num_nodes": D.N, "n_test_obs": metrics["no_graph_LSTM"]["n"],
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "dy_connectedness": dy_stats,
            "metrics_ensemble": metrics, "metrics_per_seed": per_seed, "dm": dm,
            "sector_gat_context": _load_sector_dm(dy_pred, horizon, fl),
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"dy_ablation_{panel}_h{horizon}.json").write_text(json.dumps(res, indent=2, default=float))
        return res


def run_dry(panel, horizon, max_tickers, p=1, H=10, alpha=0.05, l1_ratio=0.5):
    """Build the DY adjacency + one CPU forward pass. No training. Prints a summary."""
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D, files = build_panel_masked(panel, cfg, horizon, td, keep_tickers=keep)
        adj_dy, stats = dy_adj_for(D, files, p=p, H=H, alpha=alpha, l1_ratio=l1_ratio)
        assert adj_dy.shape == (D.N, D.N), (adj_dy.shape, D.N)
        out = forward_pass_smoke(D, adj_dy, batch=2)
    off = adj_dy.copy(); np.fill_diagonal(off, 0.0)
    avg_deg = float((off > 0).sum(axis=1).mean())
    print(f"[dry] {panel} nodes={D.N}  DY total-connectedness={stats['total_connectedness_index']:.1f}%  "
          f"row_sum_mean={stats['row_sum_mean']:.4f}  avg_off_degree={avg_deg:.2f}  "
          f"own_share={stats['diag_mean_own_share']:.3f}  asym={stats['asymmetry_frob']:.3f}")
    print(f"[dry] forward pass OK: output shape {out.shape}, finite={np.isfinite(out).all()}")
    return {"n_nodes": D.N, "stats": stats, "avg_off_degree": avg_deg, "forward_shape": list(out.shape)}


def main():
    ap = argparse.ArgumentParser(description="DY (2014) spillover-edge ablation (CPU-forced by default).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12,
                    help="dry mode: cap universe so the CPU forward-pass smoke stays fast")
    ap.add_argument("--var-lag", type=int, default=1, help="VAR lag p (Demirer et al. high-dim: 1 or 2)")
    ap.add_argument("--fevd-h", type=int, default=10, help="FEVD forecast horizon H (DY use 10)")
    ap.add_argument("--alpha", type=float, default=0.05, help="elastic-net penalty")
    ap.add_argument("--l1-ratio", type=float, default=0.5, help="elastic-net l1_ratio")
    ap.add_argument("--train-epochs", type=int, default=None,
                    help="TRAIN mode: run the N-epoch CPU comparison (unset -> dry mode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "dy_spillover_ablation"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers, a.var_lag, a.fevd_h, a.alpha, a.l1_ratio)
        return
    base = Config()
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = Config(epochs=a.train_epochs, patience=base.patience,
                 min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds)
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "gpu"
    print(f"[train:{dev}] {a.panel} DY ablation h{a.horizon}, {cfg.epochs} epochs, seeds={cfg.seeds}, "
          f"VAR({a.var_lag}) H={a.fevd_h} alpha={a.alpha}")
    res = run_training(a.panel, cfg, a.horizon, out_dir=a.out_dir,
                       p=a.var_lag, H=a.fevd_h, alpha=a.alpha, l1_ratio=a.l1_ratio)
    print(f"  {'model':18} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics_ensemble"].items():
        print(f"  {k:18} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:20} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
