"""CORR+LIFT graph-edge ablation for the LSTM+GAT volatility model (Sonani, Badii & Moin 2025 §3.2).

Sixth HNX edge probe. Compares three edge choices under the SAME MaskedRichNet / HAR-X pipeline:
  * corr_lift_GAT   : use_graph=True, adj = combined Pearson|rho|>0.7 OR lift>1.7 edge (this baseline)
  * stat_GAT_vol2pk : use_graph=True, adj = D.adj_vol2pk (shipped directed volume-shock->PK edge; context)
  * no_graph_LSTM   : use_graph=False

The corr+lift adjacency (``corrlift_edge.build_corrlift_adjacency``) is TRAIN-ONLY (close rows strictly
before D.d_va[0]) and frozen -- a clean leave-one-out edge swap vs the sector runner it mirrors
(``baselines/2026-08-29_sector_gat_ablation/code/run_sector_ablation.py``).

All new code lives here / in ``corrlift_edge.py``; the live-training-path files are IMPORTED read-only
(never edited): estimator_forecast_ablation (processed writer + screened universe), masked_rich (panel),
run_masked_rich (MaskedRichNet + train_masked_rich), volatility_estimators (PRICE dirs), config.

GPU is used when available (``.venv_gpu_encode`` torch 2.6.0). Set ``CORRLIFT_FORCE_CPU=1`` to hide CUDA
(e.g. to stay off a live GPU job). Default mode is DRY/SMOKE (build adj + one forward pass, no training);
``--train-epochs N`` runs the comparison.

Dry:   python run_corrlift_ablation.py --panel hnx --horizon 1 --max-tickers 12
Train: python run_corrlift_ablation.py --panel hnx --horizon 1 --train-epochs 10 --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if os.environ.get("CORRLIFT_FORCE_CPU", "0") == "1":  # pragma: no cover - import-time env side effect
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # HARD-set: hide the GPU before torch import when requested

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only import)
import masked_rich as MR                   # noqa: E402  (read-only import)
import run_masked_rich as RMR              # noqa: E402  (read-only import)
import volatility_estimators as VE         # noqa: E402  (read-only import)
from config import Config, SMOKE           # noqa: E402  (read-only import)

from corrlift_edge import build_corrlift_adjacency, load_close_wide  # noqa: E402


def _device_label():
    """Actual training device: torch.cuda.is_available() is the single source of truth (it already returns
    False when CORRLIFT_FORCE_CPU hid the GPU via CUDA_VISIBLE_DEVICES=''). Do NOT infer from the raw env var:
    an UNSET CUDA_VISIBLE_DEVICES still lets torch use the GPU, so an env-only check would mislabel it 'cpu'."""
    return "gpu" if torch.cuda.is_available() else "cpu"


def build_panel_masked(panel: str, cfg: Config, horizon: int, out_dir: str, keep_tickers=None):
    """Write the panel's Parkinson processed CSVs (read-only estimator writer) and build the masked panel."""
    files = EFA._write_estimator_processed(panel, "parkinson", out_dir, keep_tickers=keep_tickers)
    if len(files) < 2:
        raise RuntimeError(f"{panel} processed build produced {len(files)} files (<2)")
    price_dir = VE.PRICE[panel]
    D = MR.build_masked_rich(files, str(price_dir), cfg.lookback, horizon,
                             edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)
    return D, files


def corrlift_adj_for(D, price_dir):
    """Combined corr+lift adjacency aligned to ``D.tickers`` (TRAIN-only, frozen) + edge-density diag.

    Boundary = ``D.d_va[0]`` (first VALIDATION target date): every close row fed to the graph is strictly
    before every val/test target -> no evaluation leakage. This is marginally looser than the delivered
    ``adj_corr`` / ``adj_vol2pk`` cut (their last-TRAIN-target row excludes the ~horizon purge-gap rows); the
    difference is a handful of rows out of thousands and touches no val/test data.
    """
    if not D.d_va:
        raise RuntimeError("empty val split -- cannot locate the train/val boundary for the frozen graph")
    close_wide = load_close_wide(list(D.tickers), price_dir)
    cutoff = D.d_va[0]                                     # first val TARGET date -> strictly-before-val boundary
    adj, diag = build_corrlift_adjacency(close_wide, cutoff)
    return adj, diag


def forward_pass_smoke(D, adj, batch=2):
    """ONE CPU forward pass of MaskedRichNet(use_graph=True) on the corr+lift adjacency. Returns [b,N]."""
    b = int(min(batch, len(D.X_te)))
    if b < 1:
        raise RuntimeError("empty test split -- cannot run forward-pass smoke")
    net = RMR.MaskedRichNet(64, 4, 0.2, use_graph=True).eval()
    xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))
    nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float()
    base = torch.from_numpy(np.ascontiguousarray(adj)).float()
    adj_b = base.unsqueeze(0) * nmb.unsqueeze(1)
    with torch.no_grad():
        out = net(xb, adj_b)
    out_np = out.numpy()
    if not np.isfinite(out_np).all():
        raise RuntimeError("MaskedRichNet produced non-finite output on the corr+lift adjacency")
    return out_np


def _train_variant(D, cfg, use_graph, adj):
    """One edge choice: (ensemble TEST pred dict, per-seed TEST pred dicts, per-seed split outputs)."""
    outs = [RMR.train_masked_rich(D, cfg, s, use_graph, adj, return_splits=True) for s in cfg.seeds]
    seed_dicts = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in outs]
    return RMR._ens(seed_dicts), seed_dicts, outs


def run_training(panel, cfg, horizon, out_dir=None):
    """corr_lift_GAT vs stat_GAT_vol2pk vs no_graph_LSTM on all 5 metrics + date-clustered DM + fit evidence."""
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        adj_cl, diag = corrlift_adj_for(D, str(VE.PRICE[panel]))
        cl_pred, cl_seeds, cl_outs = _train_variant(D, cfg, True, adj_cl)
        stat_pred, stat_seeds, stat_outs = _train_variant(D, cfg, True, D.adj_vol2pk)
        lstm_pred, lstm_seeds, lstm_outs = _train_variant(D, cfg, False, D.adj_vol2pk)
        fl = cfg.qlike_floor
        preds = {"corr_lift_GAT": cl_pred, "stat_GAT_vol2pk": stat_pred, "no_graph_LSTM": lstm_pred}
        seedmap = {"corr_lift_GAT": cl_seeds, "stat_GAT_vol2pk": stat_seeds, "no_graph_LSTM": lstm_seeds}
        outsmap = {"corr_lift_GAT": cl_outs, "stat_GAT_vol2pk": stat_outs, "no_graph_LSTM": lstm_outs}
        metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seedmap.items()}
        # OVER/UNDER-FIT EVIDENCE (CLAUDE.md mandate 2026-08-29): seed-ensembled train/val split metrics +
        # per-model fit verdict + per-seed learning curves so the JSON can PROVE generalisation.
        tr_pred = {k: RMR._ens_split(outsmap[k], "train") for k in outsmap}
        va_pred = {k: RMR._ens_split(outsmap[k], "val") for k in outsmap}
        train_metrics = {k: RMR._split_metrics(tr_pred[k], D.y_tr, D.tmask_tr, fl) for k in outsmap}
        val_metrics = {k: RMR._split_metrics(va_pred[k], D.y_va, D.tmask_va, fl) for k in outsmap}
        fit_diagnostics = {k: RMR.OF.classify_fit(train_metrics[k], val_metrics[k], metrics[k]) for k in outsmap}
        learning_curves = {k: {"train": [o["train_curve"] for o in outsmap[k]],
                               "val": [o["val_curve"] for o in outsmap[k]],
                               "best_epoch": [o["best_epoch"] for o in outsmap[k]]} for k in outsmap}
        dm = {
            "corr_lift_vs_no_graph": RMR._dm_all(cl_pred, lstm_pred, horizon, fl),
            "corr_lift_vs_stat": RMR._dm_all(cl_pred, stat_pred, horizon, fl),
            "stat_vs_no_graph": RMR._dm_all(stat_pred, lstm_pred, horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon, "design": "corrlift-graph-edge-ablation",
            "paper": "Sonani, Badii & Moin 2025 (arXiv:2502.15813) sec 3.2 combined corr+lift edge",
            "device": _device_label(),
            "num_nodes": D.N, "n_test_obs": metrics["no_graph_LSTM"]["n"],
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "edge_density": diag,
            "metrics_ensemble": metrics, "metrics_per_seed": per_seed, "dm": dm,
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "fit_diagnostics": fit_diagnostics, "learning_curves": learning_curves,
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"corrlift_ablation_{panel}_h{horizon}.json").write_text(json.dumps(res, indent=2))
        return res


def run_dry(panel, horizon, max_tickers):
    """Build the corr+lift adjacency + one CPU forward pass. No training. Prints a summary."""
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=keep)
        adj_cl, diag = corrlift_adj_for(D, str(VE.PRICE[panel]))
        assert adj_cl.shape == (D.N, D.N), (adj_cl.shape, D.N)
        out = forward_pass_smoke(D, adj_cl, batch=2)
    print(f"[dry] {panel} nodes={D.N}  corr_edges={diag['n_corr_edges']} lift_edges={diag['n_lift_edges']} "
          f"either={diag['n_either_edges']} both={diag['n_both_edges']}  "
          f"avg_off_degree={diag['avg_off_degree']:.3f}  singletons={diag['n_singletons']}")
    print(f"[dry] forward pass OK: output shape {out.shape}, finite={np.isfinite(out).all()}")
    return {"n_nodes": D.N, "edge_density": diag, "forward_shape": list(out.shape)}


def main():
    ap = argparse.ArgumentParser(description="Corr+lift graph-edge ablation (GPU when available).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12,
                    help="dry mode: cap universe so the forward-pass smoke stays fast")
    ap.add_argument("--train-epochs", type=int, default=None,
                    help="TRAIN mode: run the N-epoch comparison (unset -> dry mode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override seeds for a quick check (e.g. --seeds 42 123 2026)")
    ap.add_argument("--batch", type=int, default=32,
                    help="batch size (small keeps a 154-node GAT well under 8GB VRAM)")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "corrlift_ablation"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers)
        return
    base = Config()
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = Config(epochs=a.train_epochs, patience=base.patience,
                 min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds, batch_size=a.batch)
    print(f"[train:{_device_label()}] {a.panel} corr+lift ablation h{a.horizon}, {cfg.epochs} epochs, seeds={cfg.seeds}")
    res = run_training(a.panel, cfg, a.horizon, out_dir=a.out_dir)
    print(f"  {'model':18} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics_ensemble"].items():
        print(f"  {k:18} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:22} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
