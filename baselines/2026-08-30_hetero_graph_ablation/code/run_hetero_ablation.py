"""Heterogeneous 2-relation GNN edge probe for the LSTM+GAT volatility model (7th HNX edge probe).

Compares three variants under the SAME MaskedRich / HAR-X pipeline, same masked folds + seeds:
  * hetero_2rel_GAT      : HeteroRichNet -- linear_corr + nonlinear_assoc as SEPARATE relations with
                           INDEPENDENT conv weights, per-relation Min-Max-normalized edge weights (this probe).
  * squashed_lowered_GAT : MaskedRichNet(use_graph=True) on the SINGLE squashed corr+lift adjacency at the
                           SAME lowered thresholds (0.25/1.2) -- isolates "hetero vs squash" from "dense vs sparse".
  * no_graph_LSTM        : MaskedRichNet(use_graph=False).

The two relation adjacencies (``hetero_edges.build_relation_adjacencies``) and the squashed-lowered adjacency
(``corrlift_edge.build_corrlift_adjacency``) are TRAIN-ONLY (close rows strictly before ``D.d_va[0]``) and
frozen. Context (report only): prior squashed@paper-thresholds (0.7/1.7) QLIKE from results/corrlift_ablation.

All live-training-path files are IMPORTED read-only (never edited): estimator_forecast_ablation, masked_rich,
run_masked_rich, volatility_estimators, config, corrlift_edge. New code = hetero_edges + hetero_model + this.

GPU is used when available (``.venv_gpu_encode`` torch 2.6.0). ``torch.cuda.is_available()`` is the single
source of truth for the device label. Default mode is DRY/SMOKE (build adjacencies + one forward pass, no
training); ``--train-epochs N`` runs the comparison.

Dry:   python run_hetero_ablation.py --panel hnx --horizon 1 --max-tickers 12
Train: python run_hetero_ablation.py --panel hnx --horizon 1 --train-epochs 10 --seeds 42 123 2026 --batch 32
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-08-29_corrlift_ablation" / "code",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only import)
import masked_rich as MR                    # noqa: E402  (read-only import)
import run_masked_rich as RMR               # noqa: E402  (read-only import)
import volatility_estimators as VE          # noqa: E402  (read-only import)
from config import Config, SMOKE            # noqa: E402  (read-only import)

import corrlift_edge as CL                  # noqa: E402  (read-only reuse: squashed-lowered adjacency)
import hetero_edges as HE                   # noqa: E402
import hetero_model as HM                   # noqa: E402

PRIOR_SQUASHED_PAPER_QLIKE = 1.8192  # results/corrlift_ablation/corrlift_ablation_hnx_h1.json (0.7/1.7, context)


def _device_label():
    """Actual training device: torch.cuda.is_available() is the single source of truth."""
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


def hetero_adj_for(D, price_dir):
    """The two relation adjacencies + the squashed-lowered adjacency, aligned to ``D.tickers`` (TRAIN-only,
    frozen), plus a combined edge-density diag. Cutoff = ``D.d_va[0]`` (first val TARGET) -> no leakage."""
    if not D.d_va:
        raise RuntimeError("empty val split -- cannot locate the train/val boundary for the frozen graph")
    close_wide = HE.load_close_wide(list(D.tickers), price_dir)
    cutoff = D.d_va[0]
    adj_lin, adj_nl, diag = HE.build_relation_adjacencies(close_wide, cutoff)
    adj_sq, sq_diag = CL.build_corrlift_adjacency(close_wide, cutoff,
                                                  corr_thresh=HE.CORR_THRESH, lift_thresh=HE.LIFT_THRESH)
    diag["squashed_lowered"] = {"n_either_edges": sq_diag["n_either_edges"],
                                "n_corr_edges": sq_diag["n_corr_edges"],
                                "n_lift_edges": sq_diag["n_lift_edges"],
                                "avg_off_degree": sq_diag["avg_off_degree"],
                                "n_singletons": sq_diag["n_singletons"]}
    return adj_lin, adj_nl, adj_sq, diag


def forward_pass_smoke(D, adj_lin, adj_nl, batch=2):
    """ONE CPU forward pass of HeteroRichNet on the two relation adjacencies. Returns [b,N]."""
    b = int(min(batch, len(D.X_te)))
    if b < 1:
        raise RuntimeError("empty test split -- cannot run forward-pass smoke")
    net = HM.HeteroRichNet(64, 4, 0.2).eval()
    xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))
    nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float().unsqueeze(1)
    al = torch.from_numpy(np.ascontiguousarray(adj_lin)).float().unsqueeze(0) * nmb
    an = torch.from_numpy(np.ascontiguousarray(adj_nl)).float().unsqueeze(0) * nmb
    with torch.no_grad():
        out = net(xb, al, an)
    out_np = out.numpy()
    if not np.isfinite(out_np).all():
        raise RuntimeError("HeteroRichNet produced non-finite output on the relation adjacencies")
    return out_np


def _train_hetero(D, cfg, adj_lin, adj_nl):
    outs = [HM.train_hetero_rich(D, cfg, s, adj_lin, adj_nl, return_splits=True) for s in cfg.seeds]
    seed_dicts = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in outs]
    return RMR._ens(seed_dicts), seed_dicts, outs


def _train_masked(D, cfg, use_graph, adj):
    outs = [RMR.train_masked_rich(D, cfg, s, use_graph, adj, return_splits=True) for s in cfg.seeds]
    seed_dicts = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in outs]
    return RMR._ens(seed_dicts), seed_dicts, outs


def run_training(panel, cfg, horizon, out_dir=None):
    """hetero_2rel_GAT vs squashed_lowered_GAT vs no_graph_LSTM on all 5 metrics + date-clustered DM + fit."""
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        adj_lin, adj_nl, adj_sq, diag = hetero_adj_for(D, str(VE.PRICE[panel]))
        het_pred, het_seeds, het_outs = _train_hetero(D, cfg, adj_lin, adj_nl)
        sq_pred, sq_seeds, sq_outs = _train_masked(D, cfg, True, adj_sq)
        ng_pred, ng_seeds, ng_outs = _train_masked(D, cfg, False, D.adj_vol2pk)
        fl = cfg.qlike_floor
        preds = {"hetero_2rel_GAT": het_pred, "squashed_lowered_GAT": sq_pred, "no_graph_LSTM": ng_pred}
        seedmap = {"hetero_2rel_GAT": het_seeds, "squashed_lowered_GAT": sq_seeds, "no_graph_LSTM": ng_seeds}
        outsmap = {"hetero_2rel_GAT": het_outs, "squashed_lowered_GAT": sq_outs, "no_graph_LSTM": ng_outs}
        metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seedmap.items()}
        # OVER/UNDER-FIT EVIDENCE (CLAUDE.md mandate): seed-ensembled train/val split metrics + per-model
        # fit verdict + per-seed learning curves so the JSON can PROVE generalisation.
        tr_pred = {k: RMR._ens_split(outsmap[k], "train") for k in outsmap}
        va_pred = {k: RMR._ens_split(outsmap[k], "val") for k in outsmap}
        train_metrics = {k: RMR._split_metrics(tr_pred[k], D.y_tr, D.tmask_tr, fl) for k in outsmap}
        val_metrics = {k: RMR._split_metrics(va_pred[k], D.y_va, D.tmask_va, fl) for k in outsmap}
        fit_diagnostics = {k: RMR.OF.classify_fit(train_metrics[k], val_metrics[k], metrics[k]) for k in outsmap}
        learning_curves = {k: {"train": [o["train_curve"] for o in outsmap[k]],
                               "val": [o["val_curve"] for o in outsmap[k]],
                               "best_epoch": [o["best_epoch"] for o in outsmap[k]]} for k in outsmap}
        dm = {
            "hetero_vs_no_graph": RMR._dm_all(het_pred, ng_pred, horizon, fl),
            "hetero_vs_squashed_lowered": RMR._dm_all(het_pred, sq_pred, horizon, fl),
            "squashed_lowered_vs_no_graph": RMR._dm_all(sq_pred, ng_pred, horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon, "design": "hetero-2relation-graph-edge-ablation",
            "paper": "arXiv:2502.15813 sec 3.2 (denser-graph heterogeneous VARIANT: separate relations, "
                     "independent conv weights, per-relation Min-Max; thresholds 0.25/1.2 DEPART from 0.7/1.7)",
            "device": _device_label(),
            "aggregation": "sum",
            "num_nodes": D.N, "n_test_obs": metrics["no_graph_LSTM"]["n"],
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "prior_squashed_paper_qlike": PRIOR_SQUASHED_PAPER_QLIKE,
            "edge_density": diag,
            "metrics_ensemble": metrics, "metrics_per_seed": per_seed, "dm": dm,
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "fit_diagnostics": fit_diagnostics, "learning_curves": learning_curves,
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"hetero_graph_ablation_{panel}_h{horizon}.json").write_text(json.dumps(res, indent=2))
        return res


def run_dry(panel, horizon, max_tickers):
    """Build the two relation adjacencies + one CPU forward pass. No training. Prints a summary."""
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=keep)
        adj_lin, adj_nl, adj_sq, diag = hetero_adj_for(D, str(VE.PRICE[panel]))
        assert adj_lin.shape == (D.N, D.N) and adj_nl.shape == (D.N, D.N), (adj_lin.shape, adj_nl.shape, D.N)
        out = forward_pass_smoke(D, adj_lin, adj_nl, batch=2)
    lc, nl = diag["linear_corr"], diag["nonlinear_assoc"]
    print(f"[dry] {panel} nodes={D.N}  linear_corr edges={lc['n_edges']} (avg_deg={lc['avg_off_degree']:.3f}, "
          f"singletons={lc['n_singletons']})  nonlinear_assoc edges={nl['n_edges']} "
          f"(avg_deg={nl['avg_off_degree']:.3f}, singletons={nl['n_singletons']})  "
          f"both={diag['n_both_relations_edges']}")
    print(f"[dry] forward pass OK: output shape {out.shape}, finite={np.isfinite(out).all()}")
    return {"n_nodes": D.N, "edge_density": diag, "forward_shape": list(out.shape)}


def main():
    ap = argparse.ArgumentParser(description="Heterogeneous 2-relation graph-edge ablation (GPU when available).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12,
                    help="dry mode: cap universe so the forward-pass smoke stays fast")
    ap.add_argument("--train-epochs", type=int, default=None,
                    help="TRAIN mode: run the N-epoch comparison (unset -> dry mode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override seeds (e.g. --seeds 42 123 2026)")
    ap.add_argument("--batch", type=int, default=32,
                    help="batch size (small keeps a 154-node 2-relation GAT well under 8GB VRAM)")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "hetero_graph_ablation"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers)
        return
    base = Config()
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = Config(epochs=a.train_epochs, patience=base.patience,
                 min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds, batch_size=a.batch)
    print(f"[train:{_device_label()}] {a.panel} hetero 2-relation ablation h{a.horizon}, "
          f"{cfg.epochs} epochs, seeds={cfg.seeds}")
    res = run_training(a.panel, cfg, a.horizon, out_dir=a.out_dir)
    print(f"  {'model':22} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics_ensemble"].items():
        print(f"  {k:22} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:28} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
