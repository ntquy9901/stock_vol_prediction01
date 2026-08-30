"""SERIAL LSTM->GNN hybrid runner (Sonani, Badii & Moin 2025, arXiv:2502.15813 sec 2.3 + 3.2), HNX h1.

Controlled comparison under the SAME masked panel / folds / seeds / HAR-X pipeline:
  * serial_hybrid_corrlift  : the SERIAL LSTM->GNN (GNN input = LSTM embedding) on the combined corr+lift graph
  * no_graph_LSTM           : the SAME serial model with use_graph=False (plain temporal baseline / the bar)
  * delivered_parallel_vol2pk: the shipped PARALLEL MaskedRichNet (LSTM + GAT vol->PK) -- CONTEXT only

Date-clustered DM (QLIKE/SE/AE): serial vs no_graph, serial vs delivered VolGA. The combined corr+lift
adjacency (corrlift_edge.build_corrlift_adjacency) is TRAIN-only (close rows strictly before D.d_va[0]) and
frozen -- no leakage. Primary thresholds rho>0.25, lift>1.2 (dense enough to exercise the graph on thin HNX
returns); the paper's 0.7/1.7 near-empty density is ALSO recorded.

All new code is here / in serial_hybrid_net.py; every live-training-path file is imported READ-ONLY.

Dry:   python run_serial_hybrid.py --panel hnx --horizon 1 --max-tickers 12
Train: python run_serial_hybrid.py --panel hnx --horizon 1 --train-epochs 10 --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if os.environ.get("SERIAL_FORCE_CPU", "0") == "1":  # pragma: no cover - import-time env side effect
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-08-29_corrlift_ablation" / "code",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only)
import masked_rich as MR                   # noqa: E402  (read-only)
import run_masked_rich as RMR              # noqa: E402  (read-only: helpers + delivered PARALLEL trainer)
import volatility_estimators as VE         # noqa: E402  (read-only)
from config import Config, SMOKE           # noqa: E402  (read-only)

import serial_hybrid_net as SH             # noqa: E402  (this baseline's model + trainer)
from corrlift_edge import build_corrlift_adjacency, load_close_wide  # noqa: E402  (read-only)

# Dense thresholds (single source of truth, recorded in edge_density). The paper's 0.7/1.7 give a near-empty
# graph on thin HNX returns; these fire ~312 corr + ~262 lift edges so the LSTM->GNN architecture is actually
# exercised (design.md sec 2). The paper density is also recorded via PAPER_CORR / PAPER_LIFT for completeness.
DENSE_CORR = 0.25
DENSE_LIFT = 1.2
PAPER_CORR = 0.7
PAPER_LIFT = 1.7


def _device_label():
    return "gpu" if torch.cuda.is_available() else "cpu"


def build_panel_masked(panel, cfg, horizon, out_dir, keep_tickers=None):
    """Write the panel's Parkinson processed CSVs (read-only estimator writer) and build the masked panel."""
    files = EFA._write_estimator_processed(panel, "parkinson", out_dir, keep_tickers=keep_tickers)
    if len(files) < 2:
        raise RuntimeError(f"{panel} processed build produced {len(files)} files (<2)")
    price_dir = VE.PRICE[panel]
    D = MR.build_masked_rich(files, str(price_dir), cfg.lookback, horizon,
                             edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)
    return D, files


def serial_adj_for(D, price_dir, corr_thresh=DENSE_CORR, lift_thresh=DENSE_LIFT):
    """Combined corr+lift adjacency aligned to ``D.tickers`` (TRAIN-only, frozen) + edge-density diag.

    Boundary = ``D.d_va[0]`` (first VALIDATION target date): every close row fed to the graph is strictly
    before every val/test target -> no evaluation leakage (mirrors corrlift_edge / the delivered edges).
    """
    if not D.d_va:
        raise RuntimeError("empty val split -- cannot locate the train/val boundary for the frozen graph")
    close_wide = load_close_wide(list(D.tickers), price_dir)
    cutoff = D.d_va[0]
    adj, diag = build_corrlift_adjacency(close_wide, cutoff, corr_thresh=corr_thresh, lift_thresh=lift_thresh)
    return adj, diag


def forward_pass_smoke(D, adj, batch=2):
    """ONE CPU forward pass of the SERIAL model (use_graph=True) on the corr+lift adjacency. Returns [b,N]."""
    b = int(min(batch, len(D.X_te)))
    if b < 1:
        raise RuntimeError("empty test split -- cannot run forward-pass smoke")
    net = SH.SerialLSTMGNN(64, 4, 0.2, use_graph=True).eval()
    xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))
    nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float()
    base = torch.from_numpy(np.ascontiguousarray(adj)).float()
    adj_b = base.unsqueeze(0) * nmb.unsqueeze(1)
    with torch.no_grad():
        out = net(xb, adj_b)
    out_np = out.numpy()
    if not np.isfinite(out_np).all():
        raise RuntimeError("SerialLSTMGNN produced non-finite output on the corr+lift adjacency")
    return out_np


def _train_serial_variant(D, cfg, use_graph, adj):
    """One SERIAL edge choice: (ensemble TEST pred dict, per-seed TEST pred dicts, per-seed split outputs)."""
    outs = [SH.train_serial(D, cfg, s, use_graph, adj, return_splits=True) for s in cfg.seeds]
    seed_dicts = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in outs]
    return RMR._ens(seed_dicts), seed_dicts, outs


def _train_parallel_variant(D, cfg):
    """The delivered PARALLEL MaskedRichNet (LSTM + GAT vol->PK) via the UNMODIFIED upstream trainer."""
    outs = [RMR.train_masked_rich(D, cfg, s, True, D.adj_vol2pk, return_splits=True) for s in cfg.seeds]
    seed_dicts = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in outs]
    return RMR._ens(seed_dicts), seed_dicts, outs


def run_training(panel, cfg, horizon, out_dir=None):
    """serial_hybrid vs no_graph_LSTM vs delivered_parallel on all 5 metrics + date-clustered DM + fit
    evidence + corr+lift edge density (dense + paper thresholds)."""
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        price_dir = str(VE.PRICE[panel])
        adj_cl, diag = serial_adj_for(D, price_dir, DENSE_CORR, DENSE_LIFT)
        _, diag_paper = serial_adj_for(D, price_dir, PAPER_CORR, PAPER_LIFT)
        ser_pred, ser_seeds, ser_outs = _train_serial_variant(D, cfg, True, adj_cl)
        nog_pred, nog_seeds, nog_outs = _train_serial_variant(D, cfg, False, adj_cl)
        par_pred, par_seeds, par_outs = _train_parallel_variant(D, cfg)
        fl = cfg.qlike_floor
        preds = {"serial_hybrid_corrlift": ser_pred, "no_graph_LSTM": nog_pred,
                 "delivered_parallel_vol2pk": par_pred}
        seedmap = {"serial_hybrid_corrlift": ser_seeds, "no_graph_LSTM": nog_seeds,
                   "delivered_parallel_vol2pk": par_seeds}
        outsmap = {"serial_hybrid_corrlift": ser_outs, "no_graph_LSTM": nog_outs,
                   "delivered_parallel_vol2pk": par_outs}
        metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seedmap.items()}
        tr_pred = {k: RMR._ens_split(outsmap[k], "train") for k in outsmap}
        va_pred = {k: RMR._ens_split(outsmap[k], "val") for k in outsmap}
        train_metrics = {k: RMR._split_metrics(tr_pred[k], D.y_tr, D.tmask_tr, fl) for k in outsmap}
        val_metrics = {k: RMR._split_metrics(va_pred[k], D.y_va, D.tmask_va, fl) for k in outsmap}
        fit_diagnostics = {k: RMR.OF.classify_fit(train_metrics[k], val_metrics[k], metrics[k]) for k in outsmap}
        learning_curves = {k: {"train": [o["train_curve"] for o in outsmap[k]],
                               "val": [o["val_curve"] for o in outsmap[k]],
                               "best_epoch": [o["best_epoch"] for o in outsmap[k]]} for k in outsmap}
        dm = {
            "serial_vs_no_graph": RMR._dm_all(ser_pred, nog_pred, horizon, fl),
            "serial_vs_delivered_parallel": RMR._dm_all(ser_pred, par_pred, horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon, "design": "serial-lstm-gnn-hybrid-corrlift",
            "paper": "Sonani, Badii & Moin 2025 (arXiv:2502.15813) sec 2.3 serial LSTM->GNN + sec 3.2 corr+lift edge",
            "architecture": "SERIAL: GNN node features ARE the per-stock LSTM temporal embeddings "
                            "(distinct from the delivered PARALLEL MaskedRichNet whose GAT reads raw day-t features)",
            "device": _device_label(),
            "num_nodes": D.N, "n_test_obs": metrics["no_graph_LSTM"]["n"],
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "edge_density": diag, "edge_density_paper_thresholds": diag_paper,
            "metrics_ensemble": metrics, "metrics_per_seed": per_seed, "dm": dm,
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "fit_diagnostics": fit_diagnostics, "learning_curves": learning_curves,
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"lstm_gnn_serial_hybrid_{panel}_h{horizon}.json").write_text(json.dumps(res, indent=2))
        return res


def run_dry(panel, horizon, max_tickers):
    """Build the corr+lift adjacency + one CPU forward pass of the SERIAL model. No training."""
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=keep)
        adj_cl, diag = serial_adj_for(D, str(VE.PRICE[panel]))
        assert adj_cl.shape == (D.N, D.N), (adj_cl.shape, D.N)
        out = forward_pass_smoke(D, adj_cl, batch=2)
    print(f"[dry] {panel} nodes={D.N}  corr_edges={diag['n_corr_edges']} lift_edges={diag['n_lift_edges']} "
          f"either={diag['n_either_edges']} both={diag['n_both_edges']}  "
          f"avg_off_degree={diag['avg_off_degree']:.3f}  singletons={diag['n_singletons']}")
    print(f"[dry] forward pass OK: output shape {out.shape}, finite={np.isfinite(out).all()}")
    return {"n_nodes": D.N, "edge_density": diag, "forward_shape": list(out.shape)}


def main():
    ap = argparse.ArgumentParser(description="Serial LSTM->GNN hybrid (GPU when available).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12,
                    help="dry mode: cap universe so the forward-pass smoke stays fast")
    ap.add_argument("--train-epochs", type=int, default=None,
                    help="TRAIN mode: run the N-epoch comparison (unset -> dry mode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override seeds (e.g. --seeds 42 123 2026)")
    ap.add_argument("--batch", type=int, default=16,
                    help="batch size (small keeps a ~162-node GAT well under 8GB VRAM)")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "lstm_gnn_serial_hybrid"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers)
        return
    base = Config()
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = Config(epochs=a.train_epochs, patience=base.patience,
                 min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds, batch_size=a.batch)
    print(f"[train:{_device_label()}] {a.panel} serial LSTM->GNN h{a.horizon}, {cfg.epochs} epochs, seeds={cfg.seeds}")
    res = run_training(a.panel, cfg, a.horizon, out_dir=a.out_dir)
    print(f"  {'model':26} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics_ensemble"].items():
        print(f"  {k:26} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:30} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
