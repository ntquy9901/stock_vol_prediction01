"""SECTOR-graph ablation for the LSTM+GAT volatility model (panel-agnostic).

Compares three edge choices under the SAME MaskedRichNet / HAR-X pipeline on a chosen panel:
  * sector-GAT   : use_graph=True, adj = static sector adjacency (this baseline's contribution)
  * stat-GAT     : use_graph=True, adj = D.adj_vol2pk (the shipped directed volume-shock->PK edge)
  * no-graph LSTM: use_graph=False

Sector source per panel: VN panels (hnx/vn100/vn30) use the vnstock ICB map (``vn_sectors.csv``);
sp500 uses the GICS map (``sp500_gics_sectors.csv``). The sector edge is static metadata -> no OOS
drift, no train/test leakage (unlike the ~9-30%-persistent statistical edges).

All new code lives here; the live-training-path files are IMPORTED read-only (never edited):
  estimator_forecast_ablation (processed writer + screened universe), masked_rich (panel build),
  run_masked_rich (MaskedRichNet + train_masked_rich), config (Config/SMOKE).

CPU IS FORCED by default (``SECTOR_ABLATION_FORCE_CPU=1``): CUDA is hidden BEFORE torch import so a
run here never contends with the live GPU job. Set ``SECTOR_ABLATION_FORCE_CPU=0`` to allow GPU once
it is free. Default mode is DRY/SMOKE (build adj + one forward pass, no training); ``--train-epochs N``
runs the CPU comparison.

Dry run:   python run_sector_ablation.py --panel hnx --horizon 1 --max-tickers 12
CPU train: python run_sector_ablation.py --panel hnx --horizon 1 --train-epochs 8 --seeds 42 123
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Force CPU BEFORE importing torch so torch.cuda.is_available() (checked inside train_masked_rich, a
# file we must not edit) returns False -> the ablation stays off the live-GPU job.
if os.environ.get("SECTOR_ABLATION_FORCE_CPU", "1") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # HARD-set (not setdefault): a preset value must not leak the GPU

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "garch_masked",
           REPO / "scripts" / "eda",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import torch  # noqa: E402

import estimator_forecast_ablation as EFA  # noqa: E402  (read-only import)
import masked_rich as MR                   # noqa: E402  (read-only import)
import run_masked_rich as RMR              # noqa: E402  (read-only import)
import volatility_estimators as VE         # noqa: E402  (read-only import)
from config import Config, SMOKE           # noqa: E402  (read-only import)

from sector_adjacency import build_sector_adjacency, coverage, load_sector_map  # noqa: E402

VN_SECTOR_CSV = _HERE.parent / "vn_sectors.csv"
SP500_SECTOR_CSV = _HERE.parent / "sp500_gics_sectors.csv"


def default_sector_csv(panel: str) -> Path:
    """The sector map file for a panel (VN panels -> ICB map, sp500 -> GICS map)."""
    return SP500_SECTOR_CSV if panel == "sp500" else VN_SECTOR_CSV


def build_panel_masked(panel: str, cfg: Config, horizon: int, out_dir: str, keep_tickers=None):
    """Write the panel's Parkinson processed CSVs (reusing the read-only estimator writer) and build the
    masked panel. ``keep_tickers`` (a set) restricts the universe -- used to keep smoke builds tiny."""
    files = EFA._write_estimator_processed(panel, "parkinson", out_dir, keep_tickers=keep_tickers)
    if len(files) < 2:
        raise RuntimeError(f"{panel} processed build produced {len(files)} files (<2)")
    price_dir = VE.PRICE[panel]
    D = MR.build_masked_rich(files, str(price_dir), cfg.lookback, horizon,
                             edge_min_overlap=MR.EDGE_MIN_OVERLAP, top_k=MR.EDGE_TOP_K)
    return D, files


def sector_adj_for(tickers, sector_csv, top_k=None):
    """Sector adjacency aligned to ``tickers`` + a coverage/degree summary."""
    smap = load_sector_map(sector_csv)
    adj = build_sector_adjacency(list(tickers), smap, top_k=top_k)
    return adj, coverage(list(tickers), smap)


def forward_pass_smoke(D, adj, batch=2):
    """ONE CPU forward pass of MaskedRichNet(use_graph=True) given the sector adjacency.

    Replicates the training-loop adjacency batching (base * valid-source-node mask). Returns [b,N]."""
    b = int(min(batch, len(D.X_te)))
    if b < 1:
        raise RuntimeError("empty test split -- cannot run forward-pass smoke")
    net = RMR.MaskedRichNet(64, 4, 0.2, use_graph=True).eval()   # CPU (no .to(cuda))
    xb = torch.from_numpy(np.ascontiguousarray(D.X_te[:b]))                       # [b,N,seq,5]
    nmb = torch.from_numpy(np.ascontiguousarray(D.nmask_te[:b])).float()          # [b,N]
    base = torch.from_numpy(np.ascontiguousarray(adj)).float()                    # [N,N]
    adj_b = base.unsqueeze(0) * nmb.unsqueeze(1)                                  # [b,N,N]
    with torch.no_grad():
        out = net(xb, adj_b)
    out_np = out.numpy()
    if not np.isfinite(out_np).all():
        raise RuntimeError("MaskedRichNet produced non-finite output on the sector adjacency")
    return out_np


def _train_variant(D, cfg, use_graph, adj):
    """Seed-ensembled TEST prediction dict for one edge choice (run_masked_rich conventions)."""
    seed_dicts = [RMR._pred_dict(RMR.train_masked_rich(D, cfg, s, use_graph, adj),
                                 D.y_te, D.tmask_te, D.d_te, D.N) for s in cfg.seeds]
    return RMR._ens(seed_dicts), seed_dicts


def run_training(panel, cfg, horizon, sector_csv=None, out_dir=None):
    """5-10 epoch comparison: sector-GAT vs stat-GAT vs no-graph LSTM on all 5 metrics + date-clustered DM.

    CPU-forced by default (see module header). A quick DIRECTIONAL check, not a final number."""
    sector_csv = sector_csv or default_sector_csv(panel)
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=EFA.screened_tickers(panel))
        adj_sec, cov = sector_adj_for(D.tickers, sector_csv)
        sector_pred, sector_seeds = _train_variant(D, cfg, True, adj_sec)
        stat_pred, stat_seeds = _train_variant(D, cfg, True, D.adj_vol2pk)
        lstm_pred, lstm_seeds = _train_variant(D, cfg, False, D.adj_vol2pk)
        fl = cfg.qlike_floor
        preds = {"sector_GAT": sector_pred, "stat_GAT_vol2pk": stat_pred, "no_graph_LSTM": lstm_pred}
        seedmap = {"sector_GAT": sector_seeds, "stat_GAT_vol2pk": stat_seeds, "no_graph_LSTM": lstm_seeds}
        metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
        per_seed = {k: RMR.seed_metric_stats(v, fl) for k, v in seedmap.items()}
        dm = {
            "sector_vs_stat": RMR._dm_all(sector_pred, stat_pred, horizon, fl),
            "sector_vs_no_graph": RMR._dm_all(sector_pred, lstm_pred, horizon, fl),
            "stat_vs_no_graph": RMR._dm_all(stat_pred, lstm_pred, horizon, fl),
        }
        res = {
            "panel": panel, "horizon": horizon, "design": "sector-graph-ablation",
            "device": "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "gpu",
            "num_nodes": D.N, "n_test_obs": metrics["no_graph_LSTM"]["n"],
            "seeds": list(cfg.seeds), "epochs": cfg.epochs, "lookback": cfg.lookback,
            "sector_coverage": {k: cov[k] for k in ("n_tickers", "n_mapped", "coverage_frac",
                                                    "n_sectors", "avg_off_degree", "max_off_degree",
                                                    "n_singletons")},
            "metrics_ensemble": metrics, "metrics_per_seed": per_seed, "dm": dm,
        }
        if out_dir:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            (out / f"sector_ablation_{panel}_h{horizon}.json").write_text(json.dumps(res, indent=2))
        return res


def run_dry(panel, horizon, max_tickers, sector_csv=None):
    """Build the sector adjacency + one CPU forward pass. No training. Prints a summary."""
    sector_csv = sector_csv or default_sector_csv(panel)
    cfg = SMOKE
    keep = EFA.screened_tickers(panel)
    if max_tickers and keep is not None:
        keep = set(sorted(keep)[:max_tickers])
    with tempfile.TemporaryDirectory() as td:
        D, _ = build_panel_masked(panel, cfg, horizon, td, keep_tickers=keep)
        adj_sec, cov = sector_adj_for(D.tickers, sector_csv)
        assert adj_sec.shape == (D.N, D.N), (adj_sec.shape, D.N)
        out = forward_pass_smoke(D, adj_sec, batch=2)
    print(f"[dry] {panel} nodes={D.N}  sector coverage={cov['n_mapped']}/{cov['n_tickers']} "
          f"({cov['coverage_frac']*100:.1f}%)  sectors={cov['n_sectors']}  "
          f"avg_off_degree={cov['avg_off_degree']:.2f}  singletons={cov['n_singletons']}")
    print(f"[dry] forward pass OK: output shape {out.shape}, finite={np.isfinite(out).all()}")
    return {"n_nodes": D.N, "coverage": cov, "forward_shape": list(out.shape)}


def main():
    ap = argparse.ArgumentParser(description="Sector-graph ablation (CPU-forced by default).")
    ap.add_argument("--panel", default="hnx", help="hnx (primary) | vn100 | vn30 | sp500")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--max-tickers", type=int, default=12,
                    help="dry mode: cap universe so the CPU forward-pass smoke stays fast")
    ap.add_argument("--sector-csv", default=None)
    ap.add_argument("--train-epochs", type=int, default=None,
                    help="TRAIN mode: run the N-epoch CPU comparison (unset -> dry mode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override seeds for a quick check (e.g. --seeds 42 123)")
    ap.add_argument("--out-dir", default=str(_HERE.parents[2] / "results" / "sector_gat_ablation"))
    a = ap.parse_args()
    if a.train_epochs is None:
        run_dry(a.panel, a.horizon, a.max_tickers, a.sector_csv)
        return
    base = Config()
    seeds = tuple(a.seeds) if a.seeds else base.seeds
    cfg = Config(epochs=a.train_epochs, patience=base.patience,
                 min_epochs=min(base.min_epochs, a.train_epochs), seeds=seeds)
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" else "gpu"
    print(f"[train:{dev}] {a.panel} sector ablation h{a.horizon}, {cfg.epochs} epochs, seeds={cfg.seeds}")
    res = run_training(a.panel, cfg, a.horizon, a.sector_csv, out_dir=a.out_dir)
    print(f"  {'model':18} {'MSE':>10} {'RMSE':>9} {'MAE':>9} {'QLIKE':>8} {'R2':>7}  n={res['n_test_obs']}")
    for k, m in res["metrics_ensemble"].items():
        print(f"  {k:18} {m['mse']:>10.3e} {m['rmse']:>9.4f} {m['mae']:>9.4f} {m['qlike']:>8.4f} {m['r2']:>7.3f}")
    for name, d in res["dm"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:22} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}")


if __name__ == "__main__":  # pragma: no cover
    main()
