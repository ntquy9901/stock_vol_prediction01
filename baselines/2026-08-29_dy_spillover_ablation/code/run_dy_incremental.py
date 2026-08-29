"""Checkpointed HNX DY-ablation driver (survives background-process kills on the shared machine).

Same experiment as ``run_dy_ablation.run_training`` (dy_GAT vs stat_GAT_vol2pk vs no_graph_LSTM, HNX h1,
all 5 metrics + date-clustered DM), but each (variant, seed) TEST prediction dict is written to a
checkpoint pickle the moment it finishes. A killed run therefore loses at most the in-flight training;
re-running SKIPS finished (variant, seed) pairs and trains only the missing ones. Once all pairs exist,
the final result JSON + connectedness stats are assembled from the checkpoints.

Seeds are looped OUTERMOST so an early kill still yields a complete lower-seed comparison. CPU-forced by
default (inherited from ``run_dy_ablation`` which hides CUDA before torch import).

    DY_ABLATION_FORCE_CPU=1 .venv_gpu_encode/Scripts/python run_dy_incremental.py \
        --panel hnx --horizon 1 --epochs 10 --seeds 42 123 2026
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import run_dy_ablation as RDA  # noqa: E402  (also sets the CPU-force env + all read-only imports)
import run_masked_rich as RMR  # noqa: E402
from config import Config      # noqa: E402

REPO = RDA.REPO
# (variant label -> (use_graph, adjacency attribute on D or the DY adj sentinel))
VARIANTS = ("dy_GAT", "stat_GAT_vol2pk", "no_graph_LSTM")


def _ckpt_path(ckpt_dir: Path, variant: str, seed: int) -> Path:
    return ckpt_dir / f"{variant}_seed{seed}.pkl"


def _adj_for(variant: str, D, adj_dy):
    """(use_graph, adjacency) for a variant. stat/no-graph reuse the shipped vol2pk edge."""
    if variant == "dy_GAT":
        return True, adj_dy
    if variant == "stat_GAT_vol2pk":
        return True, D.adj_vol2pk
    return False, D.adj_vol2pk               # no_graph_LSTM


def train_one(D, cfg, variant, seed, adj_dy):
    """Train a single (variant, seed) and return the FULL split output (test/val/train predictions +
    per-epoch learning curves + best_epoch), so the assembled result carries over/under-fit evidence."""
    use_graph, adj = _adj_for(variant, D, adj_dy)
    return RMR.train_masked_rich(D, cfg, seed, use_graph, adj, return_splits=True)


def assemble(ckpt_dir: Path, seeds, horizon, floor, dy_stats, panel, D, n_obs, epochs, lookback):
    """Build the final result dict from all per-(variant, seed) checkpoints (all must exist).

    Carries the CLAUDE.md-mandated over/under-fit evidence: seed-ensembled train/val/test metrics, a
    per-variant fit verdict (overfit_check.classify_fit), and per-seed learning curves — the same schema
    the sibling sector/learned-edge runs produced.
    """
    N = D.N
    outs = {v: [pickle.loads(_ckpt_path(ckpt_dir, v, s).read_bytes()) for s in seeds] for v in VARIANTS}
    # TEST prediction dicts (per seed) -> ensemble + per-seed metrics + DM
    test_dicts = {v: [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, N) for o in outs[v]] for v in VARIANTS}
    ens = {v: RMR._ens(test_dicts[v]) for v in VARIANTS}
    metrics = {v: RMR._metrics(ens[v], floor) for v in VARIANTS}
    per_seed = {v: RMR.seed_metric_stats(test_dicts[v], floor) for v in VARIANTS}
    # train/val evidence: seed-ensembled split arrays -> masked split metrics -> fit verdict
    tr_pred = {v: RMR._ens_split(outs[v], "train") for v in VARIANTS}
    va_pred = {v: RMR._ens_split(outs[v], "val") for v in VARIANTS}
    train_metrics = {v: RMR._split_metrics(tr_pred[v], D.y_tr, D.tmask_tr, floor) for v in VARIANTS}
    val_metrics = {v: RMR._split_metrics(va_pred[v], D.y_va, D.tmask_va, floor) for v in VARIANTS}
    fit_diagnostics = {v: RMR.OF.classify_fit(train_metrics[v], val_metrics[v], metrics[v]) for v in VARIANTS}
    learning_curves = {v: {"train": [o["train_curve"] for o in outs[v]],
                           "val": [o["val_curve"] for o in outs[v]],
                           "best_epoch": [o["best_epoch"] for o in outs[v]]} for v in VARIANTS}
    dm = {
        "dy_vs_no_graph": RMR._dm_all(ens["dy_GAT"], ens["no_graph_LSTM"], horizon, floor),
        "dy_vs_stat": RMR._dm_all(ens["dy_GAT"], ens["stat_GAT_vol2pk"], horizon, floor),
        "stat_vs_no_graph": RMR._dm_all(ens["stat_GAT_vol2pk"], ens["no_graph_LSTM"], horizon, floor),
    }
    dev = "cpu" if os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "" else "gpu"
    return {
        "panel": panel, "horizon": horizon, "design": "dy-spillover-graph-ablation-incremental",
        "edge": "DY-2014 generalized-FEVD connectedness (elastic-net VAR)",
        "device": dev, "num_nodes": N, "n_test_obs": n_obs,
        "seeds": list(seeds), "epochs": epochs, "lookback": lookback,
        "dy_connectedness": dy_stats,
        "metrics_ensemble": metrics, "metrics_per_seed": per_seed, "dm": dm,
        "train_metrics": train_metrics, "val_metrics": val_metrics,
        "fit_diagnostics": fit_diagnostics, "learning_curves": learning_curves,
        "sector_gat_context": RDA._load_sector_dm(ens["dy_GAT"], horizon, floor),
    }


def run(panel, horizon, epochs, seeds, out_dir, p=1, H=10, alpha=0.05, l1_ratio=0.5):
    base = Config()
    cfg = Config(epochs=epochs, patience=base.patience, min_epochs=min(base.min_epochs, epochs),
                 seeds=tuple(seeds))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "ckpt"; ckpt_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        D, files = RDA.build_panel_masked(panel, cfg, horizon, td,
                                          keep_tickers=RDA.EFA.screened_tickers(panel))
        adj_dy, dy_stats = RDA.dy_adj_for(D, files, p=p, H=H, alpha=alpha, l1_ratio=l1_ratio)
        (ckpt_dir / "dy_stats.json").write_text(json.dumps(dy_stats, indent=2, default=float))
        n_obs = int(D.tmask_te.astype(bool).sum())
        for s in seeds:                                  # seeds OUTERMOST -> early kill keeps a full low-seed set
            for v in VARIANTS:
                cp = _ckpt_path(ckpt_dir, v, s)
                if cp.exists():
                    print(f"[skip] {v} seed{s} (checkpoint exists)", flush=True)
                    continue
                pd_dict = train_one(D, cfg, v, s, adj_dy)
                cp.write_bytes(pickle.dumps(pd_dict))
                print(f"[done] {v} seed{s} -> {cp.name} (n={len(pd_dict)})", flush=True)
        res = assemble(ckpt_dir, seeds, horizon, cfg.qlike_floor, dy_stats, panel, D, n_obs,
                       epochs, cfg.lookback)
    (out / f"dy_ablation_{panel}_h{horizon}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[assembled] {out / f'dy_ablation_{panel}_h{horizon}.json'}", flush=True)
    for k, m in res["metrics_ensemble"].items():
        print(f"  {k:18} QLIKE={m['qlike']:.4f} RMSE={m['rmse']:.5f} MAE={m['mae']:.6f} R2={m['r2']:.3f}", flush=True)
    for name, d in res["dm"].items():
        q = d.get("qlike", {})
        print(f"  DM {name:18} QLIKE p={q.get('p_value')} favors={q.get('favors')} mean_diff={q.get('mean_diff')}", flush=True)
    return res


def main():  # pragma: no cover  (entry-driver)
    ap = argparse.ArgumentParser(description="Checkpointed DY-ablation driver (kill-resumable).")
    ap.add_argument("--panel", default="hnx")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    ap.add_argument("--var-lag", type=int, default=1)
    ap.add_argument("--fevd-h", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(REPO / "results" / "dy_spillover_ablation"))
    a = ap.parse_args()
    run(a.panel, a.horizon, a.epochs, a.seeds, a.out_dir,
        p=a.var_lag, H=a.fevd_h, alpha=a.alpha, l1_ratio=a.l1_ratio)


if __name__ == "__main__":  # pragma: no cover
    main()
