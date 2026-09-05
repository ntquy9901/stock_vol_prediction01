"""Contemporaneous-correlation edge ablation on VN100 (keeps 5 features).

Swaps the delivered VolGA edge (directed volume->Parkinson lead-lag) for a contemporaneous
sqrt(PK) correlation Top-5 edge, holding everything else fixed. Goal: a more stable edge (train->test
Top-5 overlap ~25% vs ~5% for vol->PK) so the graph's marginal value (VolGA - no-graph LSTM) is
stable across horizons. Folds are configurable (default 6 for speed; 22 later).

Run:  .venv_gpu_encode/Scripts/python.exe baselines/2026-09-04_contemp_edge/code/run_contemp.py --edge contemp --horizon 1 --folds-target 6
Smoke: ... --smoke   (1 fold, 2 epochs, 1 seed, h1 -> quick VolGA-vs-LSTM sign check)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import run_masked_rich as RMR  # noqa: E402
import masked_rich as MR  # noqa: E402
import pipeline_config as pc  # noqa: E402  (single source of truth for positivity floor)
from run_walkforward import _har_ols_preds, training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
import glob as _glob

_MODELS = ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk")


def build_contemp_adj(sqrt_pk: np.ndarray, last_row: int, top_k: int) -> np.ndarray:
    """Undirected contemporaneous edge: A[j, i] = corr(sqrt_pk_i(t), sqrt_pk_j(t)) over TRAIN dates
    (t <= last_row), Top-K |corr| sources per target, self-loop = 1. Pairwise-complete (NaN-safe)."""
    tr = sqrt_pk[:last_row + 1]
    C = pd.DataFrame(tr).corr().to_numpy()
    C = np.nan_to_num(C, nan=0.0).astype(np.float32)
    n = C.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    Cd = C.copy()
    np.fill_diagonal(Cd, 0.0)
    for j in range(n):
        k = np.argsort(-np.abs(Cd[j]))[:top_k]
        A[j, k] = Cd[j, k]
    np.fill_diagonal(A, 1.0)
    return A


def _fold_adj(panel, fold, wf, edge: str, D):
    if edge == "vol2pk":
        return D.adj_vol2pk
    last_tr_row = int(panel.anchors[fold.train][-1]) + wf.horizon
    return build_contemp_adj(np.sqrt(panel.pk), last_tr_row, MR.EDGE_TOP_K)


def run(edge: str, horizon: int, folds_target: int, epochs: int, smoke: bool, out=None, n_seeds: int = 5,
        market: str = "vn100"):  # pragma: no cover  (GPU training driver; covered by smoke run, not unit tests)
    t0 = time.time()
    lookback = 22
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)   # full screened universe (graph needs all nodes)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    # smoke: full nodes but 1 fold / 8 epochs / 3 seeds -> a meaningful VolGA-vs-LSTM sign at h1 in ~10 min
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=(1 if smoke else folds_target))
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    cfg = training_config(epochs=(8 if smoke else epochs),
                          seeds=((42, 123, 2026) if smoke else (42, 123, 2026, 7, 2024)[:n_seeds]))
    fl = cfg.qlike_floor
    pooled = {"HAR": {}, "HAR-X": {}}
    lstm_pool = [{} for _ in cfg.seeds]
    volga_pool = [{} for _ in cfg.seeds]
    print(f"[contemp] edge={edge} h{horizon}: {panel.N} nodes, {len(folds)} folds, {len(cfg.seeds)} seeds", flush=True)
    for fi, fold in enumerate(folds):
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        adj = _fold_adj(panel, fold, wf, edge, D)
        eye = np.eye(D.N, dtype=np.float32)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS  # shared floor, sourced from config (H2)
        har, harx = _har_ols_preds(D, fl, nfloor)
        lstm = [RMR.train_masked_rich(D, cfg, s, False, eye, return_splits=True) for s in cfg.seeds]
        volga = [RMR.train_masked_rich(D, cfg, s, True, adj, return_splits=True) for s in cfg.seeds]
        pooled["HAR"].update(RMR._pred_dict(har["te"], D.y_te, D.tmask_te, D.d_te, D.N))
        pooled["HAR-X"].update(RMR._pred_dict(harx["te"], D.y_te, D.tmask_te, D.d_te, D.N))
        for si, o in enumerate(lstm):
            lstm_pool[si].update(RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N))
        for si, o in enumerate(volga):
            volga_pool[si].update(RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N))
        print(f"[contemp] fold {fi + 1}/{len(folds)} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    lstm_ens = RMR._ens(lstm_pool)
    volga_ens = RMR._ens(volga_pool)
    preds = {"HAR": pooled["HAR"], "HAR-X": pooled["HAR-X"], "LSTM": lstm_ens, "LSTM_wGAT_vol2pk": volga_ens}
    metrics = {m: RMR._metrics(preds[m], fl) for m in _MODELS}
    dm = {"VolGA_vs_LSTM": RMR._dm_all(volga_ens, lstm_ens, horizon, fl),
          "VolGA_vs_HARX": RMR._dm_all(volga_ens, preds["HAR-X"], horizon, fl)}
    result = {"edge": edge, "horizon": horizon, "market": market, "num_nodes": int(panel.N),
              "n_folds": len(folds), "folds_target": wf.folds_target, "seeds": list(cfg.seeds),
              "smoke": smoke, "seconds": time.time() - t0, "metrics": metrics, "dm_date_clustered": dm}
    print(f"[contemp] QLIKE {edge} h{horizon}: " + ", ".join(f"{m}={metrics[m]['qlike']:.4f}" for m in _MODELS), flush=True)
    print(f"[contemp] DM VolGA-vs-LSTM qlike p={dm['VolGA_vs_LSTM']['qlike']['p_value']:.3f} "
          f"({dm['VolGA_vs_LSTM']['qlike']['favors']})", flush=True)
    if not smoke:
        out = Path(out) if out else REPO / "results" / "contemp_edge" / f"contemp_{edge}_{market}_h{horizon}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[contemp] wrote {out}", flush=True)
    return result


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", choices=["contemp", "vol2pk"], default="contemp")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--market", default="vn100", choices=["vn100", "vn30", "hose", "hnx", "sp500"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.edge, a.horizon, a.folds_target, a.epochs, a.smoke, a.out, a.n_seeds, a.market)


if __name__ == "__main__":  # pragma: no cover
    main()
