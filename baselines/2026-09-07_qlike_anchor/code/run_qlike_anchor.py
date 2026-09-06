"""Walk-forward driver: train LSTM/VolGA under (loss in {mse,qlike}) x (anchor in {none,harx}) and compare
to HAR-X on daily Parkinson variance. Reuses the delivered panel/fold/HAR-X/metric/DM machinery read-only;
the only new pieces are the configurable trainer (qa_train) and the anchor's OOF HAR-X target (xgb_oof).

Central questions: (1) does any (loss, anchor) let the deep model beat HAR-X on QLIKE (DM date-clustered)?
(2) when trained on QLIKE loss, do MSE/RMSE/MAE worsen on test? All five metrics are reported per model.

Run: .venv_gpu_encode/Scripts/python.exe baselines/2026-09-07_qlike_anchor/code/run_qlike_anchor.py \
       --market vn100 --horizon 1 --loss qlike --anchor harx --models LSTM,VolGA --dump-cells
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-09-05_edge_horizon_matched" / "code",
           REPO / "baselines" / "2026-09-06_xgboost_residual" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
import masked_rich as MR  # noqa: E402  (EDGE_TOP_K + edge helpers live here, as in run_edge_hmatched)
from run_walkforward import _har_ols_preds, training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_edge_hmatched import (_agg_split_metrics, _cell_rows, _edge_density, _provenance, _split_dates,  # noqa: E402
                               directed_vol2pk_hmatched)
from xgb_oof import oof_harx  # noqa: E402
import qa_config as QC  # noqa: E402
import qa_train as QT  # noqa: E402


def _select_models(spec):
    """Parse a comma-sep model list into the fixed order (LSTM, VolGA); reject unknown/empty."""
    order = ("LSTM", "VolGA")
    want = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [w for w in want if w not in order]
    if bad or not want:
        raise ValueError(f"bad --models {spec!r}; choose from {order}")
    return tuple(m for m in order if m in want)


def _pool(pred, D):
    return RMR._pred_dict(pred.reshape(D.y_te.shape), D.y_te, D.tmask_te, D.d_te, D.N)


def run(market, horizon, loss, anchor, models=("LSTM", "VolGA"), lookback=10, batch=32, n_seeds=5,
        folds_target=7, epochs=16, dump_cells=False, out=None):  # pragma: no cover - GPU walk-forward driver glue
    t0 = time.time()
    sel = _select_models(models) if isinstance(models, str) else tuple(models)
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    cfg = training_config(epochs=epochs, seeds=(42, 123, 2026, 7, 2024)[:n_seeds], batch=batch)
    fl = cfg.qlike_floor
    pooled = {m: {} for m in ("HAR", "HAR-X")}
    pooled_nn = {m: [{} for _ in cfg.seeds] for m in sel}
    tr_acc = {m: [] for m in sel}; va_acc = {m: [] for m in sel}; curves = {m: [] for m in sel}
    dens = []; cell_rows = []
    print(f"[qa] {market} h{horizon} loss={loss} anchor={anchor}: {panel.N} nodes, {len(folds)} folds, "
          f"{len(cfg.seeds)} seeds, models={list(sel)}", flush=True)
    for fi, fold in enumerate(folds):
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        har, harx = _har_ols_preds(D, fl, nfloor)
        pooled["HAR"].update(_pool(har["te"], D)); pooled["HAR-X"].update(_pool(harx["te"], D))
        harx_dict = None
        if anchor == "harx":
            oof = oof_harx(D.har5_tr, D.y_tr, D.tmask_tr.astype(bool), fl, nfloor)   # [A_tr,N], NaN on warm-up
            oof = np.where(np.isfinite(oof), oof, harx["tr"])                        # in-sample fallback on warm-up
            harx_dict = {"tr": oof, "va": harx["va"], "te": harx["te"]}
        eye = np.eye(D.N, dtype=np.float32)
        adj_map = {"LSTM": eye}
        if "VolGA" in sel:
            adj = directed_vol2pk_hmatched(panel.feats[:, :, 4], np.sqrt(panel.pk),
                                           int(panel.anchors[fold.train][-1]) + wf.horizon, wf.horizon, MR.EDGE_TOP_K)
            adj_map["VolGA"] = adj; dens.append(_edge_density(adj))
        fold_out = {m: [] for m in sel}
        for si, s in enumerate(cfg.seeds):
            for m in sel:
                o = QT.train_deep(D, cfg, s, m == "VolGA", adj_map[m], loss, anchor, harx_dict,
                                  QC.ANCHOR_CLIP, return_splits=True)
                pooled_nn[m][si].update(_pool(o["test"], D)); fold_out[m].append(o)
            print(f"[qa]   fold {fi + 1}/{len(folds)} seed {si + 1}/{len(cfg.seeds)} done "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        for m in sel:
            etr = RMR._ens_split(fold_out[m], "train"); eva = RMR._ens_split(fold_out[m], "val")
            tr_acc[m].append(RMR._split_metrics(etr, D.y_tr, D.tmask_tr, fl))
            va_acc[m].append(RMR._split_metrics(eva, D.y_va, D.tmask_va, fl))
            curves[m].append({"fold": fi, "train": [o["train_curve"] for o in fold_out[m]],
                              "val": [o["val_curve"] for o in fold_out[m]], "best_epoch": [o["best_epoch"] for o in fold_out[m]]})
        if dump_cells:
            for sname, sp, fsl in (("train", "tr", fold.train), ("val", "va", fold.val), ("test", "te", fold.forecast)):
                y = getattr(D, f"y_{sp}"); tm = getattr(D, f"tmask_{sp}"); dts = _split_dates(panel, fsl)
                cell_rows += _cell_rows(RMR._pred_dict(har[sp], y, tm, dts, D.N), "HAR", sname, fi, panel.tickers)
                cell_rows += _cell_rows(RMR._pred_dict(harx[sp], y, tm, dts, D.N), "HAR-X", sname, fi, panel.tickers)
                for m in sel:
                    cell_rows += _cell_rows(RMR._pred_dict(RMR._ens_split(fold_out[m], sname), y, tm, dts, D.N),
                                            m, sname, fi, panel.tickers)
        print(f"[qa] fold {fi + 1}/{len(folds)} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    for m in sel:
        pooled[m] = RMR._ens(pooled_nn[m])
    report_models = ("HAR", "HAR-X") + sel
    metrics = {m: RMR._metrics(pooled[m], fl) for m in report_models}
    train_metrics = {m: _agg_split_metrics(tr_acc[m]) for m in sel}
    val_metrics = {m: _agg_split_metrics(va_acc[m]) for m in sel}
    fit_diagnostics = {m: RMR.OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in sel}
    dm = {f"{m}_vs_HAR-X": RMR._dm_all(pooled[m], pooled["HAR-X"], horizon, fl) for m in sel}
    try:
        gitsha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=str(REPO)).stdout.strip() or None
    except Exception:
        gitsha = None
    prov = _provenance(lookback, batch, epochs, fl, MR.EDGE_TOP_K, 2.0)
    prov.update({"loss": loss, "anchor": anchor, "anchor_clip": QC.ANCHOR_CLIP})
    result = {"experiment": "qlike_anchor", "market": market, "horizon": horizon, "loss": loss, "anchor": anchor,
              "num_nodes": int(panel.N), "n_folds": len(folds), "seeds": list(cfg.seeds), "git_commit": gitsha,
              "config": prov, "edge_density_mean": float(np.mean(dens)) if dens else None,
              "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
              "fit_diagnostics": fit_diagnostics, "learning_curves": curves, "dm_date_clustered": dm,
              "seconds": time.time() - t0}
    print(f"[qa] QLIKE {market} h{horizon} {loss}/{anchor}: "
          + ", ".join(f"{m}={metrics[m]['qlike']:.4f}" for m in report_models), flush=True)
    for m in sel:
        q = dm[f"{m}_vs_HAR-X"]["qlike"]
        print(f"[qa]   DM {m} vs HAR-X: qlike p={q['p_value']:.3f} ({q['favors']}) | fit={fit_diagnostics[m]['status']}", flush=True)
    out = Path(out) if out else REPO / "results" / "qlike_anchor" / f"qa_{market}_{loss}_{anchor}_h{horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"[qa] wrote {out}", flush=True)
    if dump_cells and cell_rows:
        import pandas as pd
        cp = REPO / "results" / "qlike_anchor" / "cells" / f"cells_{market}_{loss}_{anchor}_h{horizon}.parquet"
        cp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cell_rows).to_parquet(cp, index=False)
        print(f"[qa] wrote {len(cell_rows)} cell-log rows -> {cp}", flush=True)
    return result


def main():  # pragma: no cover - entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn100")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--loss", default="qlike", choices=list(QC.LOSSES))
    ap.add_argument("--anchor", default="none", choices=list(QC.ANCHORS))
    ap.add_argument("--models", default="LSTM,VolGA")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--folds-target", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--dump-cells", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.market, a.horizon, a.loss, a.anchor, _select_models(a.models), a.lookback, a.batch,
        a.n_seeds, a.folds_target, a.epochs, a.dump_cells, a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
