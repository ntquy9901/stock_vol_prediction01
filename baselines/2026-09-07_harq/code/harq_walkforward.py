"""HARQ walk-forward: does adding a realized-quarticity measurement-error correction to HAR-X beat HAR-X
at long horizons on the EXACT canonical walk-forward folds (VN100 / VN30)?

Motivation (Bollerslev, Patton & Quaedvlieg 2016, "Exploiting the errors: A simple approach for improved
volatility forecasting"): realized variance is a noisy estimator of latent volatility, so the daily-RV
coefficient should be attenuated when its measurement error (proxied by realized quarticity RQ) is high.
HARQ makes the daily term time-varying via an interaction with sqrt(RQ). We only have daily OHLC, so RQ is
a PROXY: the rolling mean of (daily Parkinson variance)^2 (a variance-of-variance proxy).

This reuses the delivered walk-forward panel + folds + HAR-X OLS unchanged, and only ADDS the HARQ column to
the HAR-X design (HAR-X-Q). Both models are linear OLS refit per fold on TRAIN rows, share the per-node
positivity floor, and are compared by the repo's date-clustered Diebold-Mariano. No GPU, no deep model.
"""
from __future__ import annotations

import glob as _glob
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in ("baselines/2026-09-07_qlike_anchor/code", "baselines/2026-08-31_walkforward_volga/code",
           "baselines/2026-08-30_walkforward_harx_lstm/code", "baselines/2026-08-21_har_anchored_residual/code",
           "submission/soict_lstm_gat"):
    sys.path.insert(0, str(REPO / _p))
import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402

QWIN = 5                                   # rolling window for the realized-quarticity proxy (weekly)


def quarticity_panel(feats: np.ndarray) -> np.ndarray:
    """HARQ term per (day, node): har_daily * sqrt(mean_{last QWIN days} har_daily^2). ``feats`` is [T,N,5];
    channel 0 is the daily Parkinson variance. Causal (uses only days up to t)."""
    pk = feats[:, :, 0]
    rq = pd.DataFrame(pk ** 2).rolling(QWIN, min_periods=1).mean().to_numpy()
    return pk * np.sqrt(rq)


def _ols_predict(x_tr, y_tr, x_te, nfloor, y_shape):
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(x_tr)), x_tr]), y_tr, rcond=None)[0]
    pred = np.column_stack([np.ones(len(x_te)), x_te]) @ beta
    return np.maximum(pred.reshape(y_shape), nfloor)


def run(market="vn100", horizon=5, folds_target=7, out=None):
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, pc.LOOKBACK, horizon)
    panel = build_enriched_panel(files, pc.LOOKBACK, horizon, keep)
    harq = quarticity_panel(panel.feats)                   # [T,N]
    wf = VolgaWFConfig(lookback=pc.LOOKBACK, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    fl = training_config(epochs=1, seeds=(0,), batch=32).qlike_floor
    pooled = {"HAR-X": {}, "HAR-X-Q": {}}
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        mtr = D.tmask_tr.astype(bool)
        h5_tr = D.har5_tr.reshape(-1, 5)[mtr.reshape(-1)]; y_tr = D.y_tr[mtr]
        hq_tr = np.nan_to_num(harq[panel.anchors[fold.train]])            # [ntr,N] at train anchors
        hq_te = np.nan_to_num(harq[panel.anchors[fold.forecast]])
        h5_te = D.har5_te.reshape(-1, 5); yshape = D.y_te.shape
        px = _ols_predict(h5_tr, y_tr, h5_te, nfloor, yshape)             # HAR-X
        pq = _ols_predict(np.column_stack([h5_tr, hq_tr.reshape(-1)[mtr.reshape(-1)]]), y_tr,
                          np.column_stack([h5_te, hq_te.reshape(-1)]), nfloor, yshape)  # HAR-X-Q
        pooled["HAR-X"].update(RMR._pred_dict(px, D.y_te, D.tmask_te, D.d_te, D.N))
        pooled["HAR-X-Q"].update(RMR._pred_dict(pq, D.y_te, D.tmask_te, D.d_te, D.N))
    metrics = {m: RMR._metrics(pooled[m], fl) for m in pooled}
    dm = RMR._dm_all(pooled["HAR-X-Q"], pooled["HAR-X"], horizon, fl)     # <0 favours HAR-X-Q
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=str(REPO)).stdout.strip() or None
    except Exception:                                                    # pragma: no cover
        sha = None
    result = {"experiment": "harq_walkforward", "market": market, "horizon": horizon,
              "n_folds": len(folds), "qwin": QWIN, "git_commit": sha,
              "metrics": metrics, "dm_HARXQ_vs_HARX": dm}
    outp = Path(out) if out else REPO / "results" / "harq" / f"harq_{market}_h{horizon}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(outp, "w"), indent=1)
    q = {m: metrics[m]["qlike"] for m in metrics}
    print(f"[harq] {market} h{horizon}: HAR-X={q['HAR-X']:.4f} HAR-X-Q={q['HAR-X-Q']:.4f} "
          f"({(q['HAR-X'] - q['HAR-X-Q']) / q['HAR-X'] * 100:+.2f}%)  DM qlike p={dm['qlike']['p_value']:.4f} "
          f"({dm['qlike']['favors']})", flush=True)
    return result


def main():  # pragma: no cover - CLI driver
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn100")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.market, a.horizon, a.folds_target, a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
