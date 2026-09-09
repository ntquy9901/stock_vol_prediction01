"""Gamma-loss gradient-boosted HAR: a nonlinear, QLIKE-trained forecaster that beats HAR-X, evaluated on the
exact canonical walk-forward folds.

Key idea: the QLIKE loss equals the gamma unit deviance up to a constant (QLIKE(y,f)=y/f-log(y/f)-1), so a
gradient-boosting regressor with ``loss='gamma'`` is trained directly on the evaluation metric. Features are
the 5 HAR-X features plus a realized-quarticity proxy (measurement error, HARQ idea) and causal
"vol-is-declining" features (log-vol momentum/slope/deviation). It reuses the delivered panel + folds + the
canonical HAR-X OLS for comparison. Single GBM pooled over all tickers per fold; per-node positivity floor;
date-clustered Diebold-Mariano vs HAR-X.
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
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path(__file__).resolve().parents[3]
for _p in ("baselines/2026-09-07_harq/code", "baselines/2026-08-31_walkforward_volga/code",
           "baselines/2026-08-30_walkforward_harx_lstm/code", "baselines/2026-08-21_har_anchored_residual/code",
           "submission/soict_lstm_gat"):
    sys.path.insert(0, str(REPO / _p))
import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402

WK, MO = 5, 22


def extra_feature_panels(feats):
    """Causal RQ + log-vol decline feature panels [T,N] from the daily Parkinson channel feats[:,:,0]."""
    pk = feats[:, :, 0]
    lpk = pd.DataFrame(np.log(np.maximum(pk, pc.QLIKE_FLOOR)))
    rq = np.sqrt(pd.DataFrame(pk ** 2).rolling(WK, min_periods=1).mean().to_numpy())
    return {
        "rq": rq,
        "mr_change": lpk.diff(1).to_numpy(),
        "mr_slope5": ((lpk - lpk.shift(WK)) / float(WK)).to_numpy(),
        "mr_slope10": ((lpk - lpk.shift(2 * WK)) / float(2 * WK)).to_numpy(),
        "mr_dev5": (lpk - lpk.rolling(WK).mean()).to_numpy(),
        "mr_z22": ((lpk - lpk.rolling(MO).mean()) / (lpk.rolling(MO).std() + pc.QLIKE_FLOOR)).to_numpy(),
    }


EXTRA_KEYS = ["rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"]


def _design(har5, extras, anchors):
    """[n_anchor*N, 5+len(extras)] feature matrix: 5 HAR features + extras, at the given anchor rows."""
    n, N = har5.shape[0], har5.shape[1]
    cols = [har5.reshape(n * N, 5)]
    for k in EXTRA_KEYS:
        cols.append(np.nan_to_num(extras[k][anchors]).reshape(n * N, 1))
    return np.column_stack(cols)


def _harx_ols(D, nfloor):
    mtr = D.tmask_tr.astype(bool)
    xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr.reshape(-1, 5)[mtr.reshape(-1)]])
    cx = np.linalg.lstsq(xtr, D.y_tr[mtr], rcond=None)[0]
    f5 = D.har5_te.reshape(-1, 5)
    return np.maximum((np.column_stack([np.ones(len(f5)), f5]) @ cx).reshape(D.y_te.shape), nfloor)


def run(market="sp500_clean", horizon=5, folds_target=7, out=None):
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, pc.LOOKBACK, horizon)
    panel = build_enriched_panel(files, pc.LOOKBACK, horizon, keep)
    extras = extra_feature_panels(panel.feats)
    wf = VolgaWFConfig(lookback=pc.LOOKBACK, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    fl = training_config(epochs=1, seeds=(0,), batch=32).qlike_floor
    pooled = {"HAR-X": {}, "GBM": {}}
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        pooled["HAR-X"].update(RMR._pred_dict(_harx_ols(D, nfloor), D.y_te, D.tmask_te, D.d_te, D.N))
        mtr = D.tmask_tr.astype(bool).reshape(-1)
        xtr = _design(D.har5_tr, extras, panel.anchors[fold.train])[mtr]
        ytr = np.maximum(D.y_tr.reshape(-1)[mtr], fl)                     # gamma loss needs y>0
        gbm = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05,
                                            max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        gbm.fit(xtr, ytr)
        pte = np.maximum(gbm.predict(_design(D.har5_te, extras, panel.anchors[fold.forecast])).reshape(D.y_te.shape),
                         nfloor)
        pooled["GBM"].update(RMR._pred_dict(pte, D.y_te, D.tmask_te, D.d_te, D.N))
    metrics = {m: RMR._metrics(pooled[m], fl) for m in pooled}
    dm = RMR._dm_all(pooled["GBM"], pooled["HAR-X"], horizon, fl)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=str(REPO)).stdout.strip() or None
    except Exception:                                                    # pragma: no cover
        sha = None
    result = {"experiment": "gamma_gbm", "market": market, "horizon": horizon, "n_folds": len(folds),
              "features": ["har5"] + EXTRA_KEYS, "git_commit": sha, "metrics": metrics, "dm_GBM_vs_HARX": dm}
    outp = Path(out) if out else REPO / "results" / "gamma_gbm" / f"gbm_{market}_h{horizon}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(outp, "w"), indent=1)
    q = {m: metrics[m]["qlike"] for m in metrics}
    print(f"[gbm] {market} h{horizon}: HAR-X={q['HAR-X']:.4f} GBM={q['GBM']:.4f} "
          f"({(q['HAR-X'] - q['GBM']) / q['HAR-X'] * 100:+.2f}%)  DM p={dm['qlike']['p_value']:.4f} "
          f"({dm['qlike']['favors']})", flush=True)
    return result


def main():  # pragma: no cover - CLI driver
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="sp500_clean")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.market, a.horizon, a.folds_target, a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
