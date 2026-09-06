"""Does adding causal market-regime features to the strongest baseline (HAR-X) improve QLIKE?

Tests the Tier-1 shock-detection hypothesis on the QLIKE champion without touching the deep-model architecture:
per walk-forward fold, fit an OLS on the 5 HAR-X features and on HAR-X + the 3 regime features (regime_features),
score both on the pooled test set, and compare with a date-clustered Diebold-Mariano test. Reuses the delivered
panel/fold machinery read-only. If regime features help HAR-X here, integrating them into the LSTM/VolGA input
is the justified next step; if not, that surgery is not worth it.

Run: .venv_gpu_encode/Scripts/python.exe baselines/2026-09-06_regime_features/code/run_regime_har.py --market vn30 --horizon 1
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_folds import make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
import regime_features as RGF  # noqa: E402


def ols_fit_predict(x_tr, y_tr, x_te, floor):
    """Least-squares fit (with intercept) on train rows, floored prediction on test rows."""
    a_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    coef = np.linalg.lstsq(a_tr, y_tr, rcond=None)[0]
    a_te = np.column_stack([np.ones(len(x_te)), x_te])
    return np.maximum(a_te @ coef, floor)


def _design(har5, regime_rows, use_regime):
    """Flatten [A,N,5] HAR-X design (+ broadcast [A,3] regime to [A,N,3]) to [A*N, k]."""
    a, n, _ = har5.shape
    if not use_regime:
        return har5.reshape(a * n, 5)
    reg = np.broadcast_to(regime_rows[:, None, :], (a, n, regime_rows.shape[1]))
    return np.concatenate([har5, reg], axis=2).reshape(a * n, 5 + regime_rows.shape[1])


def _pooled(pred_flat, D, split):
    y = getattr(D, f"y_{split}"); tm = getattr(D, f"tmask_{split}"); dts = getattr(D, f"d_{split}")
    return RMR._pred_dict(pred_flat.reshape(y.shape), y, tm, dts, D.N)


def run(market, horizon, folds_target=7, lookback=22, window=None, out=None):  # pragma: no cover - driver glue
    window = pc.HAR_MONTHLY_WINDOW if window is None else window
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    market_series = np.nanmean(panel.feats[:, :, 3], axis=1)          # market_pk aggregate per day [T]
    regime_all = RGF.compute_regime_features(market_series, window)   # [T,3], causal
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    fl = training_config().qlike_floor
    pooled = {"HAR-X": {}, "HAR-X+regime": {}}
    for fi, fold in enumerate(folds):
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        reg_tr = regime_all[panel.anchors[fold.train]]
        reg_te = regime_all[panel.anchors[fold.forecast]]
        mtr = D.tmask_tr.astype(bool).reshape(-1)
        ytr = D.y_tr.reshape(-1)
        for name, use in (("HAR-X", False), ("HAR-X+regime", True)):
            xtr = _design(D.har5_tr, reg_tr, use)[mtr]
            xte = _design(D.har5_te, reg_te, use)
            pred = ols_fit_predict(xtr, ytr[mtr], xte, fl)
            pred = np.maximum(pred.reshape(D.y_te.shape), nfloor).reshape(-1)
            pooled[name].update(_pooled(pred, D, "te"))
        print(f"[regime] {market} h{horizon} fold {fi + 1}/{len(folds)} done", flush=True)
    metrics = {m: RMR._metrics(pooled[m], fl) for m in pooled}
    dm = RMR._dm_all(pooled["HAR-X+regime"], pooled["HAR-X"], horizon, fl)
    result = {"experiment": "regime_features_har", "market": market, "horizon": horizon,
              "num_nodes": int(panel.N), "n_folds": len(folds), "regime_window": window,
              "metrics": metrics, "dm_regime_vs_harx": dm}
    print(f"[regime] {market} h{horizon} QLIKE: HAR-X={metrics['HAR-X']['qlike']:.4f} "
          f"HAR-X+regime={metrics['HAR-X+regime']['qlike']:.4f} "
          f"| DM p={dm['qlike']['p_value']:.3f} ({dm['qlike']['favors']})", flush=True)
    out = Path(out) if out else REPO / "results" / "regime_features" / f"regime_{market}_h{horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[regime] wrote {out}", flush=True)
    return result


def main():  # pragma: no cover - entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn30")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=7)
    a = ap.parse_args()
    run(a.market, a.horizon, a.folds_target)


if __name__ == "__main__":  # pragma: no cover
    main()
