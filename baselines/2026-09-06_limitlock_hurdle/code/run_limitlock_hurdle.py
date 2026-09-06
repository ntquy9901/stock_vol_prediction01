"""Does a limit-lock hurdle on the QLIKE champion (HAR-X) lower QLIKE by not over-forecasting variance
on limit-lock / zero-range days (the days that blow up QLIKE)?

Per walk-forward fold (reusing the delivered panel/fold machinery read-only): fit (B) a HAR-X OLS on the
5 HAR-X features and (A) a logistic classifier on 3 causal limit-lock features predicting the target-day
``zero_range_flag``. Combine: on test cells with P(lock) >= threshold, override the HAR-X variance
forecast with the shared per-node positivity floor; else keep HAR-X. Score QLIKE (all / non-lock-robust /
lock-days-only) + standard metrics pooled across folds, and a date-clustered DM (hurdle vs plain HAR-X).
CPU-only (OLS + logistic); the GPU is untouched.

Run: .venv_gpu_encode/Scripts/python.exe baselines/2026-09-06_limitlock_hurdle/code/run_limitlock_hurdle.py --market vn30 --horizon 1
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
import limitlock_config as LC  # noqa: E402
from limitlock_features import compute_limitlock_features  # noqa: E402

_CASE_DATE = "2025-04-10"   # the VN30 market-wide limit-lock day to trace concretely


def ols_fit_predict(x_tr, y_tr, x_te, floor):
    """Least-squares fit (with intercept) on train rows, floored prediction on test rows."""
    a_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    coef = np.linalg.lstsq(a_tr, y_tr, rcond=None)[0]
    a_te = np.column_stack([np.ones(len(x_te)), x_te])
    return np.maximum(a_te @ coef, floor)


def _har_design(har5):
    """Flatten the [A,N,5] HAR-X design to [A*N, 5]."""
    a, n, _ = har5.shape
    return har5.reshape(a * n, 5)


def logistic_fit_predict_proba(x_tr, y_tr, x_te):
    """Train a logistic classifier on standardized limit-lock features; return P(lock) over ``x_te`` rows.

    A degenerate single-class train target (no lock days, or all lock days) has no decision boundary to
    learn, so return that class's constant base rate (0.0 or 1.0) rather than crash -- a graceful no-op
    that never fires the override when there were no train locks."""
    y_tr = np.asarray(y_tr, dtype=float)
    classes = np.unique(y_tr)
    if len(classes) < 2:
        return np.full(len(x_te), float(classes[0]) if len(classes) else 0.0)
    mu = x_tr.mean(0)
    sd = x_tr.std(0) + pc.SCALER_EPS
    clf = LogisticRegression(max_iter=1000)  # config-ok: solver iteration cap, not a result-affecting tunable
    clf.fit((x_tr - mu) / sd, y_tr)
    return clf.predict_proba((x_te - mu) / sd)[:, 1]


def apply_hurdle(harx_pred, lock_proba, threshold, lock_value):
    """Override the HAR-X forecast with ``lock_value`` on cells where ``lock_proba >= threshold``.

    ``lock_value`` may be a scalar or an [N]/[1,N] per-node array (broadcast against the [A,N] forecast)."""
    return np.where(lock_proba >= threshold, lock_value, harx_pred)


def _pooled(pred_flat, D, split):
    y = getattr(D, f"y_{split}"); tm = getattr(D, f"tmask_{split}"); dts = getattr(D, f"d_{split}")
    return RMR._pred_dict(pred_flat.reshape(y.shape), y, tm, dts, D.N)


def _subset(pred, keys, inside):
    """Sub-dict of ``pred`` restricted to (inside=True) / excluding (inside=False) ``keys``."""
    return {k: v for k, v in pred.items() if (k in keys) == inside}


def _confusion(y_true, proba, threshold):
    """Binary confusion counts of the lock classifier at ``threshold`` over the given cells."""
    pos = proba >= threshold
    yt = np.asarray(y_true).astype(bool)
    return {"tp": int((pos & yt).sum()), "fp": int((pos & ~yt).sum()),
            "fn": int((~pos & yt).sum()), "tn": int((~pos & ~yt).sum())}


def _metrics_full(pred, floor, lock_keys):
    """Standard metrics + qlike_robust (non-lock target cells) + qlike_lockdays (lock cells only)."""
    m = dict(RMR._metrics(pred, floor))
    nonlock = _subset(pred, lock_keys, inside=False)
    lockonly = _subset(pred, lock_keys, inside=True)
    m["qlike_robust"] = RMR._metrics(nonlock, floor)["qlike"] if nonlock else None
    m["qlike_lockdays"] = RMR._metrics(lockonly, floor)["qlike"] if lockonly else None
    m["n_lockdays"] = len(lockonly)
    return m


def _read_aligned(files, panel):
    """Read ``daily_return`` and ``zero_range_flag`` aligned to the panel's dates x tickers -> two [T,N]
    arrays (NaN off a ticker's own dates). Only the panel's kept tickers are read."""
    keep = set(panel.tickers)
    ret = np.full((len(panel.dates), panel.N), np.nan)
    lock = np.full((len(panel.dates), panel.N), np.nan)
    idx = {t: j for j, t in enumerate(panel.tickers)}
    for f in files:
        tk = Path(f).stem
        if tk not in keep:
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
        r = df["daily_return"].reindex(panel.dates).to_numpy(dtype=float)
        z = df["zero_range_flag"].astype(float).reindex(panel.dates).to_numpy(dtype=float)
        ret[:, idx[tk]] = r
        lock[:, idx[tk]] = z
    return ret, lock


def _panel_limitlock(ret, lock, n_nodes):
    """Per-ticker causal limit-lock features -> [T,N,3] (stacked over nodes)."""
    return np.stack([compute_limitlock_features(ret[:, j], lock[:, j], LC.LOCK_WINDOW,
                                                LC.LIMIT_FRAC, LC.NEAR_LIMIT_MULT)
                     for j in range(n_nodes)], axis=1)


def run(market, horizon, folds_target=7, lookback=22, out=None):  # pragma: no cover - driver glue
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    ret, lock = _read_aligned(files, panel)
    ll = _panel_limitlock(ret, lock, panel.N)                       # [T,N,3] causal
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    fl = training_config().qlike_floor
    thresholds = list(LC.LOCK_THRESHOLDS)
    # "oracle" = a perfect classifier (override on the TRUE target lock flag): the ceiling of the hurdle
    # idea itself, isolating whether the LOGISTIC classifier is the bottleneck vs the idea being unhelpful.
    models = ["HAR-X"] + [f"HAR-X+hurdle@{t}" for t in thresholds] + ["HAR-X+hurdle@oracle"]
    pooled = {m: {} for m in models}
    lock_keys = set()
    conf = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in thresholds}
    case_records = []
    for fi, fold in enumerate(folds):
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = (pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS)[None, :]   # [1,N] shared positivity floor
        mtr = D.tmask_tr.astype(bool).reshape(-1)
        ytr = D.y_tr.reshape(-1)
        # (B) HAR-X OLS variance forecast, floored identically for both compared models (H2)
        harx = ols_fit_predict(_har_design(D.har5_tr)[mtr], ytr[mtr], _har_design(D.har5_te), fl)
        harx = np.maximum(harx.reshape(D.y_te.shape), nfloor)
        pooled["HAR-X"].update(_pooled(harx.reshape(-1), D, "te"))
        # (A) lock classifier: predict zero_range_flag at t+h from causal limit-lock features
        a_tr = panel.anchors[fold.train]; a_te = panel.anchors[fold.forecast]
        lock_tr = lock[a_tr + horizon]; lock_te = lock[a_te + horizon]
        x_tr = ll[a_tr].reshape(-1, 3)[mtr]
        y_lock = np.nan_to_num(lock_tr).reshape(-1)[mtr]
        proba = logistic_fit_predict_proba(x_tr, y_lock, ll[a_te].reshape(-1, 3)).reshape(D.y_te.shape)
        mte = D.tmask_te.astype(bool)
        lock_te0 = np.nan_to_num(lock_te)
        lk = RMR._pred_dict(lock_te0, lock_te0, D.tmask_te, D.d_te, D.N)
        lock_keys |= {k for k, v in lk.items() if v[0] >= 0.5}
        yt = lock_te0[mte]; pr = proba[mte]
        for t in thresholds:
            c = _confusion(yt, pr, t)
            for kk in conf[t]:
                conf[t][kk] += c[kk]
            hp = np.maximum(apply_hurdle(harx, proba, t, nfloor), nfloor)
            pooled[f"HAR-X+hurdle@{t}"].update(_pooled(hp.reshape(-1), D, "te"))
        hp_or = np.maximum(apply_hurdle(harx, lock_te0, 0.5, nfloor), nfloor)   # perfect-classifier ceiling
        pooled["HAR-X+hurdle@oracle"].update(_pooled(hp_or.reshape(-1), D, "te"))
        # concrete 2025-04-10 case
        if _CASE_DATE in list(D.d_te):
            import metrics as M
            ai = list(D.d_te).index(_CASE_DATE)
            for j in range(D.N):
                if mte[ai, j]:
                    yv = float(D.y_te[ai, j]); pv = float(harx[ai, j]); prj = float(proba[ai, j])
                    hv = float(np.maximum(apply_hurdle(pv, prj, 0.7, nfloor[0, j]), nfloor[0, j]))
                    case_records.append({
                        "ticker": panel.tickers[j], "is_lock": bool(lock_te0[ai, j] >= 0.5),
                        "p_lock": prj, "y_true": yv, "harx_pred": pv, "hurdle_pred": hv,
                        "qlike_harx": float(M.per_obs_qlike(np.array([yv]), np.array([pv]), fl)[0]),
                        "qlike_hurdle": float(M.per_obs_qlike(np.array([yv]), np.array([hv]), fl)[0])})
        print(f"[limitlock] {market} h{horizon} fold {fi + 1}/{len(folds)} done", flush=True)

    metrics = {m: _metrics_full(pooled[m], fl, lock_keys) for m in models}
    dm = {m: RMR._dm_all(pooled[m], pooled["HAR-X"], horizon, fl) for m in models if m != "HAR-X"}
    caught = sum(1 for r in case_records if r["is_lock"] and r["p_lock"] >= 0.7)
    case = {"target_date": _CASE_DATE, "n_records": len(case_records),
            "n_lock": sum(r["is_lock"] for r in case_records), "n_lock_caught_at_0.7": caught,
            "qlike_harx_sum": float(sum(r["qlike_harx"] for r in case_records if r["is_lock"])),
            "qlike_hurdle_sum": float(sum(r["qlike_hurdle"] for r in case_records if r["is_lock"])),
            "records": case_records}
    result = {"experiment": "limitlock_hurdle", "market": market, "horizon": horizon,
              "num_nodes": int(panel.N), "n_folds": len(folds), "n_total_lock_targets": len(lock_keys),
              "limit_frac": LC.LIMIT_FRAC, "near_limit_mult": LC.NEAR_LIMIT_MULT,
              "lock_window": LC.LOCK_WINDOW, "thresholds": thresholds,
              "metrics": metrics, "classifier_confusion": conf, "dm_hurdle_vs_harx": dm, "case_2025_04_10": case}
    print(f"[limitlock] {market} h{horizon} QLIKE HAR-X={metrics['HAR-X']['qlike']:.4f}", flush=True)
    for label in [f"HAR-X+hurdle@{t}" for t in thresholds] + ["HAR-X+hurdle@oracle"]:
        mm = metrics[label]
        print(f"    {label}: qlike={mm['qlike']:.4f} robust={mm['qlike_robust']} "
              f"lockdays={mm['qlike_lockdays']} | DM p={dm[label]['qlike'].get('p_value')}", flush=True)
    out = Path(out) if out else REPO / "results" / "limitlock_hurdle" / f"hurdle_{market}_h{horizon}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[limitlock] wrote {out}", flush=True)
    return result


def main():  # pragma: no cover - entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn30")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=7)
    ap.add_argument("--lookback", type=int, default=22)
    a = ap.parse_args()
    run(a.market, a.horizon, a.folds_target, a.lookback)


if __name__ == "__main__":  # pragma: no cover
    main()
