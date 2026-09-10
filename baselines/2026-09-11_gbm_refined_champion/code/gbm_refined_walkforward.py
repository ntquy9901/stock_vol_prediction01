"""Refined SP500 champion: gamma-GBM + forward-looking earnings, consolidated with three validated feature
layers, on the canonical S&P 500 walk-forward (same panel/folds/HAR-X/DM/metrics as the committed
baselines/2026-09-09_gamma_gbm_earnings). The three layers were each shown as significant probes and are here
gathered into ONE walk-forward model, measured apples-to-apples vs HAR-X and vs the committed GBM+earn:

  1. ASYMMETRIC earnings  - earn_pre (approach ramp to next scheduled release) + earn_post (longer post-release
     decay for post-earnings-drift), added to the symmetric earn_prox/earn_soon.
  2. ESTIMATORS           - Garman-Klass, Rogers-Satchell, Yang-Zhang(n=20) daily variance (origin-time), the
     multi-estimator measurement-error channel; loaded from the enriched CSVs, aligned to each anchor's day t.
  3. SPIKE-PROPENSITY     - per-stock vol-of-vol + spike frequency/magnitude + log-vol acceleration (own-stock,
     trailing-window causal), which help the elevated-but-not-extreme deciles (d6-d9).

All features observable at forecast time t (earnings distances measured to SCHEDULED dates known ahead; all
rolling windows trail t). HistGradientBoostingRegressor(loss='gamma') handles NaN natively. Reports HAR-X,
GBM+earn (committed champion), GBM+refined; date-clustered DM refined-vs-HAR-X and refined-vs-GBM+earn; 6 metrics;
TRAIN vs TEST QLIKE as overfit evidence."""
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
for _p in ("baselines/2026-09-09_gamma_gbm_earnings/code", "baselines/2026-09-08_gamma_gbm/code",
           "baselines/2026-09-07_harq/code", "baselines/2026-08-31_walkforward_volga/code",
           "baselines/2026-08-30_walkforward_harx_lstm/code", "baselines/2026-08-21_har_anchored_residual/code",
           "submission/soict_lstm_gat"):
    sys.path.insert(0, str(REPO / _p))
import gamma_gbm_walkforward as G  # noqa: E402
import gbm_earnings_walkforward as GE  # noqa: E402
import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402

FL = pc.QLIKE_FLOOR
WK, MO = 5, 22
SPIKE_WIN = 3 * MO                          # 66-day trailing window for spike-frequency/magnitude
PRE_RAMP, POST_RAMP = GE.RAMP, 2 * GE.RAMP  # asym-earn approach ramp (=5d) / post-release decay (=10d)
EST_COLS = ["garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20"]
SPIKE_KEYS = ["vov22", "spike_rate63", "spike_mag63", "accel10"]
EARN4 = ["earn_prox", "earn_soon", "earn_pre", "earn_post"]


def _signed_dist(tgt, ed):
    """(days_to_next_release>=0, days_since_last_release>=0) capped, for each target day."""
    if ed is None or len(ed) == 0:
        cap = np.full(len(tgt), GE.CAP)
        return cap, cap
    pos = np.searchsorted(ed, tgt)
    nxt = np.where(pos < len(ed), (ed[np.clip(pos, 0, len(ed) - 1)] - tgt) / np.timedelta64(1, "D"), GE.CAP)
    prv = np.where(pos > 0, (tgt - ed[np.clip(pos - 1, 0, len(ed) - 1)]) / np.timedelta64(1, "D"), GE.CAP)
    return np.clip(np.maximum(nxt, 0.0), 0, GE.CAP), np.clip(np.maximum(prv, 0.0), 0, GE.CAP)


def asym_earnings_panels(panel, earn):
    """{earn_prox, earn_soon, earn_pre, earn_post} [A, N] from each anchor's TARGET date to scheduled earnings."""
    tgt = pd.to_datetime(panel.target_dates).values.astype("datetime64[D]")
    N, A = panel.N, len(tgt)
    out = {k: np.zeros((A, N), np.float32) for k in EARN4}
    for j, tk in enumerate(panel.tickers):
        ed = earn.get(tk)
        nxt, prv = _signed_dist(tgt, ed); dist = np.minimum(nxt, prv)
        out["earn_prox"][:, j] = np.maximum(0.0, 1.0 - dist / GE.RAMP)
        out["earn_soon"][:, j] = (dist <= GE.SOON).astype(np.float32)
        out["earn_pre"][:, j] = np.maximum(0.0, 1.0 - nxt / PRE_RAMP)
        out["earn_post"][:, j] = np.maximum(0.0, 1.0 - prv / POST_RAMP)
    return out


def spike_panels(panel):
    """per-stock spike-propensity feature panels [T, N] from the daily Parkinson channel (own-stock, causal)."""
    pk = pd.DataFrame(panel.pk); lpk = pd.DataFrame(np.log(np.maximum(panel.pk, FL)))
    med = pk.rolling(MO, min_periods=5).median()
    return {
        "vov22": lpk.rolling(MO).std().to_numpy(),
        "spike_rate63": (pk > 2 * med).rolling(SPIKE_WIN, min_periods=10).mean().to_numpy(),
        "spike_mag63": (pk.rolling(SPIKE_WIN, min_periods=10).max() / (med + FL)).to_numpy(),
        "accel10": ((lpk - lpk.shift(WK)) - (lpk.shift(WK) - lpk.shift(2 * WK))).to_numpy(),
    }


def estimator_panels(panel, files):
    """[T, N, 3] Garman-Klass / Rogers-Satchell / Yang-Zhang daily variance aligned to panel.dates (origin t)."""
    fmap = {Path(f).stem: f for f in files}
    E = np.full((len(panel.dates), panel.N, len(EST_COLS)), np.nan, np.float32)
    for j, tk in enumerate(panel.tickers):
        d = pd.read_csv(fmap[tk], parse_dates=["date"]).set_index("date")
        avail = [c for c in EST_COLS if c in d.columns]
        E[:, j, [EST_COLS.index(c) for c in avail]] = d.reindex(panel.dates)[avail].to_numpy(float)
    return E


def _design_refined(har5, extras, est, spk, asym, anchors, pos):
    """[n*N, 5+6+3+4+4] = HAR-X(5) + extras(6) + estimators(3) + spike(4) + asym-earn(4).
    HAR/extras/estimators/spike index the [T,N] axis via ``anchors``; earnings panels index [A,N] via ``pos``."""
    X = G._design(har5, extras, anchors)                       # 5 HAR + 6 extras
    n, N = har5.shape[0], har5.shape[1]
    ea = est[anchors]                                          # [n, N, 3]
    for c in range(len(EST_COLS)):
        X = np.column_stack([X, ea[:, :, c].reshape(n * N, 1)])
    for k in SPIKE_KEYS:
        X = np.column_stack([X, spk[k][anchors].reshape(n * N, 1)])
    for k in EARN4:
        X = np.column_stack([X, asym[k][pos].reshape(n * N, 1)])
    return X


def _gbm_fit(xtr, ytr):
    return HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0).fit(xtr, ytr)


def run(market="sp500_clean", horizon=5, folds_target=7, out=None):  # pragma: no cover - SP500-only ~20-min driver; logic is in the unit-tested helpers + committed results JSON
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, pc.LOOKBACK, horizon)
    panel = build_enriched_panel(files, pc.LOOKBACK, horizon, keep)
    extras = G.extra_feature_panels(panel.feats)
    earn = GE._earn_by_ticker(); asym = asym_earnings_panels(panel, earn)
    est = estimator_panels(panel, files); spk = spike_panels(panel)
    wf = VolgaWFConfig(lookback=pc.LOOKBACK, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    fl = training_config(epochs=1, seeds=(0,), batch=32).qlike_floor
    pooled = {m: {} for m in ("HAR-X", "GBM+earn", "GBM+refined")}
    tr_q = {"GBM+earn": [], "GBM+refined": []}                  # per-fold TRAIN QLIKE (overfit evidence)
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        nf_te = np.broadcast_to(nfloor, D.y_te.shape).reshape(-1)
        pooled["HAR-X"].update(RMR._pred_dict(G._harx_ols(D, nfloor), D.y_te, D.tmask_te, D.d_te, D.N))
        mtr = D.tmask_tr.astype(bool).reshape(-1); ytr = np.maximum(D.y_tr.reshape(-1)[mtr], fl)
        for tag, cols_fn in (("GBM+earn", _earn_design), ("GBM+refined", _design_refined)):
            xtr = cols_fn(D.har5_tr, extras, est, spk, asym, panel.anchors[fold.train], fold.train)[mtr]
            xte = cols_fn(D.har5_te, extras, est, spk, asym, panel.anchors[fold.forecast], fold.forecast)
            m = _gbm_fit(xtr, ytr)
            pte = np.maximum(m.predict(xte), nf_te)
            pooled[tag].update(RMR._pred_dict(pte.reshape(D.y_te.shape), D.y_te, D.tmask_te, D.d_te, D.N))
            ptr = np.maximum(m.predict(xtr), fl)             # masked train preds (align with masked ytr)
            tr_q[tag].append(float(np.mean(ytr / ptr - np.log(ytr / ptr) - 1)))  # train QLIKE, overfit evidence
    metrics = {m: RMR._metrics(pooled[m], fl) for m in pooled}
    dm_h = {m: RMR._dm_all(pooled[m], pooled["HAR-X"], horizon, fl) for m in ("GBM+earn", "GBM+refined")}
    dm_e = {"GBM+refined": RMR._dm_all(pooled["GBM+refined"], pooled["GBM+earn"], horizon, fl)}
    overfit = {m: {"train_qlike": float(np.mean(tr_q[m])), "test_qlike": metrics[m]["qlike"],
                   "train_to_test_gap_pct": (metrics[m]["qlike"] - np.mean(tr_q[m])) / np.mean(tr_q[m]) * 100}
               for m in tr_q}
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=str(REPO)).stdout.strip() or None
    except Exception:                                                    # pragma: no cover
        sha = None
    res = {"experiment": "gbm_refined_champion", "market": market, "horizon": horizon, "n_folds": len(folds),
           "git_commit": sha, "metrics": metrics, "dm_vs_HARX": dm_h, "dm_vs_GBMearn": dm_e,
           "overfit_evidence": overfit}
    outp = Path(out) if out else REPO / "results" / "gbm_refined_champion" / f"gbmR_{market}_h{horizon}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w"), indent=1)
    q = {m: metrics[m]["qlike"] for m in metrics}
    print(f"[gbmR] {market} h{horizon}: HAR-X={q['HAR-X']:.4f} +earn={q['GBM+earn']:.4f} "
          f"+refined={q['GBM+refined']:.4f} (vsHARX p={dm_h['GBM+refined']['qlike']['p_value']:.3f}, "
          f"vs+earn p={dm_e['GBM+refined']['qlike']['p_value']:.3f}) | "
          f"overfit gap {overfit['GBM+refined']['train_to_test_gap_pct']:+.1f}%", flush=True)
    return res


def _earn_design(har5, extras, est, spk, asym, anchors, pos):
    """committed GBM+earn feature matrix: HAR5 + extras + earn_prox + earn_soon (uniform signature for run())."""
    X = G._design(har5, extras, anchors)
    n, N = har5.shape[0], har5.shape[1]
    for k in ("earn_prox", "earn_soon"):
        X = np.column_stack([X, asym[k][pos].reshape(n * N, 1)])
    return X


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
