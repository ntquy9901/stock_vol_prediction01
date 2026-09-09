"""gamma-GBM + forward-looking earnings + calm-floor calibration, on the canonical S&P 500 walk-forward.

Builds on the committed gamma-GBM (baselines/2026-09-08_gamma_gbm): same panel/folds/HAR-X/features, plus
  (a) a CLEAN forward-looking earnings feature: earn_prox = max(0, 1 - days_to_nearest_earnings/RAMP) (a ramp
      that is 0 far from any scheduled earnings so it adds no noise on the ~76% far cells) + earn_soon binary.
      Earnings dates are scheduled weeks ahead, so the distance from the TARGET day (t+h) to the nearest
      earnings is known at forecast time — a legitimate forward-looking, per-stock signal.
  (b) a QLIKE-optimal calm-floor CALIBRATION fit on the validation split: per forecast-quantile bin, the
      QLIKE-optimal multiplier is c* = mean(y/f) (argmin_c QLIKE(y, c*f)); applied to test to correct the
      systematic calm-decile over-forecast (~4x at D0) without peeking at test.

Reports HAR-X, GBM(base), GBM+earn, GBM+earn+calib; date-clustered DM vs HAR-X and vs GBM(base)."""
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
for _p in ("baselines/2026-09-08_gamma_gbm/code", "baselines/2026-09-07_harq/code",
           "baselines/2026-08-31_walkforward_volga/code", "baselines/2026-08-30_walkforward_harx_lstm/code",
           "baselines/2026-08-21_har_anchored_residual/code", "submission/soict_lstm_gat"):
    sys.path.insert(0, str(REPO / _p))
import gamma_gbm_walkforward as G  # noqa: E402
import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402

FL = pc.QLIKE_FLOOR
RAMP, SOON, CAP = 5.0, 3.0, 60.0                            # earnings proximity ramp / "soon" window / cap (days)
CAL_BINS = 10
EARN_KEYS = ["earn_prox", "earn_soon"]
EARN_PARQUET = REPO / "results" / "gamma_gbm" / "sp500_earnings.parquet"


def _earn_by_ticker():
    e = pd.read_parquet(EARN_PARQUET)
    return {tk: np.sort(g["earnings_date"].values.astype("datetime64[D]")) for tk, g in e.groupby("ticker")}


def _dist(tgt, ed):
    if ed is None or len(ed) == 0:
        return np.full(len(tgt), CAP)
    pos = np.searchsorted(ed, tgt)
    nxt = np.where(pos < len(ed), (ed[np.clip(pos, 0, len(ed) - 1)] - tgt) / np.timedelta64(1, "D"), CAP)
    prv = np.where(pos > 0, (tgt - ed[np.clip(pos - 1, 0, len(ed) - 1)]) / np.timedelta64(1, "D"), CAP)
    return np.minimum(np.minimum(np.abs(nxt), np.abs(prv)), CAP)


def earnings_panels(panel, earn):
    """{earn_prox, earn_soon} [n_anchor, N] from the distance of each anchor's TARGET date to the nearest
    scheduled earnings of that ticker (clean ramp -> 0 far away)."""
    tgt = pd.to_datetime(panel.target_dates).values.astype("datetime64[D]")
    N = panel.N; prox = np.zeros((len(tgt), N), np.float32); soon = np.zeros((len(tgt), N), np.float32)
    for j, tk in enumerate(panel.tickers):
        dist = _dist(tgt, earn.get(tk))
        prox[:, j] = np.maximum(0.0, 1.0 - dist / RAMP); soon[:, j] = (dist <= SOON).astype(np.float32)
    return {"earn_prox": prox, "earn_soon": soon}


def _design(har5, extras, earn, anchors, pos, use_earn):
    # ``anchors`` = date-indices into the [T,N] feature axis (for HAR/extras via G._design);
    # ``pos`` = positional indices into the [A,N] earnings panels (fold.train/val/forecast). These two index
    # spaces DIFFER (anchors start at ~30); earnings MUST use ``pos`` to line up row-for-row with har5.
    X = G._design(har5, extras, anchors)
    if use_earn:
        n, N = har5.shape[0], har5.shape[1]
        for k in EARN_KEYS:
            X = np.column_stack([X, earn[k][pos].reshape(n * N, 1)])
    return X


def _gbm_fit_predict(xtr, ytr, *xs):
    m = HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=0).fit(xtr, ytr)
    return [m.predict(x) for x in xs]


def _calibrate(pv, yv, pt, nfloor_te):
    """QLIKE-optimal per-forecast-bin multiplier c*=mean(y/f) fit on val, applied to test (clipped)."""
    edges = np.quantile(pv, np.linspace(0, 1, CAL_BINS + 1)); internal = edges[1:-1]
    vbin = np.digitize(pv, internal); tbin = np.digitize(pt, internal); c = np.ones(CAL_BINS)
    for b in range(CAL_BINS):
        mb = vbin == b
        if mb.sum() >= 50:
            c[b] = float(np.clip(np.mean(yv[mb] / np.maximum(pv[mb], FL)), 0.3, 3.0))
    return np.maximum(pt * c[tbin], nfloor_te)


def run(market="sp500_clean", horizon=5, folds_target=7, out=None):  # pragma: no cover - SP500-only 20-min walk-forward driver; logic is in the unit-tested helpers, validated by the committed results JSON
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, pc.LOOKBACK, horizon)
    panel = build_enriched_panel(files, pc.LOOKBACK, horizon, keep)
    extras = G.extra_feature_panels(panel.feats); earn = earnings_panels(panel, _earn_by_ticker())
    wf = VolgaWFConfig(lookback=pc.LOOKBACK, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    fl = training_config(epochs=1, seeds=(0,), batch=32).qlike_floor
    pooled = {m: {} for m in ("HAR-X", "GBM", "GBM+earn", "GBM+earn+cal")}
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        nf_te = np.broadcast_to(nfloor, D.y_te.shape).reshape(-1)
        pooled["HAR-X"].update(RMR._pred_dict(G._harx_ols(D, nfloor), D.y_te, D.tmask_te, D.d_te, D.N))
        mtr = D.tmask_tr.astype(bool).reshape(-1); ytr = np.maximum(D.y_tr.reshape(-1)[mtr], fl)
        for tag, ue in (("GBM", False), ("GBM+earn", True)):
            xtr = _design(D.har5_tr, extras, earn, panel.anchors[fold.train], fold.train, ue)[mtr]
            xva = _design(D.har5_va, extras, earn, panel.anchors[fold.val], fold.val, ue)
            xte = _design(D.har5_te, extras, earn, panel.anchors[fold.forecast], fold.forecast, ue)
            pva, pte = _gbm_fit_predict(xtr, ytr, xva, xte)
            pte_f = np.maximum(pte, nf_te)
            pooled[tag].update(RMR._pred_dict(pte_f.reshape(D.y_te.shape), D.y_te, D.tmask_te, D.d_te, D.N))
            if ue:                                          # calm-floor calibration on the +earn model
                vm = D.tmask_va.astype(bool).reshape(-1)
                pcal = _calibrate(np.maximum(pva, fl)[vm], D.y_va.reshape(-1)[vm], pte_f, nf_te)
                pooled["GBM+earn+cal"].update(RMR._pred_dict(pcal.reshape(D.y_te.shape), D.y_te, D.tmask_te, D.d_te, D.N))
    metrics = {m: RMR._metrics(pooled[m], fl) for m in pooled}
    dm_h = {m: RMR._dm_all(pooled[m], pooled["HAR-X"], horizon, fl) for m in ("GBM", "GBM+earn", "GBM+earn+cal")}
    dm_g = {m: RMR._dm_all(pooled[m], pooled["GBM"], horizon, fl) for m in ("GBM+earn", "GBM+earn+cal")}
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=str(REPO)).stdout.strip() or None
    except Exception:                                                    # pragma: no cover
        sha = None
    res = {"experiment": "gamma_gbm_earnings", "market": market, "horizon": horizon, "n_folds": len(folds),
           "git_commit": sha, "metrics": metrics, "dm_vs_HARX": dm_h, "dm_vs_GBM": dm_g}
    outp = Path(out) if out else REPO / "results" / "gamma_gbm_earnings" / f"gbmE_{market}_h{horizon}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w"), indent=1)
    q = {m: metrics[m]["qlike"] for m in metrics}
    print(f"[gbmE] {market} h{horizon}: HAR-X={q['HAR-X']:.4f} GBM={q['GBM']:.4f} "
          f"+earn={q['GBM+earn']:.4f} (p_vsGBM={dm_g['GBM+earn']['qlike']['p_value']:.3f}) "
          f"+cal={q['GBM+earn+cal']:.4f} (p_vsGBM={dm_g['GBM+earn+cal']['qlike']['p_value']:.3f})", flush=True)
    return res


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
