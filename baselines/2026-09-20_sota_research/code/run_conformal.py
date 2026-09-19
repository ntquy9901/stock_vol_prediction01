"""Exp 2 --- Conformal prediction intervals on VolTree (calibrated, distribution-free uncertainty).

A NEW axis for the paper: point-QLIKE is saturated, so we add finite-sample-valid prediction INTERVALS
for the daily Parkinson variance. Two conformal variants, both leak-safe under the VolTree walk-forward:

  * Split conformal (baseline): score = |y - mu| on a calibration slice; symmetric band mu +/- Q.
  * CQR (Conformalized Quantile Regression, Romano et al. NeurIPS 2019): two XGBoost quantile boosters
    (reg:quantileerror @ 0.05/0.95) give an adaptive band, conformalized by the calibration residual Q.
    Adaptive width (wide in storms, tight in calm) suits heteroscedastic volatility.

Leak-safety: the point + quantile boosters are fit on TRAIN minus the last VALID_LEN dates; the
calibration quantile Q is computed on that held-out VALIDATION slice (strictly train dates, never test);
the frozen interval is then applied to the test window. Test never informs Q --- mirrors the VolTree
alpha-on-validation protocol. Intervals are floored at FL (variance is non-negative).

Reports per (market, horizon), for split + CQR: marginal coverage (target 1-ALPHA), mean/median interval
width, and per-regime coverage inside vs outside the pre-specified spike windows (COVID/2022/Apr-2025) ---
the HOSE robustness mandate applied to coverage. Output: results/gamma_gbm/conformal_<market>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
sys.path.insert(0, str(REPO / "baselines" / "2026-09-18_leaf_graph_paper" / "code"))
import run_leaf_graph_paper as RLG  # noqa: E402
import xgboost as xgb  # noqa: E402

FM, S1, LG, C = RLG.FM, RLG.S1, RLG.LG, RLG.C
FL = FM.FL
ALPHA = 0.10                    # target 90% intervals
Q_LO, Q_HI = 0.05, 0.95        # pinball quantiles for CQR
QUANTILE_ROUNDS = 200          # boosting rounds for the two quantile heads (interval endpoints; < the 300 point)
DEMO_HORIZONS = (1,)           # representative horizon for the overnight conformal demo (full = C.HORIZONS)
TRAIN_CAP = 250_000            # overnight speed: cap per-fold train rows for the base+quantile fits. Conformal
#                               coverage is model-AGNOSTIC (valid for any base predictor), so a train
#                               subsample only speeds fitting; the paper-grade run would drop this cap.


def _fit_quantile(trf, cols, q, seed=0):
    """XGBoost quantile booster (reg:quantileerror @ q) at VolTree capacity."""
    d = xgb.DMatrix(trf[cols].to_numpy(float), label=trf["y"].to_numpy(float))
    params = {"objective": "reg:quantileerror", "quantile_alpha": q, "eta": C.XGB_LR,
              "max_leaves": C.XGB_MAX_LEAVES, "max_depth": C.XGB_MAX_DEPTH, "grow_policy": "lossguide",
              "lambda": C.XGB_L2, "min_child_weight": C.XGB_MIN_CHILD_WEIGHT,
              "tree_method": "hist", "seed": int(seed), "verbosity": 0}
    return xgb.train(params, d, num_boost_round=QUANTILE_ROUNDS)


def _pred(bst, X):
    return bst.predict(xgb.DMatrix(np.asarray(X, float)))


def _cap_rows(df, cap, seed=0):
    """Deterministic row subsample to at most ``cap`` rows (overnight speed cap for the base+quantile
    fits). Returns df unchanged when already <= cap. Conformal coverage is model-agnostic, so this only
    affects fitting speed, not interval validity."""
    return df.sample(n=cap, random_state=seed) if len(df) > cap else df


def cqr_band(y_cal, lo_cal, hi_cal, lo_te, hi_te, alpha):
    """Conformalized quantile band: Q = (1-alpha) empirical quantile of the calibration conformity
    E_i = max(lo_i - y_i, y_i - hi_i); test band = [lo - Q, hi + Q] floored at FL. Pure + testable."""
    e = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n = len(e)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)           # finite-sample conformal level
    Q = float(np.quantile(e, q_level, method="higher"))
    lo = np.clip(lo_te - Q, FL, None); hi = np.maximum(hi_te + Q, lo)
    return lo, hi, Q


def split_band(y_cal, mu_cal, mu_te, alpha):
    """Split-conformal symmetric band: Q = (1-alpha) quantile of |y-mu| on calibration. Pure + testable."""
    e = np.abs(y_cal - mu_cal)
    n = len(e)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Q = float(np.quantile(e, q_level, method="higher"))
    lo = np.clip(mu_te - Q, FL, None); hi = mu_te + Q
    return lo, hi, Q


def _cov_width(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean()), float(np.mean(hi - lo)), float(np.median(hi - lo))


def _spike_mask(dates):
    """Boolean: test dates inside any pre-specified spike window (COVID/2022/Apr-2025)."""
    d = pd.to_datetime(dates); m = np.zeros(len(d), dtype=bool)
    for a, b in C.SPIKE_WINDOWS:
        m |= (d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))
    return m


def run_market(market, horizons=None):  # pragma: no cover - data-driven walk-forward driver
    horizons = horizons or C.HORIZONS
    frames, _sect, edates = FM.load(market)
    edates = RLG._load_earn(market, edates)
    cols, _ = RLG.resolve_cols("full", bool(edates))
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out = {"market": market, "alpha": ALPHA, "target_coverage": 1 - ALPHA, "horizons": {}}
    for h in horizons:
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * C.EMBARGO_MULT) + C.EMBARGO_BUFFER_DAYS)
        yy, sp_lo, sp_hi, cq_lo, cq_hi, dts = [], [], [], [], [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            is_val = trf["date"].isin(val_dates)
            trf_e, vaf = trf[~is_val], trf[is_val]
            trf_e = _cap_rows(trf_e, TRAIN_CAP)           # overnight speed cap (helper; conformal is model-agnostic)
            Xva, Xte = vaf[cols].to_numpy(float), tef[cols].to_numpy(float)
            yva, yte = vaf["y"].to_numpy(float), tef["y"].to_numpy(float)
            gbm = LG.fit_booster(trf_e, cols, seed=0)
            blo = _fit_quantile(trf_e, cols, Q_LO); bhi = _fit_quantile(trf_e, cols, Q_HI)
            # split conformal
            slo, shi, _ = split_band(yva, _pred(gbm, Xva), _pred(gbm, Xte), ALPHA)
            # CQR
            clo, chi, _ = cqr_band(yva, _pred(blo, Xva), _pred(bhi, Xva), _pred(blo, Xte), _pred(bhi, Xte), ALPHA)
            yy.append(yte); sp_lo.append(slo); sp_hi.append(shi); cq_lo.append(clo); cq_hi.append(chi)
            dts.append(tef["date"].to_numpy())
        if not yy:
            continue
        y = np.concatenate(yy); dd = np.concatenate(dts)
        sl, sh = np.concatenate(sp_lo), np.concatenate(sp_hi)
        cl, ch = np.concatenate(cq_lo), np.concatenate(cq_hi)
        spike = _spike_mask(dd); calm = ~spike
        rec = {"n_test": int(len(y)), "n_spike": int(spike.sum())}
        for name, lo, hi in (("split", sl, sh), ("cqr", cl, ch)):
            cov, mw, mdw = _cov_width(y, lo, hi)
            cov_s = _cov_width(y[spike], lo[spike], hi[spike])[0] if spike.any() else None
            cov_c = _cov_width(y[calm], lo[calm], hi[calm])[0] if calm.any() else None
            rec[name] = {"coverage": cov, "mean_width": mw, "median_width": mdw,
                         "coverage_spike": cov_s, "coverage_calm": cov_c}
        out["horizons"][str(h)] = rec
    return out


def main():  # pragma: no cover - entry driver: both markets, writes JSON
    outdir = REPO / "results" / "gamma_gbm"; outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for market in ("sp500", "hose"):
        res = run_market(market, horizons=DEMO_HORIZONS)
        (outdir / f"conformal_{market}.json").write_text(json.dumps(res, indent=1))
        summary[market] = res
        for h, r in res["horizons"].items():
            print(f"[conformal] {market} h{h}: split cov {r['split']['coverage']:.3f} w {r['split']['mean_width']:.2e} | "
                  f"cqr cov {r['cqr']['coverage']:.3f} w {r['cqr']['mean_width']:.2e}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
