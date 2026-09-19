"""Exp 3 --- Distributional calibration of the VolTree gamma forecast (PIT / ECE + pinball).

Completes the trustworthy-AI axis: Exp 2 gave prediction INTERVALS (conformal); this evaluates the FULL
predictive DISTRIBUTION the reg:gamma booster already implies, at near-zero extra cost (no new model).

reg:gamma predicts the gamma MEAN mu per row. We estimate a global shape k from the training ratio
y/mu (for a gamma, y/mu ~ Gamma(k, 1/k), mean 1, var 1/k => k = 1/var(y/mu)); the per-row predictive law
is then Gamma(shape=k, scale=mu/k). Calibration is assessed by the Probability Integral Transform
PIT = F(y) which must be Uniform[0,1] when calibrated; we report the PIT-based Expected Calibration Error
(ECE), the central-90% gamma-interval coverage, and the pinball loss at 0.05/0.5/0.95.

Leak-safe: k is estimated on the training slice only; PIT/scores are on the held-out test window. h1,
train capped for overnight speed. Output: results/gamma_gbm/calibration_<market>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
sys.path.insert(0, str(REPO / "baselines" / "2026-09-18_leaf_graph_paper" / "code"))
sys.path.insert(0, str(_CODE))
import run_leaf_graph_paper as RLG  # noqa: E402
import run_conformal as CF  # noqa: E402

FM, S1, LG, C = RLG.FM, RLG.S1, RLG.LG, RLG.C
FL = FM.FL
DEMO_HORIZONS = (1,)
TRAIN_CAP = CF.TRAIN_CAP
Q_GRID = (0.05, 0.5, 0.95)


def gamma_shape(y, mu):
    """Global gamma shape k from the training ratio r=y/mu (mean 1): k = 1/var(r), clipped positive. Pure."""
    r = np.asarray(y, float) / np.clip(np.asarray(mu, float), FL, None)
    v = float(np.var(r))
    return float(np.clip(1.0 / v, 1e-3, 1e6)) if v > 0 else 1e6


def pit_values(y, mu, k):
    """PIT = Gamma(shape=k, scale=mu/k).cdf(y). Uniform[0,1] iff calibrated. Pure."""
    scale = np.clip(np.asarray(mu, float), FL, None) / k
    return stats.gamma.cdf(np.asarray(y, float), a=k, scale=scale)


def pit_ece(pit, bins=10):
    """Expected Calibration Error from PIT uniformity: mean |empirical CDF - nominal| at bin edges. Pure."""
    p = np.sort(np.clip(np.asarray(pit, float), 0, 1))
    edges = np.linspace(0, 1, bins + 1)[1:]
    emp = np.searchsorted(p, edges, side="right") / len(p)
    return float(np.mean(np.abs(emp - edges)))


def pinball(y, q_pred, alpha):
    """Pinball (quantile) loss at level alpha for predicted quantile q_pred. Pure."""
    y = np.asarray(y, float); q = np.asarray(q_pred, float)
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def run_market(market, horizons=None):  # pragma: no cover - data-driven walk-forward driver
    horizons = horizons or DEMO_HORIZONS
    frames, _sect, edates = FM.load(market)
    edates = RLG._load_earn(market, edates)
    cols, _ = RLG.resolve_cols("full", bool(edates))
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out = {"market": market, "horizons": {}}
    for h in horizons:
        a = FM.panel(frames, edates, h)
        embargo = __import__("pandas").Timedelta(days=int(h * C.EMBARGO_MULT) + C.EMBARGO_BUFFER_DAYS)
        yy, pit_all, pin = [], [], {q: [] for q in Q_GRID}
        cov90 = []
        for k in range(len(S1.FOLDS) - 1):
            import pandas as pd
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            trf = CF._cap_rows(trf, TRAIN_CAP)
            bst = LG.fit_booster(trf, cols, seed=0)
            mu_tr = CF._pred(bst, trf[cols].to_numpy(float))
            shp = gamma_shape(trf["y"].to_numpy(float), mu_tr)
            mu_te = CF._pred(bst, tef[cols].to_numpy(float))
            yte = tef["y"].to_numpy(float)
            pit = pit_values(yte, mu_te, shp)
            yy.append(yte); pit_all.append(pit)
            scale = np.clip(mu_te, FL, None) / shp
            for q in Q_GRID:
                pin[q].append(pinball(yte, stats.gamma.ppf(q, a=shp, scale=scale), q))
            lo = stats.gamma.ppf(0.05, a=shp, scale=scale); hi = stats.gamma.ppf(0.95, a=shp, scale=scale)
            cov90.append(float(((yte >= lo) & (yte <= hi)).mean()) * len(yte))
        if not yy:
            continue
        pit = np.concatenate(pit_all); n = len(pit)
        out["horizons"][str(h)] = {
            "n_test": int(n), "gamma_ece": pit_ece(pit),
            "pit_mean": float(pit.mean()), "pit_below_median_frac": float((pit < 0.5).mean()),
            "central90_coverage": float(np.sum(cov90) / n),
            "pinball": {str(q): float(np.mean(v)) for q, v in pin.items()},
        }
    return out


def main():  # pragma: no cover - entry driver: both markets, writes JSON
    outdir = REPO / "results" / "gamma_gbm"; outdir.mkdir(parents=True, exist_ok=True)
    for market in ("sp500", "hose"):
        res = run_market(market)
        (outdir / f"calibration_{market}.json").write_text(json.dumps(res, indent=1))
        for h, r in res["horizons"].items():
            print(f"[calib] {market} h{h}: ECE {r['gamma_ece']:.3f} PIT-mean {r['pit_mean']:.3f} "
                  f"central90-cov {r['central90_coverage']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
