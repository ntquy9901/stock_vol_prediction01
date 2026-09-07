"""Causal regime-conditional convex blend of a deep volatility forecast and HAR-X.

Blend: ``f = w * deep + (1 - w) * harx`` with ``w = sigmoid(theta0 + theta1*z_level + theta2*z_disagree)``.
Causal gate features (known at forecast origin t-h): ``z_level`` = standardized ``log(harx)`` (recent-vol
regime proxy) and ``z_disagree`` = standardized ``log(deep) - log(harx)`` (the echo signal). ``theta`` is fit
per walk-forward fold on that fold's VALIDATION QLIKE and applied to its TEST split with the fold-val
standardization -- no cross-fold pooling (no look-ahead), no target-day information in the gate.

Reuses the tested 5-metric + Diebold-Mariano implementation (``submission/soict_lstm_gat/metrics.py``) and the
date-clustered DM (``baselines/2026-08-21_har_anchored_residual/code/stats.py``). QLIKE floor is the shared
``pipeline_config.QLIKE_FLOOR`` for every compared model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"):
    sys.path.insert(0, str(_p))
import metrics as M  # noqa: E402
import pipeline_config as pc  # noqa: E402
import stats as ST  # noqa: E402

FLOOR = pc.QLIKE_FLOOR                       # shared QLIKE positivity floor (identical across models)
CELLS = REPO / "results" / "qlike_anchor" / "cells"


def load_folds(market: str, h: int, deep: str = "VolGA") -> pd.DataFrame | None:
    """One row per (split, fold, ticker, date) with realized ``y`` + ``harx`` + ``deep`` forecasts."""
    p = CELLS / f"cells_{market}_qlike_none_h{h}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p, columns=["model", "split", "fold", "ticker", "date", "y_true", "y_pred"])
    out = []
    for sp in ("val", "test"):
        t = d[d["split"] == sp]
        # cells are unique per (fold, ticker, date, model) — walk-forward test windows are disjoint
        # (verified: no (ticker,date) duplicate within a split/model), so mean == identity here.
        w = t.pivot_table(index=["fold", "ticker", "date"], columns="model", values="y_pred", aggfunc="mean")
        w["y"] = t.groupby(["fold", "ticker", "date"])["y_true"].first()
        w = w.dropna(subset=["HAR-X", deep, "y"]).reset_index()
        w["split"] = sp
        out.append(w[["split", "fold", "ticker", "date", "y", "HAR-X", deep]]
                   .rename(columns={"HAR-X": "harx", deep: "deep"}))
    return pd.concat(out, ignore_index=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))     # clip = overflow guard, not a tunable


def features(df: pd.DataFrame) -> np.ndarray:
    """Causal gate features: [log(harx), log(deep) - log(harx)] (both known at origin t-h)."""
    lh = np.log(np.maximum(df["harx"].to_numpy(dtype=float), FLOOR))
    ld = np.log(np.maximum(df["deep"].to_numpy(dtype=float), FLOOR))
    return np.column_stack([lh, ld - lh])


def blend_forecast(deep: np.ndarray, harx: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w * deep + (1.0 - w) * harx


def fit_fold(val: pd.DataFrame, init=(2.0, 0.0, 0.0)) -> dict:
    """Fit theta minimizing val QLIKE of the blend; returns theta + fold-val standardization (mu, sd)."""
    X = features(val); mu = X.mean(axis=0); sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)                        # avoid divide-by-zero on a constant feature
    Z = (X - mu) / sd
    y = val["y"].to_numpy(dtype=float); deep = val["deep"].to_numpy(dtype=float)
    harx = val["harx"].to_numpy(dtype=float)

    def obj(th):
        w = sigmoid(th[0] + Z @ th[1:])
        return M.qlike(y, blend_forecast(deep, harx, w), floor=FLOOR)

    res = minimize(obj, np.asarray(init, dtype=float), method="Nelder-Mead",
                   options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-9})
    theta = res.x if np.isfinite(res.fun) else np.asarray(init, dtype=float)
    return {"theta": theta.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}


def apply_fold(df: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Blended forecast + weight for ``df`` using a fitted fold's params (val standardization)."""
    Z = (features(df) - np.asarray(params["mu"])) / np.asarray(params["sd"])
    th = np.asarray(params["theta"])
    w = sigmoid(th[0] + Z @ th[1:])
    return blend_forecast(df["deep"].to_numpy(dtype=float), df["harx"].to_numpy(dtype=float), w), w


def blend_walkforward(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-fold fit-on-val / apply-on-(val,test). Returns test+val rows with ``blend`` and ``w`` columns."""
    parts = []
    for k, g in panel.groupby("fold"):
        val = g[g["split"] == "val"]
        if len(val) < 10:                                    # too few val rows to fit -> keep deep (w=1)
            g = g.assign(blend=g["deep"].to_numpy(), w=1.0); parts.append(g); continue
        params = fit_fold(val)
        for sp in ("val", "test"):
            s = g[g["split"] == sp].copy()
            if s.empty:
                continue
            bl, w = apply_fold(s, params)
            s["blend"] = bl; s["w"] = w; s["theta"] = str(params["theta"]); parts.append(s)
    return pd.concat(parts, ignore_index=True)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "qlike": M.qlike(y, p, floor=FLOOR), "r2": M.r2(y, p)}


def _dm_qlike(a: np.ndarray, b: np.ndarray, y: np.ndarray, dates: np.ndarray, h: int) -> dict:
    la = M.per_obs_qlike(y, a, floor=FLOOR); lb = M.per_obs_qlike(y, b, floor=FLOOR)
    r = ST.date_clustered_dm(la, lb, dates, h)
    return {"p": round(float(r["p_value"]), 4), "mean_diff": float(r["mean_diff"]),
            "favors_A": bool(r["mean_diff"] < 0)}


def evaluate(market: str, h: int, deep: str = "VolGA") -> dict | None:
    """Full evaluation for one (market, horizon, deep base): metrics + DM + overfit + weight evidence."""
    panel = load_folds(market, h, deep)
    if panel is None:
        return None
    scored = blend_walkforward(panel)
    te = scored[scored["split"] == "test"]; va = scored[scored["split"] == "val"]
    y = te["y"].to_numpy(dtype=float); dates = te["date"].to_numpy()
    dv = te["deep"].to_numpy(dtype=float); hx = te["harx"].to_numpy(dtype=float)
    bl = te["blend"].to_numpy(dtype=float)
    return {
        "market": market, "horizon": h, "deep": deep, "n_test": int(len(te)),
        "test_metrics": {"deep": _metrics(y, dv), "harx": _metrics(y, hx), "blend": _metrics(y, bl)},
        # in-sample fit quality (theta was fit on val) — NOT held-out; the beat-HAR-X claim uses test only.
        "val_metrics": {"blend_qlike_insample": M.qlike(va["y"].to_numpy(dtype=float),
                                                        va["blend"].to_numpy(dtype=float), floor=FLOOR)},
        "dm_blend_vs_harx": _dm_qlike(bl, hx, y, dates, h),
        "dm_blend_vs_deep": _dm_qlike(bl, dv, y, dates, h),
        "dm_deep_vs_harx": _dm_qlike(dv, hx, y, dates, h),
        "mean_w": round(float(te["w"].mean()), 3),
        "mean_w_by_fold": {int(k): round(float(g["w"].mean()), 3) for k, g in te.groupby("fold")},
    }
