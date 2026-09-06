"""Pure evaluation helpers over pooled ``{(node, date): (y_true, y_pred)}`` prediction dicts.

Kept separate from the driver so every diagnostic (limit-lock split, top-1%-date exclusion, ticker /
unique-date win rate, count summary, shock-month trace, fit metrics) is unit-testable without training
an XGBoost model. All QLIKE uses the project's shared positivity floor via ``metrics.per_obs_qlike``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import metrics as M  # noqa: E402


def subset(pred, keys, inside):
    """Sub-dict of ``pred`` restricted to (inside=True) / excluding (inside=False) ``keys``."""
    return {k: v for k, v in pred.items() if (k in keys) == inside}


def _yp(pred):
    ks = sorted(pred)
    y = np.array([pred[k][0] for k in ks], dtype=float)
    p = np.array([pred[k][1] for k in ks], dtype=float)
    return ks, y, p


def qlike_share(pred, keys, floor):
    """Fraction of total QLIKE loss contributed by the cells in ``keys`` (e.g. limit-lock cells)."""
    ks, y, p = _yp(pred)
    if not ks:
        return None
    per = M.per_obs_qlike(y, p, floor)
    total = float(per.sum())
    if total == 0.0:
        return 0.0
    inside = np.array([k in keys for k in ks])
    return float(per[inside].sum() / total)


def exclude_top_pct_dates(pred, floor, pct):
    """QLIKE after removing the ``pct`` fraction of unique dates with the largest total QLIKE
    contribution. Returns ``(qlike_excluded, n_dates_excluded, n_dates_total)``."""
    ks, y, p = _yp(pred)
    if not ks:
        return None, 0, 0
    per = M.per_obs_qlike(y, p, floor)
    dates = np.array([k[1] for k in ks])
    uniq = np.unique(dates)
    contrib = {d: float(per[dates == d].sum()) for d in uniq}
    n_excl = max(1, int(np.ceil(len(uniq) * pct)))
    worst = set(sorted(uniq, key=lambda d: contrib[d], reverse=True)[:n_excl])
    keep = ~np.isin(dates, list(worst))
    if keep.sum() == 0:
        return None, n_excl, len(uniq)
    return float(per[keep].mean()), n_excl, len(uniq)


def win_rate_vs(pred_a, pred_b, floor):
    """``(ticker_win_rate, unique_date_win_rate)`` of model A vs model B on shared cells (A wins a
    ticker / date when its mean per-obs QLIKE over that ticker / date is strictly lower)."""
    ks = sorted(set(pred_a) & set(pred_b))
    if not ks:
        return None, None
    y = np.array([pred_a[k][0] for k in ks], dtype=float)
    pa = np.array([pred_a[k][1] for k in ks], dtype=float)
    pb = np.array([pred_b[k][1] for k in ks], dtype=float)
    la = M.per_obs_qlike(y, pa, floor)
    lb = M.per_obs_qlike(y, pb, floor)
    nodes = np.array([k[0] for k in ks])
    dates = np.array([k[1] for k in ks])

    def _rate(group):
        wins = tot = 0
        for g in np.unique(group):
            m = group == g
            tot += 1
            wins += int(la[m].mean() < lb[m].mean())
        return wins / tot if tot else None

    return _rate(nodes), _rate(dates)


def count_summary(pred):
    """``(n_ticker_date, n_unique_dates, n_tickers)`` -- never conflate ticker-date obs with dates."""
    ks = list(pred)
    return len(ks), len({k[1] for k in ks}), len({k[0] for k in ks})


def fit_metrics(y_rows, p_rows, floor):
    """Split fit metrics (mse / qlike / r2 / n) on flat masked (y, pred) rows -- overfit evidence."""
    y = np.asarray(y_rows, dtype=float)
    p = np.asarray(p_rows, dtype=float)
    return {"mse": M.mse(y, p), "qlike": M.qlike(y, p, floor), "r2": M.r2(y, p), "n": int(len(y))}


def shock_month_diagnostics(pred_model, pred_harx, lock_keys, floor, month):
    """April-2025-style shock-cluster trace for target dates whose ``YYYY-MM`` prefix == ``month``.

    Reports affected unique dates, locked-cell fraction, mean target/forecast, and per-model mean QLIKE
    with vs without the lock cells -- so under/over-prediction and lock dominance are visible.
    """
    ks = [k for k in sorted(set(pred_model) & set(pred_harx)) if str(k[1]).startswith(month)]
    if not ks:
        return {"month": month, "n_ticker_date": 0, "n_dates": 0}
    y = np.array([pred_model[k][0] for k in ks], dtype=float)
    pm = np.array([pred_model[k][1] for k in ks], dtype=float)
    ph = np.array([pred_harx[k][1] for k in ks], dtype=float)
    is_lock = np.array([k in lock_keys for k in ks])
    qm = M.per_obs_qlike(y, pm, floor)
    qh = M.per_obs_qlike(y, ph, floor)
    nonlock = ~is_lock
    return {
        "month": month, "n_ticker_date": len(ks), "n_dates": len({k[1] for k in ks}),
        "n_lock_cells": int(is_lock.sum()), "lock_fraction": float(is_lock.mean()),
        "mean_target": float(y.mean()), "mean_forecast_model": float(pm.mean()),
        "mean_forecast_harx": float(ph.mean()),
        "qlike_model_all": float(qm.mean()), "qlike_harx_all": float(qh.mean()),
        "qlike_model_nonlock": float(qm[nonlock].mean()) if nonlock.any() else None,
        "qlike_harx_nonlock": float(qh[nonlock].mean()) if nonlock.any() else None,
        "underprediction_rate_model": float(np.mean(pm < y)),
    }
