"""Out-of-sample predictive neighbour test (plan section 33, Gates 6-7).

The crux test for GNN justification: does cross-stock neighbour information improve a
simple leakage-safe OOS forecast beyond an own-stock baseline (Gate 6) AND beyond an
own-stock + market-factor baseline (Gate 7)?

Design (leakage-safe):
- Target y_j(t) = PK_var of stock j at t+h (default h=1), a strict future value.
- Features use data <= t only.
- Neighbour identity, market factor and the linear coefficients are learned on TRAIN
  rows only; metrics are reported on the untouched TEST slice.
- One ridge model per stock (own-scale), aggregated across stocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graphs import top_k_neighbors


def _har_own_features(pk_col: pd.Series) -> pd.DataFrame:
    """Own-stock HAR-style predictors available at t (daily / weekly / monthly PK)."""
    return pd.DataFrame(
        {
            "pk_d": pk_col,
            "pk_w": pk_col.rolling(5).mean(),
            "pk_m": pk_col.rolling(22).mean(),
        }
    )


def _ridge_fit_predict(x_tr, y_tr, x_te, lam=1e-4):
    mu = x_tr.mean(0)
    sd = x_tr.std(0)
    sd[sd == 0] = 1.0
    xs_tr = (x_tr - mu) / sd
    xs_te = (x_te - mu) / sd
    xb_tr = np.column_stack([np.ones(len(xs_tr)), xs_tr])
    xb_te = np.column_stack([np.ones(len(xs_te)), xs_te])
    p = xb_tr.shape[1]
    reg = lam * np.eye(p)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xb_tr.T @ xb_tr + reg, xb_tr.T @ y_tr)
    return xb_te @ beta


def _metrics(y_true, y_pred, per_stock_diracc=None):
    """Aggregate metrics on pooled arrays.

    QLIKE excludes non-positive targets (high==low collapse days would otherwise blow
    it up). Directional accuracy is passed in pre-computed per-stock (``per_stock_diracc``,
    a list of per-stock DirAcc%) and averaged, to avoid diffing across stock boundaries in
    a pooled array.
    """
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    ss = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum(err**2) / ss) if ss > 0 else np.nan
    pos = y_true > 0
    if pos.sum() > 0:
        yt = y_true[pos]
        yp = np.clip(y_pred[pos], 1e-12, None)
        qlike = float(np.mean(yt / yp - np.log(yt / yp) - 1))
    else:
        qlike = np.nan
    if per_stock_diracc is not None:
        da = float(np.nanmean(per_stock_diracc)) if len(per_stock_diracc) else np.nan
    elif len(y_true) > 1:
        da = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)
    else:
        da = np.nan
    return {"rmse": rmse, "r2": r2, "qlike": qlike, "dir_acc": da, "n": int(len(y_true))}


def _diracc(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)


def run_baselines(
    pk_wide: pd.DataFrame,
    market: pd.Series,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_corr: pd.DataFrame,
    horizon: int = 1,
    k: int = 5,
    return_per_stock: bool = False,
):
    """Fit baselines A/B/C per stock on a SHARED valid mask and pool TEST predictions.

    A: own HAR features. B: A + market factor. C: A + Top-K TRAIN-selected neighbour PK.
    Neighbours come from ``train_corr`` (train-only correlation) -> no leakage.

    Comparability (review M1): a single ``valid`` mask (own AND market AND neighbour
    features AND target all present) is applied to every baseline, so A/B/C are evaluated
    on identical (stock, test-row) pairs. DirAcc is computed per stock then averaged
    (review N1). Set ``return_per_stock`` to also get a per-stock RMSE table for a
    sign/Diebold-Mariano-style test (review M2).
    """
    neighbors = top_k_neighbors(train_corr, k, use_abs=True)
    pooled = {"A": ([], [], []), "B": ([], [], []), "C": ([], [], [])}
    per_stock_rows = []
    idx = pk_wide.index
    mkt = market.reindex(idx)
    for j in pk_wide.columns:
        y = pk_wide[j].shift(-horizon)  # future target aligned to t
        own = _har_own_features(pk_wide[j])
        feat = {
            "A": own,
            "B": own.assign(mkt=mkt.values, mkt_w=mkt.rolling(5).mean().values),
            "C": own.assign(
                **{f"nb_{i}": pk_wide[nb].values for i, nb in enumerate(neighbors[j])}
            ),
        }
        # shared validity across ALL baselines' features + target
        valid = y.notna()
        for X in feat.values():
            valid = valid & X.notna().all(axis=1)
        tr = valid & pd.Series(train_mask, index=idx)
        te = valid & pd.Series(test_mask, index=idx)
        if tr.sum() < 50 or te.sum() < 10:
            continue
        y_tr = y[tr].values
        y_te = y[te].values
        pos = y_tr[y_tr > 0]
        floor = float(np.percentile(pos, 1)) if pos.size else 1e-8
        row = {"ticker": j}
        for name, X in feat.items():
            pred = np.clip(_ridge_fit_predict(X[tr].values, y_tr, X[te].values), floor, None)
            pooled[name][0].append(y_te)
            pooled[name][1].append(pred)
            pooled[name][2].append(_diracc(y_te, pred))
            row[f"rmse_{name}"] = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        per_stock_rows.append(row)
    rows = []
    for name, (yt, yp, da) in pooled.items():
        if not yt:
            continue
        m = _metrics(np.concatenate(yt), np.concatenate(yp), per_stock_diracc=da)
        m["baseline"] = {"A": "A_own", "B": "B_own+market", "C": "C_own+neighbors"}[name]
        rows.append(m)
    if not rows:
        agg = pd.DataFrame(columns=["rmse", "r2", "qlike", "dir_acc", "n"]).rename_axis(
            "baseline"
        )
    else:
        agg = pd.DataFrame(rows).set_index("baseline")
    if return_per_stock:
        return agg, pd.DataFrame(per_stock_rows)
    return agg
