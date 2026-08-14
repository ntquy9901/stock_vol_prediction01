"""Nonlinear edge discovery: gradient-boosting ΔQLIKE + market-residual mutual information.

The linear (ridge) supervised ΔQLIKE test returned STOP. This checks the remaining angle: a
NONLINEAR / tail relationship where only extreme source volatility drives a target's future
volatility, which a linear model underrates. Two probes, both leakage-safe:

- ``predictive_dqlike_edges_gb``: the SAME val-select / test-confirm / floor / median-gate harness as
  the linear test (``edge_discovery.predictive_dqlike_edges``), but the per-pair predictor is a
  regularized gradient-boosting regressor, so a genuine tail/interaction edge is recovered.
- ``mi_residual``: mutual information between source i(t) and the market-and-E2-residual of target
  j(t+h), versus a source-permutation null — a model-free descriptive check for leftover
  (incl. nonlinear) dependence beyond the E2 base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression

from graph_eda import edge_discovery as ed
from graph_eda.predictive import _ridge_fit_predict


def _gb_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """Regularized gradient-boosting fit/predict (shallow trees, L2, early stopping)."""
    model = HistGradientBoostingRegressor(
        max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0)
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


def predictive_dqlike_edges_gb(
    pk_var: pd.DataFrame, market: pd.Series, volz: pd.DataFrame,
    train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray,
    horizon: int = 5, with_regime: bool = False,
) -> pd.DataFrame:
    """Directed ΔQLIKE edges under a gradient-boosting predictor (nonlinear/tail edges)."""
    return ed.predictive_dqlike_edges(
        pk_var, market, volz, train_mask, val_mask, test_mask,
        horizon=horizon, with_regime=with_regime, predict_fn=_gb_predict)


def mi_residual(
    pk_var: pd.DataFrame, market: pd.Series, volz: pd.DataFrame, train_mask: np.ndarray,
    horizon: int = 5, n_shuffle: int = 10, seed: int = 0,
) -> pd.DataFrame:
    """MI between source i(t) and the E2-residual of target j(t+h), vs a source-permutation null.

    Target residual = PK_j(t+h) minus its E2 (HAR + MarketPK + volz) ridge prediction (TRAIN betas
    applied full-sample), so any remaining MI is dependence beyond the base — including nonlinear.
    Returns per-pair MI plus a global permutation-null mean/p95 (source order shuffled).
    """
    idx = pk_var.index
    tickers = list(pk_var.columns)
    tr = pd.Series(train_mask, index=idx)
    mkt = market.reindex(idx)
    resid: dict[str, pd.Series] = {}
    for j in tickers:
        y = pk_var[j].shift(-horizon)
        base = pd.concat([ed._har(pk_var[j]), mkt.rename("mkt"), volz[j].rename("volz")], axis=1)
        valid = y.notna() & base.notna().all(axis=1)
        trm = valid & tr
        e = pd.Series(np.nan, index=idx)
        if trm.sum() >= 100:
            pred = _ridge_fit_predict(base[trm].values, y[trm].values, base[valid].values)
            e[valid] = y[valid].values - pred
        resid[j] = e

    rng = np.random.default_rng(seed)
    rows, nulls = [], []
    for j in tickers:
        e = resid[j]
        for i in tickers:
            if i == j:
                continue
            m = e.notna() & pk_var[i].notna()
            if m.sum() < 100:
                continue
            x = pk_var[i][m].values.reshape(-1, 1)
            yv = e[m].values
            mi = float(mutual_info_regression(x, yv, random_state=seed)[0])
            rows.append({"source": i, "target": j, "mi": mi, "n": int(m.sum())})
            for _ in range(n_shuffle):
                xp = x[rng.permutation(len(x))]
                nulls.append(float(mutual_info_regression(xp, yv, random_state=seed)[0]))
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges["mi_null_mean"] = float(np.mean(nulls)) if nulls else 0.0
    edges["mi_null_p95"] = float(np.percentile(nulls, 95)) if nulls else 0.0
    return edges.sort_values("mi", ascending=False).reset_index(drop=True)
