"""Supervised edge discovery: rank DIRECTED cross-stock edges by out-of-sample ΔQLIKE.

The prior EDA ranked edges by RMSE gain over HAR/HAR+market at h=1. But the deep models beat HAR
on QLIKE (calibration), not squared error, so an edge useful on QLIKE could be missed. This module
answers the sharper question, on the metric that matters and conditioned on the winning node
features (E2 = HAR + MarketPK + volume_zscore_20):

  Does knowing source i's history LOWER the out-of-sample QLIKE of target j's t+h Parkinson
  variance, AFTER the base already has j's own HAR + the market factor + j's volume z-score?

Leakage-safe: ridge coefficients fit on TRAIN; edges selected by ΔQLIKE>0 on VAL and confirmed on
TEST (sign-stable); BH-FDR over all directed pairs; regime split uses TRAIN-only MarketPK quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from graph_eda.predictive import _ridge_fit_predict
from graph_eda.relationships import benjamini_hochberg, cross_lead_lag_matrix, market_adjust_residuals


def _har(pk_col: pd.Series) -> pd.DataFrame:
    """Own HAR-style predictors available at t (daily / weekly / monthly PK variance)."""
    return pd.DataFrame({"pk_d": pk_col, "pk_w": pk_col.rolling(5).mean(),
                         "pk_m": pk_col.rolling(22).mean()})


def _qlike_obs(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-observation QLIKE for positive targets (NaN elsewhere); predictions floored at eps."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, None)
    out = np.full(y_true.shape, np.nan)
    pos = y_true > 0
    ratio = y_true[pos] / y_pred[pos]
    out[pos] = ratio - np.log(ratio) - 1.0
    return out


def predictive_dqlike_edges(
    pk_var: pd.DataFrame,
    market: pd.Series,
    volz: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    horizon: int = 5,
    lam: float = 1e-4,
    with_regime: bool = False,
    predict_fn=None,
) -> pd.DataFrame:
    """Directed ΔQLIKE edge table: for each ordered (source i -> target j), the OOS QLIKE reduction
    from adding source i's PK history to the E2 base model of target j.

    Positive ``dqlike_*`` means source i lowers target j's QLIKE (helps). Base and augmented models
    are fit on the SAME per-pair valid mask so the comparison is like-for-like. Returns one row per
    ordered pair with val/test ΔQLIKE, a paired-t p-value on the test per-obs QLIKE difference,
    BH-FDR q, a sign-stability flag, and (optional) low/high-MarketPK-regime test ΔQLIKE.

    ``predict_fn(x_tr, y_tr, x_te) -> preds`` overrides the model (default: ridge); the nonlinear
    (gradient-boosting) edge test reuses this same leakage-safe harness with a GB predictor.
    """
    if predict_fn is None:
        def predict_fn(x_tr, y_tr, x_te):
            return _ridge_fit_predict(x_tr, y_tr, x_te, lam)
    idx = pk_var.index
    tickers = list(pk_var.columns)
    tr = pd.Series(train_mask, index=idx)
    va = pd.Series(val_mask, index=idx)
    te = pd.Series(test_mask, index=idx)
    mkt = market.reindex(idx)
    lo_thr = hi_thr = None
    if with_regime:
        lo_thr, hi_thr = mkt[train_mask].quantile([1 / 3, 2 / 3])

    rows = []
    for j in tickers:
        y = pk_var[j].shift(-horizon)
        base = pd.concat([_har(pk_var[j]), mkt.rename("mkt"), volz[j].rename("volz")], axis=1)
        base_valid = y.notna() & base.notna().all(axis=1)
        for i in tickers:
            if i == j:
                continue
            src = pk_var[i].rename("src")
            valid = base_valid & src.notna()
            trm, vam, tem = valid & tr, valid & va, valid & te
            if trm.sum() < 100 or vam.sum() < 20 or tem.sum() < 20:
                continue
            full = pd.concat([base, src], axis=1)
            y_tr = y[trm].values
            yb_val, yb_te = y[vam].values, y[tem].values
            # positivity floor from TRAIN positive targets, applied identically to base and full
            # predictions (mirrors the models' floor) so QLIKE is not dominated by tail spikes
            pos_tr = y_tr[y_tr > 0]
            floor = float(np.percentile(pos_tr, 1)) if pos_tr.size else 1e-8

            def _fit(x_te_key):
                pb = predict_fn(base[trm].values, y_tr, x_te_key(base))
                pf = predict_fn(full[trm].values, y_tr, x_te_key(full))
                return np.clip(pb, floor, None), np.clip(pf, floor, None)

            pb_val, pf_val = _fit(lambda X: X[vam].values)
            pb_te, pf_te = _fit(lambda X: X[tem].values)
            d_val = _qlike_obs(yb_val, pb_val) - _qlike_obs(yb_val, pf_val)
            d_te = _qlike_obs(yb_te, pb_te) - _qlike_obs(yb_te, pf_te)
            dd = d_te[~np.isnan(d_te)]
            p_test = (float(stats.ttest_1samp(dd, 0.0).pvalue)
                      if dd.size > 2 and np.nanstd(dd) > 0 else 1.0)
            row = {"source": i, "target": j,
                   "dqlike_val": float(np.nanmean(d_val)),
                   "dqlike_test": float(np.nanmean(d_te)),
                   "p_test": p_test, "n_test": int(tem.sum())}
            if with_regime:
                mkt_te = mkt[tem].values
                lo, hi = mkt_te <= lo_thr, mkt_te > hi_thr
                row["dqlike_test_lo_regime"] = (float(np.nanmean(d_te[lo]))
                                                if lo.sum() > 2 else np.nan)
                row["dqlike_test_hi_regime"] = (float(np.nanmean(d_te[hi]))
                                                if hi.sum() > 2 else np.nan)
            rows.append(row)
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges["fdr_q"] = benjamini_hochberg(edges["p_test"].values)
    edges["stable"] = (edges["dqlike_val"] > 0) & (edges["dqlike_test"] > 0)
    return edges.sort_values("dqlike_test", ascending=False).reset_index(drop=True)


def residual_leadlag(
    pk_vol: pd.DataFrame, market: pd.Series, train_mask: np.ndarray, horizon: int = 5
) -> pd.DataFrame:
    """Directed lead-lag of MARKET-RESIDUAL Parkinson vol: L[i,j] = Corr(eps_i(t), eps_j(t+h)).

    eps = pk_vol regressed on the market factor with TRAIN-only betas (removes the common factor),
    so a surviving lead-lag is cross-stock structure beyond the market. Diagonal is the own h-lag
    residual autocorrelation (not forced to 1).
    """
    resid = market_adjust_residuals(pk_vol, market, train_mask)
    resid = resid.dropna(how="all")
    return cross_lead_lag_matrix(resid, resid, horizon)
