"""Incremental-over-HAR value of node features and edge definitions (leakage-safe).

The bar is HAR. For a target stock the base model is its OWN HAR features
(PK daily/weekly/monthly). A candidate (a node feature, or a Top-K neighbour edge
construction) is judged by whether ADDING it lowers out-of-sample RMSE of future PK
beyond HAR, and beyond HAR+market. Each candidate is fit on a shared valid mask so the
base and augmented models use identical (stock, test-row) pairs; per-stock RMSE deltas
give a sign test (Diebold-Mariano-style direction test across stocks).

All neighbour selection uses TRAIN-only matrices; all features are dated <= t; the target
is PK variance at t+h (strictly future). Nothing here uses test data for any decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .predictive import _diracc, _ridge_fit_predict


def har_base(pk_var_col: pd.Series) -> pd.DataFrame:
    """Own HAR features available at t: daily / weekly / monthly PK variance."""
    return pd.DataFrame(
        {
            "pk_d": pk_var_col,
            "pk_w": pk_var_col.rolling(5).mean(),
            "pk_m": pk_var_col.rolling(22).mean(),
        }
    )


def _incoming_neighbors(mat: pd.DataFrame, k: int, use_abs: bool) -> dict[str, list[str]]:
    """Top-K sources i for each target column j by mat[i, j] (directed, incoming)."""
    out: dict[str, list[str]] = {}
    for j in mat.columns:
        col = mat[j].drop(labels=[j], errors="ignore")
        rank = col.abs() if use_abs else col
        out[j] = list(rank.sort_values(ascending=False).head(k).index)
    return out


def _evaluate(
    target_wide: pd.DataFrame,
    base_of: dict[str, pd.DataFrame],
    addon_of: dict[str, pd.DataFrame],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    horizon: int,
) -> dict:
    """Fit base vs base+addon per stock on a shared mask; pool + per-stock RMSE deltas."""
    idx = target_wide.index
    tr_s = pd.Series(train_mask, index=idx)
    te_s = pd.Series(test_mask, index=idx)
    yb, pb, db = [], [], []      # base pooled
    yf, pf, df_ = [], [], []     # full (base+addon) pooled
    ps_base, ps_full = [], []    # per-stock RMSE
    for j in target_wide.columns:
        y = target_wide[j].shift(-horizon)
        base = base_of[j]
        addon = addon_of.get(j)
        full = base if addon is None else pd.concat([base, addon], axis=1)
        valid = y.notna() & base.notna().all(axis=1)
        if addon is not None:
            valid = valid & addon.notna().all(axis=1)
        tr, te = valid & tr_s, valid & te_s
        if tr.sum() < 50 or te.sum() < 10:
            continue
        y_tr, y_te = y[tr].values, y[te].values
        pos = y_tr[y_tr > 0]
        floor = float(np.percentile(pos, 1)) if pos.size else 1e-8
        p_base = np.clip(_ridge_fit_predict(base[tr].values, y_tr, base[te].values), floor, None)
        p_full = np.clip(_ridge_fit_predict(full[tr].values, y_tr, full[te].values), floor, None)
        yb.append(y_te)
        pb.append(p_base)
        db.append(_diracc(y_te, p_base))
        yf.append(y_te)
        pf.append(p_full)
        df_.append(_diracc(y_te, p_full))
        ps_base.append(float(np.sqrt(np.mean((p_base - y_te) ** 2))))
        ps_full.append(float(np.sqrt(np.mean((p_full - y_te) ** 2))))
    if not yb:
        return {}
    rmse_base = float(np.sqrt(np.mean((np.concatenate(pb) - np.concatenate(yb)) ** 2)))
    rmse_full = float(np.sqrt(np.mean((np.concatenate(pf) - np.concatenate(yf)) ** 2)))
    ps_base = np.array(ps_base)
    ps_full = np.array(ps_full)
    wins = int(np.sum(ps_full < ps_base))
    n = len(ps_base)
    return {
        "rmse_base": rmse_base,
        "rmse_full": rmse_full,
        "rmse_gain_pct": float(100 * (rmse_base - rmse_full) / rmse_base),
        "win_rate": float(wins / n),
        "n_stocks": n,
        "sign_p": float(stats.binomtest(wins, n).pvalue),
        "dir_acc_base": float(np.nanmean(db)),
        "dir_acc_full": float(np.nanmean(df_)),
    }


def rank_candidates(
    pk_var: pd.DataFrame,
    node_features: dict[str, pd.DataFrame],
    edge_neighbors: dict[str, tuple[dict[str, list[str]], pd.DataFrame]],
    market: pd.Series,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank node features and edge definitions by incremental OOS value over HAR.

    node_features: name -> wide (date x ticker) own-feature panel.
    edge_neighbors: name -> (neighbor_dict target->[sources], source_value_wide). The
        addon columns for target j are source_value_wide[sources] (values at t).
    Returns (node_rank, edge_rank). Edge ranking reports gain over HAR AND over HAR+market.
    """
    tickers = list(pk_var.columns)
    har_of = {j: har_base(pk_var[j]) for j in tickers}
    mkt = market.reindex(pk_var.index)
    harmkt_of = {
        j: har_base(pk_var[j]).assign(mkt=mkt.values, mkt_w=mkt.rolling(5).mean().values)
        for j in tickers
    }

    node_rows = []
    for name, wide in node_features.items():
        addon = {j: wide[[j]].rename(columns={j: name}) for j in tickers if j in wide}
        r = _evaluate(pk_var, har_of, addon, train_mask, test_mask, horizon)
        if r:
            node_rows.append({"node_feature": name, **r})
    node_rank = (
        pd.DataFrame(node_rows).sort_values("rmse_gain_pct", ascending=False)
        if node_rows else pd.DataFrame()
    )

    edge_rows = []
    for name, (neigh, src) in edge_neighbors.items():
        addon = {}
        for j in tickers:
            cols = [nb for nb in neigh.get(j, []) if nb in src.columns]
            if cols:
                a = src[cols].copy()
                a.columns = [f"{name}_{i}" for i in range(len(cols))]
                addon[j] = a
        over_har = _evaluate(pk_var, har_of, addon, train_mask, test_mask, horizon)
        over_mkt = _evaluate(pk_var, harmkt_of, addon, train_mask, test_mask, horizon)
        if over_har and over_mkt:
            edge_rows.append(
                {
                    "edge_definition": name,
                    "gain_over_har_pct": over_har["rmse_gain_pct"],
                    "winrate_over_har": over_har["win_rate"],
                    "sign_p_over_har": over_har["sign_p"],
                    "gain_over_market_pct": over_mkt["rmse_gain_pct"],
                    "winrate_over_market": over_mkt["win_rate"],
                    "sign_p_over_market": over_mkt["sign_p"],
                    "n_stocks": over_har["n_stocks"],
                }
            )
    edge_rank = (
        pd.DataFrame(edge_rows).sort_values("gain_over_market_pct", ascending=False)
        if edge_rows else pd.DataFrame()
    )
    return node_rank, edge_rank
