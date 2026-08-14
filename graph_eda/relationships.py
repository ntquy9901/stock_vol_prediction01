"""Pairwise relationships: correlation + significance, lead-lag, market adjustment.

Leakage note: full-period matrices here are DESCRIPTIVE EDA (plan sections 8-9 request
them explicitly). All quantities that feed a *decision* (regime thresholds, market
betas, neighbour selection, scalers) are fit on TRAIN-only slices by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pairwise_corr(wide: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Symmetric contemporaneous correlation matrix over aligned columns."""
    return wide.corr(method=method)


def pair_long_table(
    wide: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Long table of unique unordered pairs with corr, raw p-value, n, and BH q-value.

    Uses complete-observation pairwise rows. p-values are analytic (Pearson t / Spearman
    t approximation); BH-FDR is applied across all pairs (plan section 16).
    """
    cols = list(wide.columns)
    rows = []
    arr = wide
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = arr.iloc[:, i]
            b = arr.iloc[:, j]
            m = a.notna() & b.notna()
            n = int(m.sum())
            if n < 10:
                r, p = np.nan, np.nan
            elif method == "pearson":
                r, p = stats.pearsonr(a[m], b[m])
            else:
                r, p = stats.spearmanr(a[m], b[m])
            rows.append((cols[i], cols[j], r, p, n))
    df = pd.DataFrame(rows, columns=["source", "target", "corr", "raw_p", "n_obs"])
    df["fdr_q"] = benjamini_hochberg(df["raw_p"].values)
    df["significant_fdr_0.05"] = df["fdr_q"] < 0.05
    return df


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (NaN-safe; NaN p -> NaN q)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[valid] = out
    return q


def permutation_null_mean_abs_corr(
    wide: pd.DataFrame, n_iter: int = 500, method: str = "pearson", seed: int = 0
) -> dict:
    """Shuffle-null for cross-stock dependence (plan section 17).

    Independently permutes each column's time order (destroying cross-sectional
    alignment while preserving marginals), recomputes the mean absolute off-diagonal
    pairwise correlation, and compares the observed statistic to the null distribution.
    """
    rng = np.random.default_rng(seed)
    x = wide.dropna(how="any")
    obs = _mean_abs_offdiag(x.corr(method=method).values)
    null = np.empty(n_iter)
    vals = x.values
    n, k = vals.shape
    for it in range(n_iter):
        perm = np.empty_like(vals)
        for c in range(k):
            perm[:, c] = vals[rng.permutation(n), c]
        pdf = pd.DataFrame(perm, columns=x.columns)
        null[it] = _mean_abs_offdiag(pdf.corr(method=method).values)
    p_emp = float((np.sum(null >= obs) + 1) / (n_iter + 1))
    return {
        "observed_mean_abs_corr": float(obs),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "null_max": float(null.max()),
        "p_value_empirical": p_emp,
        "n_iter": n_iter,
    }


def bootstrap_ci_mean(
    values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0, use_abs: bool = False
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (plan section 56).

    NaNs are dropped; ``use_abs`` takes absolute values first (for |correlation| means).
    Returns (nan, nan) when no finite value remains.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if use_abs:
        v = np.abs(v)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _mean_abs_offdiag(mat: np.ndarray) -> float:
    k = mat.shape[0]
    mask = ~np.eye(k, dtype=bool)
    return float(np.nanmean(np.abs(mat[mask])))


def cross_corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation matrix C[i,j] = corr(x[:,i], y[:,j]).

    Rows with any NaN in either matrix are dropped pairwise-globally (complete-case over
    the shared row set). Vectorised (standardise then matrix product).
    """
    m = ~(np.isnan(x).any(1) | np.isnan(y).any(1))
    xs = x[m]
    ys = y[m]
    n = xs.shape[0]
    if n < 10:
        return np.full((x.shape[1], y.shape[1]), np.nan)
    xsd = xs.std(0)
    ysd = ys.std(0)
    xsd[xsd == 0] = np.nan  # constant column -> undefined correlation, not inf
    ysd[ysd == 0] = np.nan
    xz = (xs - xs.mean(0)) / xsd
    yz = (ys - ys.mean(0)) / ysd
    return (xz.T @ yz) / n


def lead_lag_matrix(wide: pd.DataFrame, k: int, method: str = "pearson") -> pd.DataFrame:
    """Directed lead-lag matrix L[i,j] = Corr(X_i(t), X_j(t+k)).

    Entry (row i, col j) is the association of i's value today with j's value k days
    ahead. Leakage-safe: only future targets are shifted; the estimate is directional
    and NOT symmetric. Pearson uses a vectorised path; Spearman ranks then reuses it.
    """
    cols = list(wide.columns)
    base = wide.iloc[:-k] if k > 0 else wide
    lead = wide.shift(-k).iloc[:-k] if k > 0 else wide
    if method == "spearman":
        base = base.rank()
        lead = lead.rank()
    c = cross_corr_matrix(base.values, lead.values)
    return pd.DataFrame(c, index=cols, columns=cols)


def cross_lead_lag_matrix(
    source_wide: pd.DataFrame, target_wide: pd.DataFrame, k: int
) -> pd.DataFrame:
    """Directed cross-feature lead-lag L[i,j] = Corr(source_i(t), target_j(t+k))."""
    cols = list(source_wide.columns)
    src = source_wide.iloc[:-k] if k > 0 else source_wide
    tgt = target_wide.shift(-k).iloc[:-k] if k > 0 else target_wide
    c = cross_corr_matrix(src.values, tgt.values)
    return pd.DataFrame(c, index=cols, columns=cols)


def directional_asymmetry(lead1: pd.DataFrame) -> pd.DataFrame:
    """A_ij = L_ij(1) - L_ji(1) for the lag-1 lead-lag matrix (plan section 15)."""
    return lead1 - lead1.T


def market_factor(pk_wide: pd.DataFrame, how: str = "median") -> pd.Series:
    """Market-wide daily Parkinson factor (plan section 41): median or mean over stocks."""
    return pk_wide.median(axis=1) if how == "median" else pk_wide.mean(axis=1)


def market_adjust_residuals(
    pk_wide: pd.DataFrame, market: pd.Series, train_mask: np.ndarray
) -> pd.DataFrame:
    """Residuals of PK_i regressed on the market factor with TRAIN-only betas.

    Fits alpha_i + beta_i * Market on training rows only, then applies those
    coefficients to the full sample (plan sections 32/41). Residual correlation of these
    residuals is the market-adjusted structure.
    """
    resid = pd.DataFrame(index=pk_wide.index, columns=pk_wide.columns, dtype=float)
    mkt = market.values
    for c in pk_wide.columns:
        y = pk_wide[c].values
        tm = train_mask & ~np.isnan(y) & ~np.isnan(mkt)
        x_tr = np.column_stack([np.ones(tm.sum()), mkt[tm]])
        beta, *_ = np.linalg.lstsq(x_tr, y[tm], rcond=None)
        resid[c] = y - (beta[0] + beta[1] * mkt)
    return resid
