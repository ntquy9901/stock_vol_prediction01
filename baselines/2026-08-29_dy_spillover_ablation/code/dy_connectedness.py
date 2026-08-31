"""Directed volatility-connectedness adjacency following Diebold & Yilmaz (2014) EXACTLY.

Source: Diebold, F.X. & Yilmaz, K. (2014), "On the network topology of variance decompositions:
Measuring the connectedness of financial firms", *Journal of Econometrics* 182(1):119-134. The
connectedness network is the row-normalised GENERALIZED forecast-error-variance decomposition
(generalized FEVD of Pesaran & Shin 1998, *Economics Letters* 58(1):17-29), which DY 2014 adopt so the
network is invariant to variable ordering (unlike a Cholesky FEVD).

Exact equations implemented (DY 2014, their generalized variance-decomposition definition):

  VAR(p):     x_t = sum_{i=1}^{p} Phi_i x_{t-i} + eps_t,   eps_t ~ (0, Sigma)
  VMA(inf):   x_t = sum_{h=0}^{inf} A_h eps_{t-h},   A_0 = I_N,   A_h = sum_{i=1}^{p} Phi_i A_{h-i}
                                                     (A_h = 0 for h < 0)

  Generalized FEVD (DY 2014 Eq. for theta_ij(H); Pesaran-Shin):

                sigma_jj^{-1} * sum_{h=0}^{H-1} ( e_i' A_h Sigma e_j )^2
  theta_ij(H) = --------------------------------------------------------          (DY 2014)
                     sum_{h=0}^{H-1} ( e_i' A_h Sigma A_h' e_i )

  where e_i is the i-th selection (unit) vector and sigma_jj = Sigma[j, j]. Because the generalized FEVD
  does not use an orthogonalised shock, rows do not sum to 1, so DY row-normalise:

  theta_tilde_ij(H) = theta_ij(H) / sum_{k=1}^{N} theta_ik(H)   =>  sum_k theta_tilde_ik = 1   (DY 2014)

The row-normalised theta_tilde is the DIRECTED connectedness network: theta_tilde_ij = the fraction of
i's H-step forecast-error variance attributable to shocks in j, i.e. a directed edge j -> i. It is used
directly as the model adjacency A[i, j] (edge from source j into target i), matching the vol->PK /
correlation / sector adjacencies' [target i, source j] convention.

High-dimensionality (DY's original firm panel is small; HNX has N ~ 154 so an unregularised VAR is
ill-posed): follow the standard high-dimensional-connectedness fix of Demirer, Diebold, Liu & Yilmaz
(2018), "Estimating global bank network connectedness", *Journal of Applied Econometrics* 33(1):1-15,
who extend DY to high dimensions by estimating each VAR equation with an elastic-net / LASSO penalty.
Here: VAR(1), elastic-net (l1_ratio=0.5) per equation, on per-ticker z-scored TRAIN series, FEVD horizon
H=10. Everything is estimated on TRAIN rows only and frozen for validation/test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


def train_vol_panel(files: list[str], tickers: list[str], val_start_date: str) -> np.ndarray:
    """Wide (date x ticker) Parkinson-variance TRAIN panel aligned to ``tickers`` (node order), keeping
    only rows STRICTLY BEFORE ``val_start_date`` (the first validation target date, ``D.d_va[0]``).

    This is the leakage-safe train window: no validation/test row enters the VAR estimation, and the
    resulting matrix is frozen. Columns are reindexed to ``tickers`` (the post-node-drop universe from
    the masked panel), so the DY matrix aligns exactly with ``D.tickers``.
    Returns ``[T_train, N]`` float array WITH NaN where a ticker is missing (imputed downstream).
    """
    series = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        tk = Path(f).name.replace("_processed.csv", "")
        series[tk] = df.set_index("date")["parkinson_variance"]
    wide = pd.DataFrame(series).sort_index()
    wide = wide.reindex(columns=list(tickers))          # align to node order (drops extras)
    cutoff = pd.Timestamp(val_start_date)
    wide = wide.loc[wide.index < cutoff]                # strictly before the first validation target
    return wide.to_numpy(dtype=float)


def impute_panel(x: np.ndarray) -> np.ndarray:
    """Complete the ``[T,N]`` panel for a balanced VAR: per-column forward-fill then back-fill (within the
    TRAIN window only -> no future val/test information). A fully-empty column is filled with 0.0."""
    df = pd.DataFrame(x).ffill().bfill()
    return df.fillna(0.0).to_numpy(dtype=float)


def _zscore(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column z-score (train stats). Returns (z, mean, std) with std floored at 1e-12."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def fit_var_elasticnet(z: np.ndarray, p: int = 1, alpha: float = 0.05,
                       l1_ratio: float = 0.5) -> tuple[list[np.ndarray], np.ndarray]:
    """Elastic-net VAR(``p``) fitted equation-by-equation (Demirer et al. 2018 high-dim connectedness).

    ``z`` is ``[T,N]`` (z-scored). Returns (``[Phi_1..Phi_p]`` each ``[N,N]``, residual covariance
    ``Sigma`` ``[N,N]``). Row i of ``Phi_k`` are equation i's coefficients on lag-k of all N series.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    T, N = z.shape
    if T <= p + 1:
        raise ValueError(f"not enough train rows ({T}) for VAR({p})")
    # design: predict x_t from [x_{t-1}, ..., x_{t-p}]  -> features [T-p, N*p]
    y = z[p:]                                               # [T-p, N]
    lags = [z[p - k: T - k] for k in range(1, p + 1)]       # each [T-p, N], lag k
    X = np.concatenate(lags, axis=1)                        # [T-p, N*p]
    phis = [np.zeros((N, N)) for _ in range(p)]
    resid = np.zeros_like(y)
    for i in range(N):
        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                        max_iter=5000, random_state=0)
        en.fit(X, y[:, i])
        coef = en.coef_                                     # [N*p]
        for k in range(p):
            phis[k][i] = coef[k * N:(k + 1) * N]
        resid[:, i] = y[:, i] - en.predict(X)
    sigma = np.cov(resid, rowvar=False, ddof=1)             # [N,N] residual covariance
    sigma = np.atleast_2d(sigma)
    return phis, sigma


def vma_from_var(phis: list[np.ndarray], H: int) -> list[np.ndarray]:
    """Invertible MA(inf) coefficients A_0..A_{H-1} from VAR coefficient matrices.

    A_0 = I; A_h = sum_{i=1}^{p} Phi_i A_{h-i} (A_h = 0 for h < 0). Standard VAR->VMA recursion.
    """
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")
    N = phis[0].shape[0]
    p = len(phis)
    A = [np.eye(N)]
    for h in range(1, H):
        Ah = np.zeros((N, N))
        for i in range(1, p + 1):
            if h - i >= 0:
                Ah += phis[i - 1] @ A[h - i]
        A.append(Ah)
    return A


def generalized_fevd(A: list[np.ndarray], sigma: np.ndarray) -> np.ndarray:
    """Row-normalised generalized FEVD ``theta_tilde`` ``[N,N]`` (DY 2014 / Pesaran-Shin).

    theta_ij = sigma_jj^{-1} * sum_h (A_h Sigma)[i,j]^2  /  sum_h (A_h Sigma A_h')[i,i];
    theta_tilde_ij = theta_ij / sum_k theta_ik  (rows sum to 1). ``A`` = A_0..A_{H-1}.
    """
    N = sigma.shape[0]
    sig_jj = np.diag(sigma).copy()
    sig_jj = np.where(sig_jj < 1e-300, 1e-300, sig_jj)      # guard a degenerate (zero-variance) shock
    num = np.zeros((N, N))
    den = np.zeros(N)
    for Ah in A:
        AS = Ah @ sigma                                     # [N,N]; (A_h Sigma)[i,j]
        num += AS ** 2                                      # sum_h (e_i' A_h Sigma e_j)^2
        den += np.einsum("ij,ij->i", AS, Ah)               # (A_h Sigma A_h')[i,i] = sum_j AS[i,j]*Ah[i,j]
    num = num / sig_jj[None, :]                             # sigma_jj^{-1}
    den = np.where(den < 1e-300, 1e-300, den)
    theta = num / den[:, None]
    row = theta.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-300, 1e-300, row)
    return theta / row                                      # theta_tilde, rows sum to 1


def to_adjacency(theta_tilde: np.ndarray, top_k: int | None = 5,
                 self_loop: bool = True) -> np.ndarray:
    """Model adjacency from ``theta_tilde``: keep the Top-K off-diagonal spillover SOURCES per row
    (largest theta_tilde_ij, j != i), zero the rest, and set the diagonal self-loop to 1.0.

    ``top_k=None`` keeps the full dense directed matrix (off-diagonal spillover weights) with a unit
    self-loop. Matches the vol->PK adjacency convention (Top-K signed sources per target + self-loop=1),
    so the same WeightedGATLayer consumes it unchanged. Returns ``[N,N]`` float32.
    """
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be None or >= 0, got {top_k}")
    N = theta_tilde.shape[0]
    off = theta_tilde.copy()
    np.fill_diagonal(off, 0.0)
    A = np.zeros((N, N), dtype=np.float32)
    if top_k is None:
        A = off.astype(np.float32)
    else:
        for i in range(N):
            k = np.argsort(-off[i])[:top_k]                 # K largest spillover sources into i
            A[i, k] = off[i, k].astype(np.float32)
    if self_loop:
        np.fill_diagonal(A, 1.0)
    return A


def connectedness_stats(theta_tilde: np.ndarray) -> dict:
    """DY 2014 network summaries of the full ``theta_tilde`` (before Top-K / self-loop).

    - total_connectedness_index: DY's C = 100 * (sum of OFF-diagonal theta_tilde) / N  (percent of total
      forecast-error variance coming from cross-firm spillovers).
    - from_others / to_others: mean directional connectedness FROM others (row off-sum) and TO others
      (column off-sum). row_sum_mean should be ~1 (normalisation check).
    """
    N = theta_tilde.shape[0]
    off = theta_tilde.copy()
    np.fill_diagonal(off, 0.0)
    from_others = off.sum(axis=1)                           # i receives from all j != i
    to_others = off.sum(axis=0)                             # j transmits to all i != j
    return {
        "n_nodes": int(N),
        "row_sum_mean": float(theta_tilde.sum(axis=1).mean()),
        "row_sum_min": float(theta_tilde.sum(axis=1).min()),
        "row_sum_max": float(theta_tilde.sum(axis=1).max()),
        "total_connectedness_index": float(100.0 * off.sum() / N),
        "mean_from_others": float(from_others.mean()),
        "mean_to_others": float(to_others.mean()),
        "max_to_others": float(to_others.max()),
        "diag_mean_own_share": float(np.diag(theta_tilde).mean()),
        "asymmetry_frob": float(np.linalg.norm(theta_tilde - theta_tilde.T)),
    }


def build_dy_adjacency(train_panel: np.ndarray, p: int = 1, H: int = 10, alpha: float = 0.05,
                       l1_ratio: float = 0.5, top_k: int | None = 5,
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full DY (2014) pipeline on a TRAIN volatility panel ``[T,N]`` (may contain NaN).

    Returns (adjacency ``[N,N]`` float32 for the model, full ``theta_tilde`` ``[N,N]``, stats dict).
    """
    x = impute_panel(np.asarray(train_panel, dtype=float))
    z, _, _ = _zscore(x)
    phis, sigma = fit_var_elasticnet(z, p=p, alpha=alpha, l1_ratio=l1_ratio)
    A = vma_from_var(phis, H)
    theta_tilde = generalized_fevd(A, sigma)
    adj = to_adjacency(theta_tilde, top_k=top_k, self_loop=True)
    stats = connectedness_stats(theta_tilde)
    stats.update({"var_lag_p": int(p), "fevd_horizon_H": int(H), "alpha": float(alpha),
                  "l1_ratio": float(l1_ratio), "top_k": (None if top_k is None else int(top_k)),
                  "n_train_rows": int(x.shape[0])})
    return adj, theta_tilde, stats
