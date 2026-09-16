"""Covariance estimators: each maps a train return matrix R[T x N] -> Sigma_hat[N x N] (PSD, invertible).

Ladder rungs #1-#3 (sample / EWMA / Ledoit-Wolf). DCC / factor / graph are added later. Every estimator
returns a PSD, ridged matrix so the global-minimum-variance weights are well defined. A `with_gbm_diagonal`
helper re-scales any Sigma_hat to `D_gbm R D_gbm` (GBM marginals on the diagonal, estimator's correlation on
the off-diagonals) for the DCC-style GBM combination (design section 12).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

import cov_config as C


def psd_ridge(S: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Project S to the nearest PSD matrix (clip negative eigenvalues to 0) then add a relative ridge
    `eps * mean(diag)` to the diagonal so it is strictly positive-definite / invertible."""
    eps = C.RIDGE_EPS if eps is None else eps
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0.0, None)
    S = (V * w) @ V.T
    ridge = eps * float(np.mean(np.diag(S))) if np.mean(np.diag(S)) > 0 else eps
    return S + ridge * np.eye(S.shape[0])


def sample_cov(R: np.ndarray) -> np.ndarray:
    """Plain sample covariance (rolling window), PSD-ridged. Singular when N >~ T; the ridge rescues it."""
    return psd_ridge(np.cov(R, rowvar=False))


def ewma_cov(R: np.ndarray, lam: float | None = None) -> np.ndarray:
    """RiskMetrics zero-mean EWMA covariance: Sigma = sum_t w_t r_t r_t^T with w_t ∝ lam^(T-1-t),
    normalized. Most weight on the most recent rows. PSD-ridged."""
    lam = C.EWMA_LAMBDA if lam is None else lam
    T = R.shape[0]
    ages = np.arange(T - 1, -1, -1)          # row 0 is oldest -> largest age
    w = lam ** ages
    w /= w.sum()
    Rw = R * np.sqrt(w)[:, None]             # weighting under a zero-mean assumption
    return psd_ridge(Rw.T @ Rw)


def ledoit_wolf(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf linear shrinkage toward a scaled identity (sklearn). The strong baseline the factor/graph
    rungs must beat. PSD by construction; a tiny ridge keeps invertibility uniform with the others."""
    lw = LedoitWolf(assume_centered=False).fit(R)
    return psd_ridge(lw.covariance_)


def factor_rank_k(R: np.ndarray, k: int | None = None) -> np.ndarray:
    """POET/PCA approximate factor covariance: Sigma = B F B^T + D. Keep the top-k principal components of the
    sample covariance as the systematic (low-rank) part and the residual diagonal as idiosyncratic variance.
    This is the 'graph as a low-rank prior' rung — cross-stock structure compressed to k common factors."""
    k = C.FACTOR_K if k is None else k
    S = np.cov(R, rowvar=False)
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)                  # ascending eigenvalues
    idx = np.argsort(w)[::-1][:k]             # top-k
    Bk, Fk = V[:, idx], np.clip(w[idx], 0.0, None)
    systematic = (Bk * Fk) @ Bk.T            # B F B^T
    resid = np.clip(np.diag(S) - np.diag(systematic), 0.0, None)   # idiosyncratic variances
    return psd_ridge(systematic + np.diag(resid))


def with_gbm_diagonal(sigma: np.ndarray, gbm_std: np.ndarray) -> np.ndarray:
    """Return `D_gbm R D_gbm`: keep the estimator's CORRELATION structure R = D^-1 Sigma D^-1 but replace the
    marginal volatilities with the GBM per-stock forecasts `gbm_std` (design section 12 DCC combination)."""
    d = np.sqrt(np.clip(np.diag(sigma), 1e-300, None))
    inv = 1.0 / d
    corr = sigma * np.outer(inv, inv)        # R = D^-1 Sigma D^-1
    g = np.asarray(gbm_std, float)
    return psd_ridge((corr * np.outer(g, g)))


ESTIMATORS = {"sample": sample_cov, "ewma": ewma_cov, "ledoit_wolf": ledoit_wolf,
              "factor_rank_k": factor_rank_k}
