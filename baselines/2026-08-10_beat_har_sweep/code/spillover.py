"""Directed Diebold-Yilmaz spillover adjacency from a train-window VAR (C3/C5).

Implements the generalized forecast-error variance decomposition (Pesaran & Shin 1998) of a
reduced-form VAR(p), aggregated into the Diebold & Yilmaz (2012) directional connectedness table.
``adjacency[i, j]`` is the (row-normalized) share of node i's H-step forecast-error variance
attributable to shocks in node j -- a *directed* weight for node i attending to spillover source j.

The VAR is fit by OLS on the supplied panel (which MUST be the train window only, so the graph
structure never sees val/test data) and the resulting single static matrix is frozen for every
snapshot, consistent with the plan's stable-graph finding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fit_var(panel: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS VAR(lag): return companion-form MA driver ``A`` [N,N] (lag=1) and residual cov ``Sigma``.

    For lag>1 the coefficients are stacked into a companion matrix and the top-left NxN block plus
    the MA recursion below reproduce the impulse responses.
    """

    t_obs, n = panel.shape
    y = panel[lag:]
    x = np.column_stack([panel[lag - k - 1: t_obs - k - 1] for k in range(lag)])
    x = np.column_stack([x, np.ones(len(x))])  # intercept
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)  # [(n*lag+1), n]
    residuals = y - x @ coef
    sigma = residuals.T @ residuals / (len(y) - x.shape[1])
    coef_no_const = coef[:-1]  # [(n*lag), n]
    blocks = [coef_no_const[k * n:(k + 1) * n].T for k in range(lag)]  # each A_k is [N,N]
    return blocks, sigma


def _ma_coefficients(blocks: list[np.ndarray], horizon: int) -> list[np.ndarray]:
    """Vector MA(inf) coefficients Phi_h from VAR AR blocks: Phi_0=I, Phi_h=sum_k A_k Phi_{h-k}."""

    n = blocks[0].shape[0]
    phis = [np.eye(n)]
    for h in range(1, horizon):
        acc = np.zeros((n, n))
        for k, a_k in enumerate(blocks, start=1):
            if h - k >= 0:
                acc = acc + a_k @ phis[h - k]
        phis.append(acc)
    return phis


def directed_spillover_adjacency(
    panel: np.ndarray, var_lag: int = 1, fevd_horizon: int = 10
) -> np.ndarray:
    """Row-normalized generalized-FEVD connectedness matrix [N, N] from a train-window panel.

    Args:
        panel: ``[T, N]`` volatility series (train window only), columns ordered by ticker_id.
        var_lag: VAR order p.
        fevd_horizon: forecast horizon H for the variance decomposition.

    Returns:
        ``adjacency[i, j]`` = share of i's H-step forecast-error variance from shocks in j; rows
        sum to 1. Directed (asymmetric in general); the diagonal carries the own-variance share.
    """

    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2 or panel.shape[1] < 2:
        raise ValueError("spillover panel must be [T, N] with at least two series")
    if panel.shape[0] <= panel.shape[1] * var_lag + 2:
        raise ValueError("spillover panel has too few observations for the requested VAR")
    if np.ptp(panel, axis=0).min() == 0:
        raise ValueError("spillover panel has a constant (zero-variance) series")

    blocks, sigma = _fit_var(panel, var_lag)
    n = panel.shape[1]
    sigma_diag = np.diag(sigma).copy()
    if (sigma_diag <= 0).any() or not np.isfinite(sigma).all():
        raise ValueError("degenerate VAR residual covariance")
    phis = _ma_coefficients(blocks, fevd_horizon)

    theta = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        denom = 0.0
        for phi in phis:
            denom += float(e_i @ phi @ sigma @ phi.T @ e_i)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            num = 0.0
            for phi in phis:
                num += float(e_i @ phi @ sigma @ e_j) ** 2
            theta[i, j] = (num / sigma_diag[j]) / denom
    row_sums = theta.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("degenerate FEVD row (zero total variance share)")
    return (theta / row_sums).astype(np.float32)


def load_train_volatility_panel(
    processed_dir: Path | str, tickers_ordered: list[str], train_end_date: str
) -> np.ndarray:
    """Aligned ``[T, N]`` Parkinson-variance panel over dates <= train_end_date (train only).

    Columns are ordered to match ``tickers_ordered`` (i.e. by ticker_id). Dates are inner-joined
    across all tickers so the VAR sees a common, gap-free axis.
    """

    processed = Path(processed_dir)
    frames = []
    for ticker in tickers_ordered:
        frame = pd.read_csv(processed / f"{ticker}_processed.csv")
        frame = frame[frame["date"] <= train_end_date][["date", "parkinson_volatility"]]
        frames.append(frame.rename(columns={"parkinson_volatility": ticker}).set_index("date"))
    merged = pd.concat(frames, axis=1, join="inner").sort_index()
    panel = merged[tickers_ordered].to_numpy(dtype=float)
    if not np.isfinite(panel).all() or panel.shape[0] < 2 * len(tickers_ordered):
        raise ValueError("train volatility panel is too short or non-finite after alignment")
    return panel
