"""Unit tests for the DY (2014) generalized-FEVD connectedness builder.

Includes an INDEPENDENT recompute of the generalized-FEVD formula (not reusing the implementation's
vectorised code) per the "named-estimator must match published formula" rule, plus the required
properties: rows sum to ~1 after normalisation, directed/asymmetric, finite, train-only cutoff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import dy_connectedness as DY  # noqa: E402


def _reference_theta(A, sigma):
    """Independent, loop-based generalized-FEVD (Pesaran-Shin / DY 2014), NOT the module's code."""
    N = sigma.shape[0]
    theta = np.zeros((N, N))
    for i in range(N):
        e_i = np.zeros(N); e_i[i] = 1.0
        den = 0.0
        for Ah in A:
            den += e_i @ Ah @ sigma @ Ah.T @ e_i
        for j in range(N):
            e_j = np.zeros(N); e_j[j] = 1.0
            num = 0.0
            for Ah in A:
                num += (e_i @ Ah @ sigma @ e_j) ** 2
            theta[i, j] = (num / sigma[j, j]) / den
    return theta / theta.sum(axis=1, keepdims=True)


def test_gfevd_matches_independent_reference_formula():
    """generalized_fevd == a from-scratch loop over the published DY 2014 / Pesaran-Shin equation."""
    rng = np.random.default_rng(0)
    N = 4
    phi = 0.3 * rng.standard_normal((N, N)); phi *= 0.5 / max(np.abs(np.linalg.eigvals(phi)).max(), 1e-9)
    L = 0.2 * rng.standard_normal((N, N)); sigma = L @ L.T + np.eye(N)  # SPD
    A = DY.vma_from_var([phi], H=8)
    got = DY.generalized_fevd(A, sigma)
    ref = _reference_theta(A, sigma)
    assert np.allclose(got, ref, atol=1e-10)


def test_vma_recursion_two_lags():
    """A_1 = Phi_1; A_2 = Phi_1 A_1 + Phi_2 A_0 (independent recompute of the VAR->VMA recursion)."""
    rng = np.random.default_rng(1)
    N = 3
    phi1 = 0.1 * rng.standard_normal((N, N)); phi2 = 0.05 * rng.standard_normal((N, N))
    A = DY.vma_from_var([phi1, phi2], H=3)
    assert np.allclose(A[0], np.eye(N))
    assert np.allclose(A[1], phi1)
    assert np.allclose(A[2], phi1 @ phi1 + phi2 @ np.eye(N))


def test_rows_sum_to_one_and_finite():
    rng = np.random.default_rng(2)
    N = 6
    phi = 0.2 * rng.standard_normal((N, N)); phi *= 0.6 / max(np.abs(np.linalg.eigvals(phi)).max(), 1e-9)
    L = 0.3 * rng.standard_normal((N, N)); sigma = L @ L.T + np.eye(N)
    theta = DY.generalized_fevd(DY.vma_from_var([phi], 10), sigma)
    assert np.isfinite(theta).all()
    assert np.allclose(theta.sum(axis=1), np.ones(N), atol=1e-9)
    assert (theta >= -1e-12).all()                      # generalized FEVD shares are non-negative


def test_directed_asymmetric():
    """A non-symmetric VAR yields a directed (asymmetric) connectedness matrix."""
    phi = np.array([[0.5, 0.4, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])  # 1 leads 2, not vice-versa
    sigma = np.eye(3)
    theta = DY.generalized_fevd(DY.vma_from_var([phi], 10), sigma)
    assert not np.allclose(theta, theta.T)              # directed


def test_build_dy_adjacency_convention():
    """Top-K off-diagonal + self-loop=1.0; float32; shape [N,N]; edge weights are the spillover shares."""
    rng = np.random.default_rng(3)
    T, N = 400, 8
    panel = np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 1.0
    adj, theta, stats = DY.build_dy_adjacency(panel, p=1, H=10, top_k=3)
    assert adj.shape == (N, N) and adj.dtype == np.float32
    assert np.array_equal(np.diag(adj), np.ones(N, dtype=np.float32))     # self-loop=1.0
    off = adj.copy(); np.fill_diagonal(off, 0.0)
    assert (off > 0).sum(axis=1).max() <= 3                                # Top-K per row
    assert (off >= 0).all()                                               # spillover shares non-negative
    assert abs(stats["row_sum_mean"] - 1.0) < 1e-6                         # normalisation check
    assert stats["total_connectedness_index"] >= 0.0


def test_to_adjacency_no_self_loop():
    theta = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    adj = DY.to_adjacency(theta, top_k=1, self_loop=False)
    assert np.array_equal(np.diag(adj), np.zeros(3, dtype=np.float32))    # no self-loop
    assert (adj > 0).sum(axis=1).max() <= 1                               # Top-1 off-diagonal


def test_full_matrix_no_topk_is_dense_directed():
    rng = np.random.default_rng(4)
    T, N = 300, 5
    panel = np.cumsum(0.01 * rng.standard_normal((T, N)), axis=0) + 1.0
    adj, _, _ = DY.build_dy_adjacency(panel, p=1, H=10, top_k=None)
    off = adj.copy(); np.fill_diagonal(off, 0.0)
    assert (off > 0).sum() > N                            # dense (many off-diagonal edges)


def test_impute_and_zscore_handle_nan():
    x = np.array([[np.nan, 1.0], [2.0, np.nan], [3.0, 4.0]])
    filled = DY.impute_panel(x)
    assert np.isfinite(filled).all()
    assert filled[0, 0] == 2.0                            # bfill leading NaN
    assert filled[1, 1] == 1.0                            # ffill internal NaN
    z, mean, std = DY._zscore(filled)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9)


def test_train_vol_panel_is_train_only(tmp_path):
    """train_vol_panel keeps rows strictly before the first validation target date (no leakage), and
    aligns columns to the requested node order."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    for tk, base in (("AAA", 1.0), ("BBB", 2.0)):
        pd.DataFrame({"date": dates, "parkinson_variance": base + np.arange(10) * 1e-4}).to_csv(
            tmp_path / f"{tk}_processed.csv", index=False)
    files = [str(tmp_path / "AAA_processed.csv"), str(tmp_path / "BBB_processed.csv")]
    val_start = "2020-01-06"
    panel = DY.train_vol_panel(files, ["BBB", "AAA"], val_start)   # note reversed order
    assert panel.shape == (5, 2)                          # 5 rows strictly before 2020-01-06
    assert panel[0, 0] == pytest.approx(2.0)              # column 0 == BBB (node order respected)
    assert panel[0, 1] == pytest.approx(1.0)              # column 1 == AAA


def test_fit_var_rejects_bad_lag_and_short_panel():
    with pytest.raises(ValueError):
        DY.fit_var_elasticnet(np.zeros((10, 3)), p=0)
    with pytest.raises(ValueError):
        DY.fit_var_elasticnet(np.zeros((2, 3)), p=1)
    with pytest.raises(ValueError):
        DY.vma_from_var([np.eye(3)], H=0)
    with pytest.raises(ValueError):
        DY.to_adjacency(np.eye(3), top_k=-1)
