"""P5: graphical-LASSO (partial-correlation) edge. Partial correlation from a precision matrix
removes the common market factor (EDA plan section 41), so a LASSO edge is the market-factor-robust
alternative to the vol->PK edge. Core math (precision->partial-corr, Top-K adjacency) must be exact.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import edges_glasso as eg  # noqa: E402


def test_precision_to_partial_corr_formula():
    # partial corr = -P_ij / sqrt(P_ii P_jj); diagonal set to 1
    P = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    pc = eg.precision_to_partial_corr(P)
    assert np.allclose(np.diag(pc), 1.0)
    assert np.isclose(pc[0, 1], 1.0 / 2.0)      # -(-1)/sqrt(2*2)=0.5
    assert np.isclose(pc[1, 2], 0.5)
    assert np.isclose(pc[0, 2], 0.0)
    assert np.allclose(pc, pc.T)                 # symmetric


def test_topk_adjacency_keeps_top_sources_and_self_loop():
    # partial-corr matrix; ticker order A,B,C,D
    names = ["A", "B", "C", "D"]
    pc = pd.DataFrame(np.array([
        [1.0, 0.9, 0.1, 0.8],
        [0.9, 1.0, 0.7, 0.2],
        [0.1, 0.7, 1.0, 0.6],
        [0.8, 0.2, 0.6, 1.0]]), index=names, columns=names)
    ttid = {"A": 0, "B": 1, "C": 2, "D": 3}
    A = eg.topk_adjacency(pc, ttid, top_k=2)
    assert A.shape == (4, 4)
    assert np.allclose(np.diag(A), 1.0)          # self-loops
    # target A's top-2 non-self sources by |pcorr|: B(0.9), D(0.8)
    assert A[0, 1] != 0.0 and A[0, 3] != 0.0
    assert A[0, 2] == 0.0                          # C(0.1) excluded


def test_glasso_partial_corr_recovers_strong_pair():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.normal(size=n)                         # shared driver for A,B
    panel = pd.DataFrame({
        "A": z + 0.1 * rng.normal(size=n),
        "B": z + 0.1 * rng.normal(size=n),         # A,B strongly related
        "C": rng.normal(size=n),                   # independent
    })
    pc = eg.glasso_partial_corr(panel, alpha=0.01)
    assert pc.shape == (3, 3)
    assert np.allclose(np.diag(pc.to_numpy()), 1.0)
    assert abs(pc.loc["A", "B"]) > abs(pc.loc["A", "C"])   # A-B edge strongest
