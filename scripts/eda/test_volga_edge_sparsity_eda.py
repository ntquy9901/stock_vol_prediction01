"""Tests for the pure statistical helpers of the VolGA edge-sparsity EDA (the data/JSON I/O is pragma'd)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volga_edge_sparsity_eda as E  # noqa: E402


def test_density():
    kept = np.zeros((3, 3), dtype=bool)
    kept[0, 1] = kept[1, 0] = kept[2, 0] = True          # 3 edges, n=3 -> 3 / (3*2)
    assert E._density(kept, 3) == 3 / 6


def test_corr_matrices_recovers_pm1_and_nan_diagonal():
    src = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]])
    tgt = src.copy()
    R, npair = E._corr_matrices(src, tgt, min_pairs=2)
    assert np.isnan(R[0, 0]) and np.isnan(R[1, 1])       # diagonal blanked
    assert abs(R[0, 1] + 1.0) < 1e-9                     # col0 vs col1 anti-correlated
    assert abs(R[1, 0] + 1.0) < 1e-9
    assert (npair == 5).all()


def test_corr_matrices_min_pairs_blanks_thin():
    src = np.array([[1.0, 2.0], [2.0, 1.0]])
    tgt = src.copy()
    R, _ = E._corr_matrices(src, tgt, min_pairs=5)        # only 2 obs < 5 -> all NaN
    assert np.isnan(R).all()


def test_bonf_topk_keeps_topk_per_target_no_floor():
    nan = np.nan
    R = np.array([[nan, 0.9, 0.2],
                  [0.5, nan, 0.8],
                  [0.3, 0.1, nan]])
    npair = np.full((3, 3), 100.0)
    sig, kept = E._bonf_topk(R, npair, n=3, top_k=1, alpha=None)   # alpha=None -> floor off
    assert kept[1, 0] and kept[0, 1] and kept[1, 2]        # strongest source per target column
    assert kept.sum() == 3 and not kept[0, 0]              # top_k=1 per column, diagonal never kept


def test_bonf_topk_floor_prunes_weak():
    nan = np.nan
    R = np.array([[nan, 0.9], [0.01, nan]])                # source0->t1 strong, source1->t0 ~noise
    npair = np.full((2, 2), 100.0)
    sig, kept = E._bonf_topk(R, npair, n=2, top_k=5, alpha=0.05)
    assert kept[0, 1] and not kept[1, 0]                   # weak 0.01 fails the Bonferroni floor


def test_bh_fdr_count_q1_keeps_all_finite():
    nan = np.nan
    R = np.array([[nan, 0.9, 0.8],
                  [0.7, nan, 0.6],
                  [0.5, 0.4, nan]])
    npair = np.full((3, 3), 100.0)
    # q=1.0 -> every finite off-diagonal pair passes -> 2 per column x 3 columns
    assert E._bh_fdr_count(R, npair, n=3, q=1.0) == 6


def test_bh_fdr_count_skips_all_nan_column():
    nan = np.nan
    R = np.array([[nan, 0.9, nan],
                  [0.7, nan, nan],
                  [0.5, 0.4, nan]])                       # column 2 all-NaN -> m==0 -> skipped
    npair = np.full((3, 3), 100.0)
    assert E._bh_fdr_count(R, npair, n=3, q=1.0) == 4     # 2 (col0) + 2 (col1) + 0 (col2)


def test_bh_fdr_count_weak_corr_keeps_none():
    nan = np.nan
    R = np.array([[nan, 0.1], [0.1, nan]])                # weak r -> p~0.32 each
    npair = np.full((2, 2), 100.0)
    assert E._bh_fdr_count(R, npair, n=2, q=0.05) == 0    # none clears q=0.05 (below-empty branch)


def test_plots_module_imports():
    import volga_edge_sparsity_plots as P                 # covers module-level imports/constants
    assert P.OUT_HTML.name.endswith(".html") and P.JSON.name.endswith(".json")
