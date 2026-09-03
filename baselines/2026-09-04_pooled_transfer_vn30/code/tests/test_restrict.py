import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp  # noqa: E402
import masked_rich as MR  # noqa: E402


def _toy_D():
    n = 4
    ones = np.ones((2, n), np.float32)
    adj = np.ones((n, n), np.float32)
    return MR.MaskedRichData(
        tickers=["a", "b", "c", "d"], adj_vol2pk=adj, adj_corr=np.eye(n, dtype=np.float32),
        X_tr=np.zeros((2, n, 1, 5), np.float32), X_va=np.zeros((2, n, 1, 5), np.float32),
        X_te=np.zeros((2, n, 1, 5), np.float32),
        nmask_tr=ones.copy(), nmask_va=ones.copy(), nmask_te=ones.copy(),
        tmask_tr=ones.copy(), tmask_va=ones.copy(), tmask_te=ones.copy(),
        y_tr=np.ones((2, n)), y_va=np.ones((2, n)), y_te=np.ones((2, n)),
        har_tr=np.zeros((2, n, 3)), har_va=np.zeros((2, n, 3)), har_te=np.zeros((2, n, 3)),
        d_va=["2020-01-01", "2020-01-02"], d_te=["2020-01-01", "2020-01-02"],
        t_mean=np.ones(n), t_std=np.ones(n),
        har5_tr=np.zeros((2, n, 5)), har5_va=np.zeros((2, n, 5)), har5_te=np.zeros((2, n, 5)))


def test_restrict_zeros_train_outside_idx_and_isolates_graph():
    D = _toy_D()
    D2 = pp.restrict_fold(D, np.array([0, 1]))
    assert D2.tmask_tr[:, 2:].sum() == 0 and D2.tmask_tr[:, :2].sum() == 4
    assert D2.adj_vol2pk[2:, :].sum() == 0 and D2.adj_vol2pk[:, 2:].sum() == 0
    assert D2.adj_vol2pk[:2, :2].sum() == 4
    assert D.tmask_tr[:, 2:].sum() == 4          # original untouched (copy, not in-place)
    assert D2.tmask_te.sum() == D.tmask_te.sum()  # test mask unchanged
