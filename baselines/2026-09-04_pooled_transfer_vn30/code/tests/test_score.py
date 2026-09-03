import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp  # noqa: E402
from run_masked_rich import _pred_dict  # noqa: E402


def test_score_mask_keeps_only_score_nodes():
    tm = np.ones((2, 4), np.float32)
    sm = pp.score_mask(tm, np.array([1, 3]))
    assert sm[:, [0, 2]].sum() == 0 and sm[:, [1, 3]].sum() == 4
    d = _pred_dict(np.zeros((2, 4)), np.ones((2, 4)), sm, ["2020-01-01", "2020-01-02"], 4)
    assert {j for (j, _) in d} == {1, 3}
