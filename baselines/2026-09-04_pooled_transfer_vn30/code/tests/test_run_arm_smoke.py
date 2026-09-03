import glob
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp  # noqa: E402,F401  (path bootstrap)
import run_pooled_arm as ra  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_enriched_panel import build_enriched_panel  # noqa: E402
from wf_folds import make_folds  # noqa: E402


def _slice_panel():
    files = sorted(f for f in glob.glob(enriched_glob("vn100")) if "_rejections" not in f)
    if not files:
        pytest.skip("enriched vn100 absent")  # pragma: no cover
    keep = [Path(f).stem for f in files][:12]
    return build_enriched_panel(files, 22, 1, keep)


def test_run_arm_scores_only_score_idx_and_runs():
    panel = _slice_panel()
    wf = VolgaWFConfig(lookback=22, horizon=1, folds_target=1)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, n - ts)
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    cfg = training_config(epochs=2, seeds=(0,))
    score_idx = np.array([0, 1, 2])
    out = ra.run_arm(panel, folds, wf, cfg, np.arange(panel.N), score_idx)
    scored_nodes = {j for (j, _) in out["preds"]["HAR"]}
    assert scored_nodes.issubset({0, 1, 2})
    assert set(out["metrics"]) == {"HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk"}
    assert out["metrics"]["HAR"]["n"] > 0
