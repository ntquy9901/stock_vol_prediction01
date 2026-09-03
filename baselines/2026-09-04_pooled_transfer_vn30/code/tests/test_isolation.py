import glob
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp  # noqa: E402,F401
import run_pooled_arm as ra  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_enriched_panel import build_enriched_panel  # noqa: E402
from wf_folds import make_folds  # noqa: E402


def test_arm0_vn30_preds_independent_of_nonvn30_features():
    """Arm 0 (train==score==VN30) VN30 preds must be invariant to non-VN30 node feature values.

    Confirms the single-panel-mask genuinely reproduces a 31-node system (graph restricted + per-node
    LSTM). If this fails the net mixes nodes outside the adjacency -> fall back to a separate VN30 panel.
    """
    files = sorted(f for f in glob.glob(enriched_glob("vn100")) if "_rejections" not in f)
    if not files:
        pytest.skip("enriched vn100 absent")  # pragma: no cover
    keep = [Path(f).stem for f in files][:8]
    panel = build_enriched_panel(files, 22, 1, keep)
    wf = VolgaWFConfig(lookback=22, horizon=1, folds_target=1)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, n - ts)
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    cfg = training_config(epochs=2, seeds=(0,))
    score_idx = np.array([0, 1, 2])
    base = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)
    p1 = {k: v[1] for k, v in base["preds"]["LSTM_wGAT_vol2pk"].items()}
    panel.feats[:, 3:, :] += 5.0          # perturb ONLY non-VN30 nodes' features
    base2 = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)
    p2 = {k: v[1] for k, v in base2["preds"]["LSTM_wGAT_vol2pk"].items()}
    for k in p1:
        assert abs(p1[k] - p2[k]) < 1e-4, "Arm0 VN30 preds leaked from non-VN30 nodes"
