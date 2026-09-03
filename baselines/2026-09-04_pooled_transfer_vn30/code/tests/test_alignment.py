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
from wf_folds import assert_no_leakage, make_folds  # noqa: E402


def test_arms_share_identical_vn30_oos_keys_and_no_leakage():
    files = sorted(f for f in glob.glob(enriched_glob("vn100")) if "_rejections" not in f)
    if not files:
        pytest.skip("enriched vn100 absent")  # pragma: no cover
    keep = [Path(f).stem for f in files][:10]
    panel = build_enriched_panel(files, 22, 1, keep)
    wf = VolgaWFConfig(lookback=22, horizon=1, folds_target=2)
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, (n - ts) // 2)
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    cfg = training_config(epochs=2, seeds=(0,))
    score_idx = np.array([0, 1, 2])
    a1 = ra.run_arm(panel, folds, wf, cfg, np.arange(panel.N), score_idx)
    a0 = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)
    assert set(a1["preds"]["HAR"]) == set(a0["preds"]["HAR"])
