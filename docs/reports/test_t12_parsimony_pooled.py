"""Tests for the T1.2 pooled-regime parsimony aggregation script.

The analysis script is dated (leading digit) so it is loaded by path via importlib.
Covers the statistical helpers (hand-computed) and an integration test that the
report builder runs against the real committed pooled results.json files.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_SCRIPT = Path(__file__).with_name("2026-08-08_t12_parsimony_pooled.py")
_spec = importlib.util.spec_from_file_location("t12_parsimony_pooled", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mod)


def test_mean_std_sample_std():
    mean, std = mod.mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert math.isclose(std, 1.0)  # sample std, ddof=1


def test_mean_std_single_value_zero_std():
    mean, std = mod.mean_std([5.0])
    assert mean == 5.0
    assert std == 0.0


def test_paired_t_hand_computed():
    # a - b = [2, 4, 6]: mean 4, sample sd 2, se = 2/sqrt(3), t = 4/se
    md, t = mod.paired_t([3.0, 5.0, 7.0], [1.0, 1.0, 1.0])
    assert math.isclose(md, 4.0)
    assert math.isclose(t, 4.0 / (2.0 / math.sqrt(3)), rel_tol=1e-9)


def test_paired_t_zero_diff_is_nan():
    md, t = mod.paired_t([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert md == 0.0
    assert math.isnan(t)


def test_sign_consistency_counts_direction():
    # 3 improvements (all negative diffs) for a lower-is-better metric
    n_improve, n_total = mod.sign_consistency([-0.1, -0.2, -0.05], lower_is_better=True)
    assert (n_improve, n_total) == (3, 3)
    # mixed: two better one worse for a higher-is-better metric
    n_improve, n_total = mod.sign_consistency([0.1, 0.2, -0.05], lower_is_better=False)
    assert (n_improve, n_total) == (2, 3)


def test_build_report_integration_real_json():
    report = mod.build_report()
    # all four cells, each with three seeds present
    for cfg in ("P0", "P1", "P2", "P3"):
        assert len(report["cells"][cfg]["qlike"]) == 3, cfg
    # news effect (P2 vs P1) improves QLIKE in every seed (P2 lower)
    news_qlike = report["contrasts"]["news_P2_vs_P1"]["qlike"]
    assert news_qlike["n_improve"] == 3
    assert news_qlike["mean_diff"] < 0
    # gate effect (P3 vs P2) is negligible: |mean qlike diff| tiny
    gate_qlike = report["contrasts"]["gate_P3_vs_P2"]["qlike"]
    assert abs(gate_qlike["mean_diff"]) < 1e-3
