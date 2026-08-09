"""Tests for the sweep analysis aggregation (DM alignment + across-seed paired-t)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
_PILOT = Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_CODE), str(_PILOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze  # noqa: E402


def _p0(keys, targets, preds):
    return {"test": {k: (t, p) for k, t, p in zip(keys, targets, preds)}}


def _write_preds(path, keys, targets, preds):
    rows = [{"ticker_id": k[0], "target_date": k[1], "target_raw": t, "prediction_raw": p}
            for k, t, p in zip(keys, targets, preds)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_aligned_losses_reject_target_mismatch(tmp_path):
    keys = [(0, "2022-01-01"), (0, "2022-01-02"), (1, "2022-01-01")]
    p0 = _p0(keys, [1e-4, 2e-4, 1.5e-4], [1.1e-4, 2.1e-4, 1.4e-4])["test"]
    cfg = {k: (t + 1.0, p) for k, (t, p) in p0.items()}  # corrupt targets
    with pytest.raises(ValueError):
        analyze._aligned_losses(p0, cfg, analyze._qlike_vec)


def test_variant_detects_significant_qlike_win(tmp_path):
    """A config that is uniformly closer to truth than P0 on every obs → all-negative delta, DM sig."""

    rng = np.random.default_rng(0)
    n = 600
    keys = [(i % 3, f"2022-{i:04d}") for i in range(n)]
    targets = rng.uniform(2e-3, 4e-3, size=n)
    # strictly-positive predictions (no epsilon flooring): P0 has a mild multiplicative error,
    # config is much tighter, so the QLIKE loss differential is well-behaved and DM-significant.
    p0_pred = targets * rng.uniform(1.05, 1.15, size=n)       # P0 biased ~+10%
    p0 = _p0(keys, targets, p0_pred)
    paths = {}
    for seed in (42, 123, 2026):
        cp = targets * rng.uniform(1.001, 1.01, size=n)       # config near-perfect
        path = tmp_path / f"seed{seed}" / "predictions_test.json"
        _write_preds(path, keys, targets, cp)
        paths[seed] = path
    summary = analyze._analyze_variant(paths, p0)
    assert summary["n_seeds"] == 3
    assert summary["paired_t_qlike"]["all_negative"] is True
    assert summary["beats_P0_qlike_dm"] is True


def test_variant_no_win_when_worse(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    keys = [(i % 3, f"2022-{i:04d}") for i in range(n)]
    targets = rng.uniform(1e-4, 3e-4, size=n)
    p0 = _p0(keys, targets, targets + rng.normal(0, 1e-5, size=n))  # P0 good
    paths = {}
    for seed in (42, 123, 2026):
        cp = targets + rng.normal(0, 6e-5, size=n)  # config worse
        path = tmp_path / f"seed{seed}" / "predictions_test.json"
        _write_preds(path, keys, targets, cp)
        paths[seed] = path
    summary = analyze._analyze_variant(paths, p0)
    assert summary["beats_P0_qlike_dm"] is False
