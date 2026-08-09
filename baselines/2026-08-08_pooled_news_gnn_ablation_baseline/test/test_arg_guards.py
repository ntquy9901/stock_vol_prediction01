"""Argument-validation guard tests for run_pilot entry points.

These cover the pure input-validation raise branches (no GPU/data needed) that
guard the pooled/graph screening runners and the shared manifest builder, so a
bad --horizon/--regime/--epochs/--batch-size is rejected before any training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import data as data_module  # noqa: E402
import run_pilot  # noqa: E402


def test_build_screening_inputs_rejects_nonpositive_max_tickers() -> None:
    with pytest.raises(ValueError, match="max_tickers must be positive"):
        run_pilot.build_screening_inputs(smoke=False, max_tickers=0)


def test_build_screening_inputs_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        run_pilot.build_screening_inputs(smoke=False, max_tickers=1, batch_size=0)


def test_build_screening_inputs_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon must be positive"):
        run_pilot.build_screening_inputs(smoke=False, max_tickers=1, horizon=0)


def test_build_screening_inputs_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="regime must be one of"):
        run_pilot.build_screening_inputs(smoke=False, max_tickers=1, regime="bogus")


def test_run_pooled_screening_rejects_epoch_range() -> None:
    args = argparse.Namespace(
        epochs=0, batch_size=256, smoke=True, max_tickers=3, phase="pooled",
        horizon=5, regime="pooled",
    )
    with pytest.raises(ValueError, match="between 1 and 20"):
        run_pilot.run_pooled_screening(args)


def test_run_graph_screening_rejects_epoch_range() -> None:
    args = argparse.Namespace(
        epochs=99, batch_size=256, smoke=True, max_tickers=3, horizon=5,
        seed=42, device="cpu", output_dir="unused",
        p3_checkpoint=None, graph_batch_size=64, graph_train_batch_size=64,
    )
    with pytest.raises(ValueError, match="between 1 and 10"):
        run_pilot.run_graph_screening(args)


def test_assert_matched_horizon_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="must equal graph horizon"):
        run_pilot._assert_matched_horizon(5, 22)


def test_assert_matched_horizon_accepts_match() -> None:
    # Equal horizons must not raise (the happy path).
    run_pilot._assert_matched_horizon(5, 5)


def test_build_graph_manifest_rejects_bad_seq_horizon() -> None:
    # data.py guard: seq_length/horizon must be positive (raised before any work).
    with pytest.raises(ValueError, match="must be positive"):
        data_module.build_graph_manifest({}, None, None, seq_length=0, horizon=5)


def test_build_graph_manifest_rejects_single_frame() -> None:
    # data.py guard: graph ablation needs >= 2 ticker frames.
    with pytest.raises(ValueError, match="at least two ticker frames"):
        data_module.build_graph_manifest(
            {"AAA": None}, None, None, seq_length=10, horizon=5
        )
