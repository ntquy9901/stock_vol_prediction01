"""Value-freeze test for the single canonical pipeline config.

Pins EVERY centralized constant to its delivered value so a future edit that changes a delivered
number FAILS loudly. The ONE intentional change is ``VOLUME_ZSCORE_WINDOW = 22`` (was 20; the
delivered paper JSONs used 20 -- reproduce with 20). Also asserts the ``Config`` / ``data_utils``
views re-export the SAME value they source from ``pipeline_config`` (no drift between view and SoT).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SUB = Path(__file__).resolve().parents[1]
if str(_SUB) not in sys.path:
    sys.path.insert(0, str(_SUB))

import pipeline_config as pc  # noqa: E402
import config  # noqa: E402
import data_utils as du  # noqa: E402


# Frozen expected values (delivered, byte-for-byte) -- volume window intentionally 22.
FROZEN = {
    # training hyperparameters
    "LOOKBACK": 10, "EPOCHS": 20, "PATIENCE": 3, "MIN_EPOCHS": 5, "HIDDEN": 64, "HEADS": 4,
    "DROPOUT": 0.2, "LR": 1e-3, "WEIGHT_DECAY": 1e-5, "GRAD_CLIP": 1.0, "BATCH_SIZE": 512,
    # seeds + horizons
    "SEEDS": (42, 123, 2026, 7, 2024), "HORIZONS": (1, 5, 10, 22),
    # data / feature windows
    "FIRST_VALID": 21, "HAR_WEEKLY_WINDOW": 5, "HAR_MONTHLY_WINDOW": 22,
    "VOLUME_ZSCORE_WINDOW": 22,          # <-- intentional change 20 -> 22 (canonical monthly convention)
    "VOL_OF_VOL_WINDOW": 22,
    # split / drop thresholds
    "TRAIN_FRAC": 0.80, "VAL_FRAC": 0.10, "MIN_ROWS": 200, "MIN_ANCHORS": 60, "MIN_TRAIN": 30,
    "MIN_VAL": 5, "MIN_TEST": 5, "MIN_VALID_NODES": 8, "MIN_TRAIN_ROWS": 252, "MIN_COMMON_DATES": 300,
    # graph / edge
    "N_NODE_FEATURES": 5, "EDGE_TOP_K": 5, "EDGE_MIN_OVERLAP": 100, "EDGE_MIN_PAIRS_DIRECTED": 30,
    "MIN_VOL_COVERAGE": 0.5, "EMPTY_VOL_COVERAGE": 0.05,
    # floors / epsilons
    "QLIKE_FLOOR": 1e-8, "PRED_FLOOR_FRAC": 1e-3, "POS_FLOOR_FRAC": 1e-2, "POS_FLOOR_EPS": 1e-12,
    "SCALER_EPS": 1e-8, "RESIDUAL_EPS": 1e-8, "CROSSFIT_FOLDS": 5,
    # walk-forward
    "WF_RETRAIN_K": 66, "WF_VAL_TAIL": 66, "WF_TEST_FRAC": 0.90, "WF_HORIZON": 1,
}


def test_every_canonical_constant_is_frozen():
    """Each centralized constant equals its delivered value (volume window pinned to 22)."""
    for name, expected in FROZEN.items():
        actual = getattr(pc, name)
        assert actual == expected, f"pipeline_config.{name} drifted: {actual!r} != {expected!r}"


def test_no_undocumented_extra_constants():
    """The public constant set is exactly FROZEN -- a new public constant must be added to the freeze."""
    public = {n for n in dir(pc) if n.isupper() and not n.startswith("_")}
    assert public == set(FROZEN), f"public constant set changed: {public ^ set(FROZEN)}"


def test_volume_window_is_22_not_20():
    """Regression guard for THE intentional change: canonical volume window is 22 (was 20)."""
    assert pc.VOLUME_ZSCORE_WINDOW == 22


def test_config_view_sources_from_canonical():
    """Config field defaults are the SAME values as the canonical source (thin view, no drift)."""
    c = config.Config()
    assert c.lookback == pc.LOOKBACK
    assert c.epochs == pc.EPOCHS and c.patience == pc.PATIENCE and c.min_epochs == pc.MIN_EPOCHS
    assert c.hidden == pc.HIDDEN and c.heads == pc.HEADS and c.dropout == pc.DROPOUT
    assert c.lr == pc.LR and c.weight_decay == pc.WEIGHT_DECAY and c.grad_clip == pc.GRAD_CLIP
    assert c.top_k == pc.EDGE_TOP_K and c.qlike_floor == pc.QLIKE_FLOOR
    assert c.batch_size == pc.BATCH_SIZE and c.train_frac == pc.TRAIN_FRAC and c.val_frac == pc.VAL_FRAC
    assert c.seeds == pc.SEEDS
    assert c.horizons == (1, 5)   # submission-local vestigial default (unchanged)


def test_data_utils_view_sources_from_canonical():
    """data_utils module constants are the SAME values as the canonical source."""
    assert du.FIRST_VALID == pc.FIRST_VALID
    assert du.WEEKLY_WIN == pc.HAR_WEEKLY_WINDOW and du.MONTHLY_WIN == pc.HAR_MONTHLY_WINDOW
    assert du.MIN_ROWS == pc.MIN_ROWS and du.MIN_ANCHORS == pc.MIN_ANCHORS
    assert du.MIN_TRAIN == pc.MIN_TRAIN and du.MIN_VAL == pc.MIN_VAL and du.MIN_TEST == pc.MIN_TEST
    assert du._EPS == pc.SCALER_EPS
