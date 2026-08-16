"""Build per-observation QLIKE/squared-error loss series for a G0-vs-G1 DM test.

Loads the aligned per-observation prediction dumps written by ``_write_graph_predictions``
and returns the per-observation loss arrays for each config on the identical (date, node)
evaluation set. QLIKE matches ``src.common.evaluation.qlike_loss`` per observation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_EPSILON = 1e-8


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load a ``predictions.json`` dump written by the graph runner."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _qlike(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.maximum(target, _EPSILON)
    prediction = np.maximum(prediction, _EPSILON)
    ratio = target / prediction
    return ratio - np.log(ratio) - 1.0


def paired_losses(
    g0_rows: Sequence[Mapping[str, Any]], g1_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per-observation QLIKE and squared-error losses for G0 and G1, aligned by (id, date).

    G0 and G1 evaluate the identical validation snapshots in identical order, so the dumps
    must share the same (ticker_id, target_date) sequence and the same raw targets; both are
    asserted before computing losses.
    """
    if len(g0_rows) != len(g1_rows):
        raise ValueError("G0 and G1 prediction dumps are not aligned (different lengths)")
    keys0 = [(row["ticker_id"], row["target_date"]) for row in g0_rows]
    keys1 = [(row["ticker_id"], row["target_date"]) for row in g1_rows]
    if keys0 != keys1:
        raise ValueError("G0 and G1 prediction dumps are not aligned (different (id,date) order)")

    target = np.asarray([row["target_raw"] for row in g0_rows], dtype=float)
    target_g1 = np.asarray([row["target_raw"] for row in g1_rows], dtype=float)
    if not np.array_equal(target, target_g1):
        raise ValueError("G0 and G1 carry different raw targets for the same observation")

    pred_g0 = np.asarray([row["prediction_raw"] for row in g0_rows], dtype=float)
    pred_g1 = np.asarray([row["prediction_raw"] for row in g1_rows], dtype=float)
    return {
        "qlike_g0": _qlike(target, pred_g0),
        "qlike_g1": _qlike(target, pred_g1),
        "se_g0": (target - pred_g0) ** 2,
        "se_g1": (target - pred_g1) ** 2,
        "n": len(g0_rows),
    }
