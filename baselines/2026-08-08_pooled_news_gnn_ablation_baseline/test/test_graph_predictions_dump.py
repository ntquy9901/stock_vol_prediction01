"""The graph runner must persist aligned per-observation predictions for DM testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_pilot  # noqa: E402


def test_write_graph_predictions_aligns_records_with_raw_series(tmp_path: Path) -> None:
    """predictions.json rows pair each present record's id/date with its raw target+prediction.

    The runner already computes ``evaluation['predictions_raw']`` / ``['targets_raw']`` in the
    same order as ``records`` (present validation nodes). Persisting that alignment lets a
    Diebold-Mariano test compute a per-observation loss differential between G0 and G1 on the
    identical (date, node) evaluation set.
    """
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "prediction_norm": 0.5, "target_raw": 10.0},
        {"ticker_id": 1, "target_date": "2020-01-02", "prediction_norm": 1.0, "target_raw": 102.0},
    ]
    evaluation = {"predictions_raw": [11.0, 100.5], "targets_raw": [10.0, 102.0]}
    path = tmp_path / "predictions.json"

    run_pilot._write_graph_predictions(records, evaluation, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == [
        {"ticker_id": 0, "target_date": "2020-01-01", "target_raw": 10.0, "prediction_raw": 11.0},
        {"ticker_id": 1, "target_date": "2020-01-02", "target_raw": 102.0, "prediction_raw": 100.5},
    ]
