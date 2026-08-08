"""Raw-scale evaluation and one-batch pooled trainer contracts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402
from train import evaluate_records, run_training  # noqa: E402


def _store() -> PreprocessorStore:
    def preprocessor(mean: float) -> TickerPreprocessor:
        scaler = ArrayStandardizer(np.array([mean]), np.array([2.0]))
        return TickerPreprocessor(("parkinson_volatility",), "parkinson_volatility", 0.0, 1.0, scaler, scaler)

    return PreprocessorStore({0: preprocessor(10.0), 1: preprocessor(100.0)})


def test_metrics_use_raw_target_and_ticker_specific_inverse() -> None:
    result = evaluate_records([
        {"ticker_id": 0, "target_date": "2020-01-01", "prediction_norm": 0.5, "target_raw": 10.0},
        {"ticker_id": 1, "target_date": "2020-01-01", "prediction_norm": 0.5, "target_raw": 100.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "prediction_norm": 1.0, "target_raw": 12.0},
        {"ticker_id": 1, "target_date": "2020-01-02", "prediction_norm": 1.0, "target_raw": 102.0},
    ], _store())

    assert result["targets_raw"] == [10.0, 100.0, 12.0, 102.0]
    assert result["predictions_raw"] == [11.0, 101.0, 12.0, 102.0]
    assert set(result["metrics"]) >= {"mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"}


def test_directional_accuracy_never_crosses_tickers() -> None:
    result = evaluate_records([
        {"ticker_id": 0, "target_date": "2020-01-02", "prediction_norm": 1.0, "target_raw": 12.0},
        {"ticker_id": 1, "target_date": "2020-01-01", "prediction_norm": 0.0, "target_raw": 100.0},
        {"ticker_id": 0, "target_date": "2020-01-01", "prediction_norm": 0.0, "target_raw": 10.0},
        {"ticker_id": 1, "target_date": "2020-01-02", "prediction_norm": 1.0, "target_raw": 102.0},
    ], _store())

    assert result["metrics"]["directional_accuracy"] == pytest.approx(100.0)
    assert result["directional_accuracy_weighted"] == pytest.approx(100.0)
    assert result["eligible_ticker_count"] == 2


def test_nonpositive_prediction_rate_above_one_percent_fails() -> None:
    with pytest.raises(ValueError, match="nonpositive prediction rate"):
        evaluate_records([
            {"ticker_id": 0, "target_date": "2020-01-01", "prediction_norm": -10.0, "target_raw": 10.0},
            {"ticker_id": 0, "target_date": "2020-01-02", "prediction_norm": -10.0, "target_raw": 12.0},
        ], _store())


class _TinyDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.source_fixture = "tiny_fixture.json"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        return {
            "x_price": torch.tensor(record["x_price"], dtype=torch.float32),
            "x_news": torch.zeros(22, 0),
            "news_mask": torch.zeros(22, dtype=torch.bool),
            "ticker_id": torch.tensor(record["ticker_id"], dtype=torch.long),
            "y_norm": torch.tensor(record["y_norm"], dtype=torch.float32),
            "y_raw": torch.tensor(record["y_raw"], dtype=torch.float32),
            "target_date": str(record["target_date"]),
        }


@pytest.mark.smoke
def test_one_epoch_runner_writes_finite_screening_artifacts(tmp_path: Path) -> None:
    fixture = tmp_path / "tiny_fixture.json"
    fixture.write_text(json.dumps({"prices": [10.0, 12.0], "news": []}), encoding="utf-8")
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)), "y_norm": 0.0, "y_raw": 10.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "x_price": np.ones((22, 1)) * 2, "y_norm": 1.0, "y_raw": 12.0},
    ]
    loaders = {
        "train": DataLoader(_TinyDataset(records), batch_size=2, shuffle=False),
        "val": DataLoader(_TinyDataset(records), batch_size=2, shuffle=False),
        "test": DataLoader(_TinyDataset(records), batch_size=2, shuffle=False),
    }

    result_path = run_training("P1", loaders, _store(), tmp_path / "run", epochs=1, seed=42)

    for name in ("results.json", "sample_manifest.json", "preprocessors.json", "best.pt", "learning_curve_partial.png"):
        assert (tmp_path / "run" / name).exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert "test_metrics" not in payload
    assert all(math.isfinite(value) for value in payload["validation_metrics"].values())
