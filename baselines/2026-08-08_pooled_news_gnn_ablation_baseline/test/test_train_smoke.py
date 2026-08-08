"""Raw-scale evaluation and one-batch pooled trainer contracts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402
import train as train_module  # noqa: E402
from train import evaluate_records, run_training  # noqa: E402
from data import PooledManifest, PooledSample, SampleKey, SplitFrames  # noqa: E402
import run_pilot  # noqa: E402


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

    @classmethod
    def from_fixture(cls, path: Path) -> "_TinyDataset":
        return cls(json.loads(path.read_text(encoding="utf-8"))["records"])


def _fixture_loaders(path: Path) -> dict[str, DataLoader]:
    return {
        split: DataLoader(_TinyDataset.from_fixture(path), batch_size=2, shuffle=False)
        for split in ("train", "val")
    }


@pytest.mark.smoke
def test_one_epoch_runner_writes_finite_screening_artifacts(tmp_path: Path) -> None:
    fixture = tmp_path / "tiny_fixture.json"
    fixture.write_text(
        json.dumps({"records": [
            {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)).tolist(),
             "y_norm": 0.0, "y_raw": 10.0},
            {"ticker_id": 0, "target_date": "2020-01-02", "x_price": (np.ones((22, 1)) * 2).tolist(),
             "y_norm": 1.0, "y_raw": 12.0},
        ]}),
        encoding="utf-8",
    )
    loaders = _fixture_loaders(fixture)
    loaders["test"] = DataLoader(_TinyDataset.from_fixture(fixture), batch_size=2, shuffle=False)

    result_path = run_training("P1", loaders, _store(), tmp_path / "run", epochs=1, seed=42)

    for name in ("results.json", "sample_manifest.json", "preprocessors.json", "best.pt", "learning_curve_partial.png"):
        assert (tmp_path / "run" / name).exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert "test_metrics" not in payload
    assert all(math.isfinite(value) for value in payload["validation_metrics"].values())


def test_resume_continues_history_and_optimizer_to_requested_epoch(tmp_path: Path) -> None:
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)), "y_norm": 0.0, "y_raw": 10.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "x_price": np.ones((22, 1)) * 2, "y_norm": 1.0, "y_raw": 12.0},
    ]
    loaders = {
        split: DataLoader(_TinyDataset(records), batch_size=2, shuffle=False)
        for split in ("train", "val")
    }
    run_dir = tmp_path / "run"
    run_training("P1", loaders, _store(), run_dir, epochs=1, seed=42)
    result_path = run_training(
        "P1", loaders, _store(), run_dir, epochs=2, seed=42, resume_from=run_dir / "training_state.pt"
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    state = torch.load(run_dir / "training_state.pt", weights_only=False)
    assert payload["epochs"] == state["completed_epoch"] == 2
    assert len(payload["train_losses"]) == len(state["train_losses"]) == 2
    assert state["optimizer_state"]["state"]


def test_resume_rejects_manifest_or_preprocessor_mismatch(tmp_path: Path) -> None:
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)), "y_norm": 0.0, "y_raw": 10.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "x_price": np.ones((22, 1)), "y_norm": 1.0, "y_raw": 12.0},
    ]
    loaders = {split: DataLoader(_TinyDataset(records), batch_size=2, shuffle=False) for split in ("train", "val")}
    run_dir = tmp_path / "run"
    run_training("P1", loaders, _store(), run_dir, epochs=1, seed=42)
    changed = list(records)
    changed[1] = {**changed[1], "target_date": "2020-01-03"}
    changed_loaders = {
        split: DataLoader(_TinyDataset(changed), batch_size=2, shuffle=False)
        for split in ("train", "val")
    }

    with pytest.raises(ValueError, match="manifest hash"):
        run_training(
            "P1", changed_loaders, _store(), run_dir, epochs=2, seed=42, resume_from=run_dir / "training_state.pt"
        )


def test_resume_matches_uninterrupted_dropout_training(tmp_path: Path) -> None:
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)), "y_norm": 0.0, "y_raw": 10.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "x_price": np.ones((22, 1)) * 2, "y_norm": 1.0, "y_raw": 12.0},
    ]
    loaders = {split: DataLoader(_TinyDataset(records), batch_size=2, shuffle=False) for split in ("train", "val")}
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    run_training("P1", loaders, _store(), uninterrupted, epochs=2, seed=42)
    run_training("P1", loaders, _store(), resumed, epochs=1, seed=42)
    run_training("P1", loaders, _store(), resumed, epochs=2, seed=42, resume_from=resumed / "training_state.pt")

    expected = torch.load(uninterrupted / "training_state.pt", weights_only=False)
    actual = torch.load(resumed / "training_state.pt", weights_only=False)
    for key, value in expected["model_state"].items():
        torch.testing.assert_close(actual["model_state"][key], value)
    assert actual["train_losses"] == pytest.approx(expected["train_losses"])


def test_resume_rejects_tensor_change_without_overwriting_artifacts(tmp_path: Path) -> None:
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)), "y_norm": 0.0, "y_raw": 10.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "x_price": np.ones((22, 1)) * 2, "y_norm": 1.0, "y_raw": 12.0},
    ]
    loaders = {split: DataLoader(_TinyDataset(records), batch_size=2, shuffle=False) for split in ("train", "val")}
    run_dir = tmp_path / "run"
    run_training("P1", loaders, _store(), run_dir, epochs=1, seed=42)
    previous_manifest = (run_dir / "sample_manifest.json").read_bytes()
    changed = list(records)
    changed[0] = {**changed[0], "x_price": np.zeros((22, 1)), "y_raw": 11.0}
    changed_loaders = {
        split: DataLoader(_TinyDataset(changed), batch_size=2, shuffle=False)
        for split in ("train", "val")
    }

    with pytest.raises(ValueError, match="manifest hash"):
        run_training(
            "P1", changed_loaders, _store(), run_dir, epochs=2, seed=42,
            resume_from=run_dir / "training_state.pt",
        )
    assert (run_dir / "sample_manifest.json").read_bytes() == previous_manifest


def test_failure_keeps_initial_checkpoint_and_partial_curve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"ticker_id": 0, "target_date": "2020-01-01", "x_price": np.ones((22, 1)), "y_norm": 0.0, "y_raw": 10.0},
        {"ticker_id": 0, "target_date": "2020-01-02", "x_price": np.ones((22, 1)) * 2, "y_norm": 1.0, "y_raw": 12.0},
    ]
    loaders = {split: DataLoader(_TinyDataset(records), batch_size=2, shuffle=False) for split in ("train", "val")}
    monkeypatch.setattr(train_module, "_normalized_loss", lambda *args: (_ for _ in ()).throw(RuntimeError("stop")))

    with pytest.raises(RuntimeError, match="stop"):
        run_training("P1", loaders, _store(), tmp_path / "failed", epochs=1, seed=42)
    assert (tmp_path / "failed" / "training_state.pt").exists()
    assert (tmp_path / "failed" / "learning_curve_partial.png").exists()


def _har_manifest() -> PooledManifest:
    def sample(split: str, index: int) -> PooledSample:
        value = float(index + 1)
        return PooledSample(
            SampleKey(0, "AAA", f"2020-02-{index + 1:02d}"),
            np.tile([value, value + 1, value + 2], (22, 1)),
            np.zeros((22, 0)), np.zeros(22, dtype=np.int8), value + 10,
            value + 10, value + 10,
            tuple(f"2020-01-{day:02d}" for day in range(1, 23)),
        )

    return PooledManifest(
        {"train": tuple(sample("train", index) for index in range(4)),
         "val": tuple(sample("val", index) for index in range(4, 7)), "test": ()},
        {}, {"AAA": 0}, "preprocessing",
    )


def test_har_reference_uses_exact_manifest_targets(tmp_path: Path) -> None:
    manifest = _har_manifest()
    result = run_pilot.run_har_reference(manifest, _store(), tmp_path)

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == manifest.content_hash("val")
    assert payload["targets_raw"] == [15.0, 16.0, 17.0]
    assert all(math.isfinite(value) for value in payload["validation_metrics"].values())
    assert "test_metrics" not in payload


def test_runner_rejects_manifest_mismatch_before_training(tmp_path: Path) -> None:
    manifest = _har_manifest()
    changed = PooledManifest(
        {**manifest.samples, "val": manifest.samples["val"][:-1]},
        manifest.exclusions, manifest.ticker_to_id, manifest.preprocessing_hash,
    )

    with pytest.raises(ValueError, match="P0-P3 manifest mismatch"):
        run_pilot.assert_shared_manifest({"P0": manifest, "P1": changed})


def test_runner_rejects_loader_manifest_mismatch_before_training() -> None:
    manifest = _har_manifest()
    changed = PooledManifest(
        {**manifest.samples, "val": manifest.samples["val"][:-1]},
        manifest.exclusions, manifest.ticker_to_id, manifest.preprocessing_hash,
    )
    loaders = {
        split: DataLoader(run_pilot._ManifestDataset(changed.samples[split], _store()), batch_size=2, shuffle=False)
        for split in ("train", "val")
    }

    with pytest.raises(ValueError, match="P0-P3 manifest mismatch"):
        run_pilot.assert_loaders_match_manifest(loaders, manifest)


def test_cli_smoke_options_are_bounded_and_recorded(tmp_path: Path) -> None:
    args = run_pilot.parse_args([
        "--phase", "pooled", "--epochs", "1", "--seed", "42", "--output-dir", str(tmp_path),
        "--smoke", "--max-tickers", "2",
    ])

    assert args.smoke is True
    assert args.max_tickers == 2
    assert args.epochs == 1


def test_runner_news_loader_converts_sparse_panel_cells_to_zero_vectors(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"ticker": ["AAA"], "date": ["2020-01-01"], "f0": [np.nan], "f1": [2.0]}).to_parquet(
        panel_path, index=False
    )

    panel = run_pilot.load_runner_news_panel(panel_path, ["AAA"])

    np.testing.assert_allclose(panel.values[("AAA", "2020-01-01")], [0.0, 2.0])


def test_smoke_filter_limits_each_split_before_manifest_building() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4), "parkinson_volatility": [1.0] * 4})
    splits = SplitFrames({"AAA": {name: frame for name in ("train", "val", "test")}}, {"AAA": 0})

    limited = run_pilot.limit_smoke_rows(splits, 2)

    assert all(len(limited.frames["AAA"][name]) == 2 for name in ("train", "val", "test"))


def test_promotion_requires_finite_nondivergent_epoch_five_curve(tmp_path: Path) -> None:
    metrics = {"mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.0, "qlike": 1.0, "directional_accuracy": 50.0}
    candidate = {**metrics, "qlike": 0.9, "directional_accuracy": 50.0}
    results = {"P1": {"validation_metrics": metrics, "validation_losses": [1.0] * 5},
               "P2": {"validation_metrics": candidate, "validation_losses": [1.0, 2.0, 3.0, 4.0, 5.0]}}
    run_dirs = {name: tmp_path / name for name in results}
    for directory in run_dirs.values():
        directory.mkdir()
        (directory / "learning_curve_epoch_5.png").write_bytes(b"curve")

    comparison = run_pilot.comparison_payload(results, 5, run_dirs)

    assert comparison["promotion_eligible"]["P2"] is False
