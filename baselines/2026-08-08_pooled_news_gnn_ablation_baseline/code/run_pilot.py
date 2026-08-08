"""Validation-only P0-P3 pooled screening runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader, Dataset


_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
_SMOKE_ROWS_PER_SPLIT = 128
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import (  # noqa: E402
    PooledManifest,
    PooledSample,
    NewsPanel,
    SplitFrames,
    attach_news,
    build_pooled_manifest,
    load_and_split_price_data,
)
from scaling import PreprocessorStore, TickerPreprocessor  # noqa: E402
from train import evaluate_records, run_training  # noqa: E402


class _ManifestDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, samples: Sequence[PooledSample], store: PreprocessorStore) -> None:
        self.samples = tuple(samples)
        self.store = store

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        ticker_id = sample.key.ticker_id
        target = sample.y_model_raw if sample.y_model_raw is not None else sample.y_raw
        y_norm = self.store.get(ticker_id).target_scaler.transform(np.asarray([target]))[0]
        return {
            "x_price": torch.tensor(sample.x_price_raw, dtype=torch.float32),
            "x_news": torch.tensor(sample.x_news, dtype=torch.float32),
            "news_mask": torch.tensor(sample.news_mask, dtype=torch.bool),
            "ticker_id": torch.tensor(ticker_id, dtype=torch.long),
            "y_norm": torch.tensor(y_norm, dtype=torch.float32),
            "y_raw": torch.tensor(
                sample.y_eval_raw if sample.y_eval_raw is not None else sample.y_raw,
                dtype=torch.float32,
            ),
            "target_date": sample.key.target_date,
        }


@dataclass(frozen=True)
class ScreeningInputs:
    manifest: PooledManifest
    store: PreprocessorStore
    loaders: Mapping[str, DataLoader]
    smoke_filter: Mapping[str, Any]


def run_har_reference(manifest: PooledManifest, store: PreprocessorStore, output_dir: Path | str) -> Path:
    """Fit one pooled HAR regression and evaluate only its exact validation samples."""

    train = manifest.samples["train"]
    validation = manifest.samples["val"]
    if not train or not validation:
        raise ValueError("P0 requires non-empty train and validation manifest samples")
    x_train = np.asarray([sample.x_price_raw[-1, -3:] for sample in train], dtype=float)
    ids = np.asarray([sample.key.ticker_id for sample in train], dtype=np.int64)
    targets = np.asarray([
        sample.y_model_raw if sample.y_model_raw is not None else sample.y_raw for sample in train
    ], dtype=float)
    y_train = np.asarray([
        store.get(int(ticker_id)).target_scaler.transform(np.asarray([target]))[0]
        for ticker_id, target in zip(ids, targets, strict=True)
    ])
    model = LinearRegression().fit(x_train, y_train)
    x_val = np.asarray([sample.x_price_raw[-1, -3:] for sample in validation], dtype=float)
    predictions = model.predict(x_val)
    records = [
        {
            "ticker_id": sample.key.ticker_id,
            "target_date": sample.key.target_date,
            "prediction_norm": float(prediction),
            "target_raw": float(sample.y_eval_raw if sample.y_eval_raw is not None else sample.y_raw),
        }
        for sample, prediction in zip(validation, predictions, strict=True)
    ]
    evaluation = evaluate_records(records, store)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = out / "results.json"
    _write_json(result, {
        "config_name": "P0",
        "manifest_hash": manifest.content_hash("val"),
        "ordered_validation_keys": _ordered_keys(validation),
        "targets_raw": evaluation["targets_raw"],
        "predictions_raw": evaluation["predictions_raw"],
        "validation_metrics": evaluation["metrics"],
        "per_ticker": evaluation["per_ticker"],
    })
    return result


def assert_shared_manifest(manifests: Mapping[str, PooledManifest]) -> PooledManifest:
    """Reject any P0-P3 content or ordering difference before model work starts."""

    required = tuple(sorted(manifests))
    if not required:
        raise ValueError("at least one manifest is required")
    reference = manifests[required[0]]
    expected = _manifest_identity(reference)
    for name, manifest in manifests.items():
        if _manifest_identity(manifest) != expected:
            raise ValueError(f"P0-P3 manifest mismatch: {name} differs from {required[0]}")
    return reference


def run_pooled_screening(args: argparse.Namespace) -> Path:
    """Build one filtered manifest, run selected P0-P3 configurations, and compare validation."""

    if args.epochs < 1 or args.epochs > 10:
        raise ValueError("screening epochs must be between 1 and 10")
    inputs = build_screening_inputs(args.smoke, args.max_tickers)
    manifest = assert_shared_manifest({name: inputs.manifest for name in _phase_configs(args.phase)})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "screening_metadata.json", {
        "phase": args.phase, "epochs": args.epochs, "seed": args.seed,
        "smoke_filter": dict(inputs.smoke_filter),
        "manifest_hashes": {split: manifest.content_hash(split) for split in ("train", "val", "test")},
    })
    results: dict[str, dict[str, Any]] = {}
    run_dirs: dict[str, Path] = {}
    for config_name in _phase_configs(args.phase):
        config_dir = out / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        run_dirs[config_name] = config_dir
        assert_loaders_match_manifest(inputs.loaders, manifest)
        _write_json(config_dir / "screening_manifest.json", _screening_manifest_payload(manifest))
        if config_name == "P0":
            result_path = run_har_reference(manifest, inputs.store, config_dir)
        else:
            state_path = config_dir / "training_state.pt"
            result_path = run_training(
                config_name, inputs.loaders, inputs.store, config_dir, args.epochs, args.seed,
                resume_from=state_path if state_path.exists() else None,
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("manifest_hash") not in (None, manifest.content_hash("val")):
            raise ValueError(f"P0-P3 manifest mismatch: {config_name} validation hash differs")
        results[config_name] = payload
    comparison = comparison_payload(results, args.epochs, run_dirs)
    _write_json(out / "validation_comparison.json", comparison)
    _write_comparison_csv(out / "validation_comparison.csv", comparison)
    return out / "validation_comparison.json"


def build_screening_inputs(smoke: bool, max_tickers: int | None) -> ScreeningInputs:
    """Apply ticker filtering before fitting preprocessing or constructing the shared manifest."""

    if max_tickers is not None and max_tickers < 1:
        raise ValueError("max_tickers must be positive")
    requested = 3 if smoke and max_tickers is None else max_tickers
    raw_splits = load_and_split_price_data(_ROOT / "data" / "processed")
    selected = sorted(raw_splits.ticker_to_id)[:requested] if requested else sorted(raw_splits.ticker_to_id)
    splits = _select_tickers(raw_splits, selected)
    if smoke:
        splits = limit_smoke_rows(splits, _SMOKE_ROWS_PER_SPLIT)
    preprocessors = {
        ticker_id: TickerPreprocessor.fit(
            splits.frames[ticker]["train"], ["parkinson_volatility"], "parkinson_volatility"
        )
        for ticker, ticker_id in splits.ticker_to_id.items()
    }
    store = PreprocessorStore(preprocessors)
    manifest = build_pooled_manifest(splits, store)
    panel = load_runner_news_panel(_ROOT / "data" / "features" / "dual_group_news_panel.parquet", selected)
    attached = {
        split: tuple(attach_news(manifest.samples[split], panel, panel.feature_cols))
        for split in ("train", "val", "test")
    }
    manifest = PooledManifest(attached, manifest.exclusions, manifest.ticker_to_id, manifest.preprocessing_hash)
    loaders = {
        split: DataLoader(_ManifestDataset(manifest.samples[split], store), batch_size=64, shuffle=False)
        for split in ("train", "val")
    }
    return ScreeningInputs(manifest, store, loaders, {
        "enabled": smoke, "max_tickers": requested, "selected_tickers": selected,
        "rows_per_split": _SMOKE_ROWS_PER_SPLIT if smoke else None,
    })


def load_runner_news_panel(path: Path | str, tickers: Sequence[str]) -> NewsPanel:
    """Load sparse cached news features, treating absent source-group cells as zero features."""

    frame = pd.read_parquet(path, filters=[("ticker", "in", list(tickers))])
    required = {"ticker", "date"}
    if not required.issubset(frame.columns):
        raise ValueError("news panel must contain ticker and date")
    feature_cols = tuple(column for column in frame.columns if column not in required)
    if not feature_cols or frame.duplicated(["ticker", "date"]).any():
        raise ValueError("news panel must have unique rows and feature columns")
    numeric = frame.loc[:, feature_cols].apply(pd.to_numeric, errors="raise").fillna(0.0)
    values = {
        (str(row.ticker), str(pd.Timestamp(row.date).date())): numeric.iloc[offset].to_numpy(dtype=np.float32)
        for offset, (_, row) in enumerate(frame.loc[:, ["ticker", "date"]].iterrows())
    }
    return NewsPanel(values, feature_cols, {"source": str(path), "sparse_cells_filled_with_zero": True})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pooled", "P0", "P1", "P2", "P3"), default="pooled")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "results" / "pooled_news_gnn_pilot")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-tickers", type=int)
    return parser.parse_args(argv)


def _select_tickers(raw: SplitFrames, selected: Sequence[str]) -> SplitFrames:
    selected_set = set(selected)
    if not selected_set:
        raise ValueError("ticker filtering left no tickers")
    frames = {ticker: raw.frames[ticker] for ticker in sorted(selected_set)}
    return SplitFrames(frames, {ticker: index for index, ticker in enumerate(sorted(selected_set))})


def limit_smoke_rows(splits: SplitFrames, rows_per_split: int) -> SplitFrames:
    """Bound smoke data after chronological splitting and before any manifest work."""

    if rows_per_split < 1:
        raise ValueError("rows_per_split must be positive")
    return SplitFrames(
        {
            ticker: {name: frame.iloc[:rows_per_split].copy() for name, frame in frames.items()}
            for ticker, frames in splits.frames.items()
        },
        dict(splits.ticker_to_id),
    )


def _phase_configs(phase: str) -> tuple[str, ...]:
    return ("P0", "P1", "P2", "P3") if phase == "pooled" else (phase,)


def assert_loaders_match_manifest(loaders: Mapping[str, DataLoader], manifest: PooledManifest) -> None:
    """Reject altered or reordered P1-P3 loader samples before each configuration runs."""

    for split in ("train", "val"):
        dataset_samples = getattr(getattr(loaders[split], "dataset", None), "samples", None)
        if dataset_samples is None:
            raise ValueError("P0-P3 manifest mismatch: loader must expose manifest samples")
        expected = manifest.samples[split]
        if _sample_identity(dataset_samples) != _sample_identity(expected):
            raise ValueError(f"P0-P3 manifest mismatch: {split} loader differs")


def _screening_manifest_payload(manifest: PooledManifest) -> dict[str, Any]:
    return {
        "hashes": {split: manifest.content_hash(split) for split in ("train", "val", "test")},
        "ordered_keys": {split: _ordered_keys(manifest.samples[split]) for split in ("train", "val", "test")},
    }


def _manifest_identity(manifest: PooledManifest) -> tuple[tuple[str, tuple[str, ...], str], ...]:
    return tuple(
        (split, tuple(_ordered_keys(manifest.samples[split])), manifest.content_hash(split))
        for split in ("train", "val", "test")
    )


def _ordered_keys(samples: Sequence[PooledSample]) -> list[str]:
    return [f"{sample.key.ticker_id}:{sample.key.target_date}" for sample in samples]


def _sample_identity(samples: Sequence[PooledSample]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            sample.key.ticker_id, sample.key.ticker, sample.key.target_date,
            sample.x_price_raw.tobytes(), sample.x_news.tobytes(), sample.news_mask.tobytes(),
            sample.y_raw, sample.y_model_raw, sample.y_eval_raw, sample.input_dates,
        )
        for sample in samples
    )


def comparison_payload(
    results: Mapping[str, Mapping[str, Any]], epochs: int, run_dirs: Mapping[str, Path]
) -> dict[str, Any]:
    rows = []
    for config_name, result in results.items():
        metrics = result["validation_metrics"]
        if not all(np.isfinite(float(value)) for value in metrics.values()):
            raise ValueError(f"{config_name} has non-finite validation metrics")
        rows.append({"config_name": config_name, **{name: float(value) for name, value in metrics.items()}})
    p1 = results.get("P1", {}).get("validation_metrics")
    promotions = {}
    for name in ("P2", "P3"):
        candidate = results.get(name, {}).get("validation_metrics")
        promotions[name] = bool(
            p1 and candidate and epochs >= 5
            and _has_valid_epoch_five_curve(results[name], run_dirs[name])
            and candidate["qlike"] < p1["qlike"]
            and candidate["rmse"] <= p1["rmse"] * 1.01
            and candidate["directional_accuracy"] >= p1["directional_accuracy"] - 1.0
        )
    return {"scope": "validation_only", "rows": rows, "promotion_eligible": promotions}


def _has_valid_epoch_five_curve(result: Mapping[str, Any], run_dir: Path) -> bool:
    losses = result.get("validation_losses", [])
    if len(losses) < 5 or not np.isfinite(np.asarray(losses[:5], dtype=float)).all():
        return False
    return (run_dir / "learning_curve_epoch_5.png").exists() and float(losses[4]) <= float(losses[0])


def _write_comparison_csv(path: Path, comparison: Mapping[str, Any]) -> None:
    rows = comparison["rows"]
    fields = ["config_name", "mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    run_pooled_screening(parse_args())
