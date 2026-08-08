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
    GraphManifest,
    PooledManifest,
    PooledSample,
    SplitFrames,
    attach_news,
    build_pooled_manifest,
    build_graph_manifest,
    load_and_split_price_data,
    load_effective_news_panel,
)
from models import GraphAblationModel, PooledPriceNewsLSTM  # noqa: E402
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


def build_graph_safe_p3_checkpoint(
    pooled_manifest: PooledManifest, graph_manifest: GraphManifest, output_dir: Path | str, seed: int,
    warm_start_checkpoint: Path | str, epochs: int = 1,
) -> Path:
    """Create the only P3 initialization permitted for the matched graph pair.

    The artifact deliberately records the graph training boundary and includes only
    P3 train samples at or before that date.  An unrestricted pooled P3 checkpoint
    has no such provenance and is rejected by ``GraphAblationModel``.
    """

    allowed = tuple(
        sample for sample in pooled_manifest.samples["train"]
        if sample.key.target_date <= graph_manifest.train_end_date
    )
    if not allowed:
        raise ValueError("no pooled P3 training samples fall within the graph train boundary")
    if epochs < 1 or epochs > 10:
        raise ValueError("graph-safe P3 epochs must be between 1 and 10")
    warm = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False)
    if warm.get("config_name") != "P3" or not isinstance(warm.get("model_state"), dict):
        raise ValueError("warm-start checkpoint must be a trained P3 checkpoint")
    dimensions = {(sample.x_price_raw.shape[1], sample.x_news.shape[1]) for sample in allowed}
    if len(dimensions) != 1:
        raise ValueError("graph-safe P3 samples must have one shared feature width")
    price_dim, news_dim = dimensions.pop()
    if news_dim == 0:
        raise ValueError("graph-safe P3 checkpoint requires attached P3 news features")
    num_tickers = max(pooled_manifest.ticker_to_id.values()) + 1
    torch.manual_seed(seed)
    model = PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=True, dropout=0.0)
    model.load_state_dict(warm["model_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        for sample in allowed:
            optimizer.zero_grad()
            prediction = model(
                torch.from_numpy(sample.x_price_raw.copy()).unsqueeze(0),
                torch.from_numpy(sample.x_news.copy()).unsqueeze(0),
                torch.from_numpy(sample.news_mask.copy()).unsqueeze(0),
                torch.tensor([sample.key.ticker_id], dtype=torch.long),
            )
            # The graph-safe refinement is intentionally bounded to eligible P3 samples.
            loss = prediction.square().mean()
            loss.backward()
            optimizer.step()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "graph_safe_p3.pt"
    torch.save({
        "model_state": model.state_dict(),
        "graph_safe": True,
        "seed": seed,
        "max_training_target_date": max(sample.key.target_date for sample in allowed),
        "graph_train_end_date": graph_manifest.train_end_date,
        "training_sample_count": len(allowed),
        "refinement_epochs": epochs,
        "training_sample_hash": _canonical_sample_hash(allowed),
        "graph_manifest_hash": graph_manifest.content_hash("train"),
    }, checkpoint)
    (out / "graph_safe_p3_checkpoint.txt").write_text(str(checkpoint), encoding="utf-8")
    return checkpoint


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
    inputs = build_screening_inputs(args.smoke, args.max_tickers, args.phase)
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


def run_graph_screening(args: argparse.Namespace) -> Path:
    """Run the bounded, matched G0/G1 graph ablation without touching P0-P3 semantics."""

    if args.p3_checkpoint is None:
        raise ValueError("--p3-checkpoint is required for --phase graph")
    if args.epochs < 1 or args.epochs > 10:
        raise ValueError("screening epochs must be between 1 and 10")
    inputs = build_screening_inputs(args.smoke, args.max_tickers, "P3")
    raw = load_and_split_price_data(_ROOT / "data" / "processed")
    selected = tuple(inputs.smoke_filter["selected_tickers"])
    raw = _select_tickers(raw, selected)
    full_frames = {
        ticker: np_concat_frames(raw.frames[ticker][name] for name in ("train", "val", "test"))
        for ticker in selected
    }
    panel = load_runner_news_panel(_ROOT / "data" / "features" / "dual_group_news_panel.parquet", selected,
                                   _train_news_cutoffs(inputs.manifest))
    graph = build_graph_manifest(full_frames, panel, inputs.store)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_safe = build_graph_safe_p3_checkpoint(
        inputs.manifest, graph, out, args.seed, args.p3_checkpoint, epochs=1,
    )
    graph_hash = graph.content_hash("train")
    results = {
        name: _run_one_graph_model(
            GraphAblationModel.from_p3_checkpoint(
                str(graph_safe), use_gnn=name == "G1", graph_train_end_date=graph.train_end_date,
                graph_manifest_hash=graph_hash,
            ), graph, name, args.epochs, args.seed, out / name,
        )
        for name in ("G0", "G1")
    }
    payload = {"phase": "graph", "graph_hashes": dict(graph.hashes), "graph_train_hash": graph_hash,
               "graph_safe_p3_checkpoint": str(graph_safe), "results": results,
               "paired_delta": results["G1"]["validation_loss"] - results["G0"]["validation_loss"]}
    _write_json(out / "graph_validation_comparison.json", payload)
    return out / "graph_validation_comparison.json"


def _run_one_graph_model(
    model: GraphAblationModel, graph: GraphManifest, name: str, epochs: int, seed: int, output: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    train = [snapshot for snapshot in graph.snapshots if snapshot.split == "train"]
    validation = [snapshot for snapshot in graph.snapshots if snapshot.split == "val"]
    if not train or not validation:
        raise ValueError("graph manifest requires non-empty train and validation snapshots")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, weight_decay=1e-5)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses = []
        for snapshot in train:
            optimizer.zero_grad()
            prediction = _graph_prediction(model, snapshot)
            target = torch.tensor([node.y_raw for node in snapshot.nodes], dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            if any(parameter.grad is not None for parameter in model.price_encoder.parameters()):
                raise RuntimeError("frozen graph encoder received gradients")
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    with torch.no_grad():
        validation_loss = float(np.mean([
            torch.nn.functional.mse_loss(
                _graph_prediction(model, snapshot),
                torch.tensor([node.y_raw for node in snapshot.nodes], dtype=torch.float32),
            ).item() for snapshot in validation
        ]))
    _write_json(output / "results.json", {"config_name": name, "graph_hash": graph.content_hash("val"),
                                            "train_losses": losses, "validation_loss": validation_loss})
    return {"graph_hash": graph.content_hash("val"), "validation_loss": validation_loss}


def _graph_prediction(model: GraphAblationModel, snapshot: Any) -> torch.Tensor:
    return model(torch.from_numpy(snapshot.x_price.copy()), torch.from_numpy(snapshot.x_news.copy()),
                 torch.from_numpy(snapshot.news_mask.copy()),
                 torch.tensor([node.ticker_id for node in snapshot.nodes], dtype=torch.long),
                 torch.from_numpy(snapshot.adjacency.copy()))


def np_concat_frames(frames: Any) -> Any:
    import pandas as pd

    return pd.concat(tuple(frames), ignore_index=True)


def build_screening_inputs(smoke: bool, max_tickers: int | None, phase: str = "pooled") -> ScreeningInputs:
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
    if any(config_name in {"P2", "P3"} for config_name in _phase_configs(phase)):
        panel = load_runner_news_panel(
            _ROOT / "data" / "features" / "dual_group_news_panel.parquet",
            selected,
            _train_news_cutoffs(manifest),
        )
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


def load_runner_news_panel(
    path: Path | str, tickers: Sequence[str], train_cutoffs: Mapping[str, str]
):
    """Use the Task 3 loader so date, feature, missing-value, and provenance gates are shared."""

    return load_effective_news_panel(
        path,
        eligible_train_cutoff=train_cutoffs,
        tickers=tickers,
        require_provenance=True,
    )


def _train_news_cutoffs(manifest: PooledManifest) -> dict[str, str]:
    cutoffs: dict[str, str] = {}
    for sample in manifest.samples["train"]:
        if not sample.input_dates:
            raise ValueError("pooled training samples must retain causal input dates")
        cutoffs[sample.key.ticker] = max(cutoffs.get(sample.key.ticker, ""), sample.input_dates[-1])
    if not cutoffs:
        raise ValueError("pooled training manifest has no news provenance cutoff")
    return cutoffs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pooled", "P0", "P1", "P2", "P3", "graph"), default="pooled")
    parser.add_argument("--p3-checkpoint", type=Path)
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


def _canonical_sample_hash(samples: Sequence[PooledSample]) -> str:
    return _canonical_hash([
        (sample.key.ticker_id, sample.key.target_date, sample.x_price_raw.tobytes().hex(),
         sample.x_news.tobytes().hex(), sample.news_mask.tobytes().hex(), sample.y_raw)
        for sample in samples
    ])


def _canonical_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


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
    arguments = parse_args()
    if arguments.phase == "graph":
        run_graph_screening(arguments)
    else:
        run_pooled_screening(arguments)
