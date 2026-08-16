"""Validation-only P0-P3 pooled screening runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import os

import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader, Dataset


def _resolve_data_root() -> Path:
    """Locate the repo root that holds ``data/processed`` for train/infer.

    The reproducibility bundle can be dropped anywhere, so the original fixed
    ``parents[3]`` assumption no longer holds. Resolution order:
    1. ``TRACK_B_DATA_ROOT`` env var (explicit advisor override), then
    2. the nearest ancestor directory that actually contains ``data/processed``,
    3. falling back to the original ``parents[3]`` for backward compatibility.
    ``view`` never touches data, so a missing dataset only affects train/infer.
    """

    override = os.environ.get("TRACK_B_DATA_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "data" / "processed").is_dir():
            return candidate
    return here.parents[3]


_ROOT = _resolve_data_root()
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
    common_trading_dates,
    load_and_split_price_data,
    load_effective_news_panel,
    restrict_train_to_common_dates,
)
from models import GraphAblationModel, PooledPriceNewsLSTM  # noqa: E402
from scaling import PreprocessorStore, TickerPreprocessor  # noqa: E402
from train import evaluate_records, run_training  # noqa: E402


class _ManifestDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, samples: Sequence[PooledSample], store: PreprocessorStore) -> None:
        self.samples = tuple(samples)
        self.store = store
        # Compute the scalar transform once.  ``__getitem__`` is on the hot path
        # for every epoch; repeating sklearn-style scaling there was unnecessary
        # CPU work and did not change the causal preprocessing contract.
        self._y_norm = tuple(
            float(store.get(sample.key.ticker_id).target_scaler.transform(np.asarray([
                sample.y_model_raw if sample.y_model_raw is not None else sample.y_raw
            ]))[0])
            for sample in self.samples
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        ticker_id = sample.key.ticker_id
        y_norm = self._y_norm[index]
        return {
            # as_tensor/from_numpy avoids an extra per-sample copy.  DataLoader
            # performs the batch transfer; pinning is enabled at loader creation
            # for CUDA runs.
            "x_price": torch.as_tensor(sample.x_price_raw.copy(), dtype=torch.float32),
            "x_news": torch.as_tensor(sample.x_news.copy(), dtype=torch.float32),
            "news_mask": torch.as_tensor(sample.news_mask.copy(), dtype=torch.bool),
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


def _pooled_training_batches(
    samples: Sequence[PooledSample], store: PreprocessorStore, device: torch.device, batch_size: int,
):
    """Yield stacked ``[batch, time, feature]`` pooled tensors and per-ticker normalized targets.

    Grouping the previously per-sample warm-start / graph-safe training into mini-batches is a
    deliberate SGD re-baseline (approved R10 semantics): it reproduces the historical per-sample
    loop exactly at ``batch_size == 1`` and otherwise averages the gradient over the batch. Each
    sample keeps its own ticker's target scaler, so the leakage/scaler contract is unchanged.
    """

    non_blocking = device.type == "cuda"
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        x_price = torch.from_numpy(np.stack([sample.x_price_raw for sample in chunk])).to(
            device, non_blocking=non_blocking)
        x_news = torch.from_numpy(np.stack([sample.x_news for sample in chunk])).to(
            device, non_blocking=non_blocking)
        news_mask = torch.from_numpy(np.stack([sample.news_mask for sample in chunk])).to(
            device, non_blocking=non_blocking)
        ticker_ids = torch.tensor([sample.key.ticker_id for sample in chunk], dtype=torch.long,
                                  device=device)
        targets = torch.tensor([
            float(store.get(sample.key.ticker_id).target_scaler.transform(np.asarray([
                sample.y_model_raw if sample.y_model_raw is not None else sample.y_raw
            ]))[0])
            for sample in chunk
        ], dtype=torch.float32, device=device)
        yield x_price, x_news, news_mask, ticker_ids, targets


def build_graph_bound_p3_warm_start(
    pooled_manifest: PooledManifest, graph_manifest: GraphManifest, output_dir: Path | str,
    seed: int, store: PreprocessorStore, epochs: int = 1, device: torch.device | None = None,
    train_batch_size: int = 256,
) -> Path:
    """Train a fresh P3 only on graph-bound samples and attest its exact provenance."""

    allowed = tuple(sample for sample in pooled_manifest.samples["train"]
                    if sample.key.target_date <= graph_manifest.train_end_date)
    if not allowed:
        raise ValueError("no pooled P3 training samples fall within the graph train boundary")
    if epochs < 1 or epochs > 10:
        raise ValueError("graph-bound P3 epochs must be between 1 and 10")
    if train_batch_size < 1:
        raise ValueError("graph-bound P3 train_batch_size must be positive")
    price_dim, news_dim = allowed[0].x_price_raw.shape[1], allowed[0].x_news.shape[1]
    if news_dim == 0:
        raise ValueError("graph-bound P3 requires attached news features")
    selected_device = device or torch.device("cpu")
    _validate_graph_manifest(graph_manifest)
    _seed_graph_device(seed, selected_device)
    model = PooledPriceNewsLSTM(price_dim, news_dim, max(pooled_manifest.ticker_to_id.values()) + 1,
                                 use_gate=True, dropout=0.0).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    for _ in range(epochs):
        for x_price, x_news, news_mask, ticker_ids, targets in _pooled_training_batches(
            allowed, store, selected_device, train_batch_size,
        ):
            optimizer.zero_grad()
            prediction = model(x_price, x_news, news_mask, ticker_ids)
            torch.nn.functional.mse_loss(prediction, targets).backward()
            optimizer.step()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "graph_bound_p3_warm_start.pt"
    torch.save({"config_name": "P3", "seed": seed, "model_state": model.state_dict(),
                "graph_bound_warm_start": True,
                "max_training_target_date": max(sample.key.target_date for sample in allowed),
                "graph_train_end_date": graph_manifest.train_end_date,
                "training_sample_hash": _canonical_sample_hash(allowed),
                "graph_manifest_hash": graph_manifest.content_hash("train")}, path)
    return path


def build_graph_safe_p3_checkpoint(
    pooled_manifest: PooledManifest, graph_manifest: GraphManifest, output_dir: Path | str, seed: int,
    warm_start_checkpoint: Path | str, store: PreprocessorStore, epochs: int = 1,
    device: torch.device | None = None, train_batch_size: int = 256,
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
    if train_batch_size < 1:
        raise ValueError("graph-safe P3 train_batch_size must be positive")
    warm = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=False)
    if warm.get("config_name") != "P3" or not isinstance(warm.get("model_state"), dict):
        raise ValueError("warm-start checkpoint must be a trained P3 checkpoint")
    expected_sample_hash = _canonical_sample_hash(allowed)
    expected_graph_hash = graph_manifest.content_hash("train")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not warm.get("graph_bound_warm_start"):
        raise ValueError("warm-start checkpoint must be a verified graph-bound P3 source")
    if (warm.get("graph_train_end_date") != graph_manifest.train_end_date
            or warm.get("training_sample_hash") != expected_sample_hash
            or warm.get("graph_manifest_hash") != expected_graph_hash
            or warm.get("max_training_target_date") != max(sample.key.target_date for sample in allowed)):
        raise ValueError("graph-bound P3 warm-start provenance does not match restricted graph samples")
    dimensions = {(sample.x_price_raw.shape[1], sample.x_news.shape[1]) for sample in allowed}
    if len(dimensions) != 1:
        raise ValueError("graph-safe P3 samples must have one shared feature width")
    price_dim, news_dim = dimensions.pop()
    if news_dim == 0:
        raise ValueError("graph-safe P3 checkpoint requires attached P3 news features")
    num_tickers = max(pooled_manifest.ticker_to_id.values()) + 1
    selected_device = device or torch.device("cpu")
    _validate_graph_manifest(graph_manifest)
    _seed_graph_device(seed, selected_device)
    model = PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=True, dropout=0.0)
    model.load_state_dict(warm["model_state"], strict=True)
    model.to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        for x_price, x_news, news_mask, ticker_ids, targets in _pooled_training_batches(
            allowed, store, selected_device, train_batch_size,
        ):
            optimizer.zero_grad()
            prediction = model(x_price, x_news, news_mask, ticker_ids)
            loss = torch.nn.functional.mse_loss(prediction, targets)
            loss.backward()
            optimizer.step()
    checkpoint = out / "graph_safe_p3.pt"
    torch.save({
        "model_state": model.state_dict(),
        "graph_safe": True,
        "seed": seed,
        "max_training_target_date": max(sample.key.target_date for sample in allowed),
        "graph_train_end_date": graph_manifest.train_end_date,
        "training_sample_count": len(allowed),
        "refinement_epochs": epochs,
        "training_sample_hash": expected_sample_hash,
        "graph_manifest_hash": expected_graph_hash,
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

    if args.epochs < 1 or args.epochs > 20:
        raise ValueError("screening epochs must be between 1 and 20")
    if args.batch_size == 256:
        inputs = build_screening_inputs(
            args.smoke, args.max_tickers, args.phase, horizon=args.horizon, regime=args.regime
        )
    else:
        inputs = build_screening_inputs(
            args.smoke, args.max_tickers, args.phase, args.batch_size, horizon=args.horizon,
            regime=args.regime,
        )
    manifest = assert_shared_manifest({name: inputs.manifest for name in _phase_configs(args.phase)})
    out = Path(args.output_dir) / f"h{args.horizon}"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "screening_metadata.json", {
        "phase": args.phase, "regime": args.regime, "horizon": args.horizon, "epochs": args.epochs,
        "seed": args.seed,
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

    if args.epochs < 1 or args.epochs > 10:
        raise ValueError("screening epochs must be between 1 and 10")
    horizon = args.horizon
    if args.batch_size == 256:
        inputs = build_screening_inputs(args.smoke, args.max_tickers, "P3", horizon=horizon)
    else:
        inputs = build_screening_inputs(args.smoke, args.max_tickers, "P3", args.batch_size, horizon=horizon)
    raw = load_and_split_price_data(_ROOT / "data" / "processed")
    selected = tuple(inputs.smoke_filter["selected_tickers"])
    raw = _select_tickers(raw, selected)
    full_frames = {
        ticker: np_concat_frames(raw.frames[ticker][name] for name in ("train", "val", "test"))
        for ticker in selected
    }
    panel = load_runner_news_panel(_ROOT / "data" / "features" / "dual_group_news_panel.parquet", selected,
                                   _train_news_cutoffs(inputs.manifest))
    graph_store = _fit_graph_preprocessors(full_frames)
    graph_horizon = horizon
    pooled_horizon = horizon
    graph = build_graph_manifest(full_frames, panel, graph_store, horizon=graph_horizon)
    _validate_graph_manifest(graph)
    device = resolve_graph_device(args.device)
    graph_pooled = build_pooled_manifest(raw, graph_store, horizon=pooled_horizon)
    # The pooled and graph manifests are hashed independently, so a horizon mismatch
    # between them would not be caught by any content-hash check: it would silently
    # train the graph-safe P3 checkpoint on targets a different number of steps ahead
    # than the graph snapshots it initializes.  Reject it explicitly.
    _assert_matched_horizon(pooled_horizon, graph_horizon)
    graph_pooled = PooledManifest(
        {split: tuple(attach_news(graph_pooled.samples[split], panel, panel.feature_cols))
         for split in ("train", "val", "test")},
        graph_pooled.exclusions, graph_pooled.ticker_to_id, graph_pooled.preprocessing_hash,
    )
    out = Path(args.output_dir) / f"h{horizon}"
    out.mkdir(parents=True, exist_ok=True)
    warm_start = args.p3_checkpoint or build_graph_bound_p3_warm_start(
        graph_pooled, graph, out, args.seed, graph_store, epochs=1, device=device,
        train_batch_size=args.batch_size,
    )
    graph_safe = build_graph_safe_p3_checkpoint(graph_pooled, graph, out, args.seed, warm_start,
                                                 graph_store, epochs=1, device=device,
                                                 train_batch_size=args.batch_size)
    graph_hash = graph.content_hash("train")
    results = {
        name: _run_one_graph_model(
            GraphAblationModel.from_p3_checkpoint(
                str(graph_safe), use_gnn=name == "G1", graph_train_end_date=graph.train_end_date,
                graph_manifest_hash=graph_hash,
            ), graph, graph_store, name, args.epochs, args.seed, out / name, device,
            validation_batch_size=args.graph_batch_size, train_batch_size=args.graph_train_batch_size,
        )
        for name in ("G0", "G1")
    }
    payload = {"phase": "graph", "horizon": horizon, "graph_hashes": dict(graph.hashes),
               "graph_train_hash": graph_hash,
               "graph_safe_p3_checkpoint": str(graph_safe), "runtime": _graph_runtime_metadata(args.device, device),
               "results": results,
               "paired_delta": results["G1"]["validation_loss"] - results["G0"]["validation_loss"]}
    _write_json(out / "graph_validation_comparison.json", payload)
    return out / "graph_validation_comparison.json"


def _run_one_graph_model(
    model: GraphAblationModel, graph: GraphManifest, store: PreprocessorStore, name: str,
    epochs: int, seed: int, output: Path, device: str | torch.device = "cpu",
    validation_batch_size: int = 32, train_batch_size: int = 32,
) -> dict[str, Any]:
    selected_device = resolve_graph_device(device) if isinstance(device, str) else device
    _validate_graph_run_provenance(model, graph)
    if not graph.snapshots:
        raise ValueError("graph manifest must contain snapshots")
    model.configure_positivity(store)
    _seed_graph_device(seed, selected_device)
    model.to(selected_device)
    output.mkdir(parents=True, exist_ok=True)
    train = [snapshot for snapshot in graph.snapshots if snapshot.split == "train"]
    validation = [snapshot for snapshot in graph.snapshots if snapshot.split == "val"]
    if not train or not validation:
        raise ValueError("graph manifest requires non-empty train and validation snapshots")
    if validation_batch_size < 1:
        raise ValueError("validation_batch_size must be positive")
    if train_batch_size < 1:
        raise ValueError("train_batch_size must be positive")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, weight_decay=1e-5)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        # Training keeps its deterministic chronological (unshuffled) snapshot order; snapshots
        # are stacked into equal-weighted mini-batches so one optimizer.step covers many graphs,
        # for both G0 and G1 identically. This is an approved R10 SGD re-baseline (one step per
        # batch instead of per snapshot); at train_batch_size=1 it matches the per-snapshot loop.
        epoch_loss_sum = torch.zeros((), device=selected_device)
        epoch_snapshot_count = 0
        for start in range(0, len(train), train_batch_size):
            snapshots = train[start:start + train_batch_size]
            optimizer.zero_grad()
            predictions, targets = _graph_prediction_batch(model, snapshots, selected_device)
            loss = _mean_snapshot_mse(predictions, targets)
            loss.backward()
            if any(parameter.grad is not None for parameter in model.price_encoder.parameters()):
                raise RuntimeError("frozen graph encoder received gradients")
            optimizer.step()
            epoch_loss_sum = epoch_loss_sum + loss.detach() * len(snapshots)
            epoch_snapshot_count += len(snapshots)
        losses.append((epoch_loss_sum / epoch_snapshot_count).item())
    model.eval()
    records = []
    with torch.no_grad():
        validation_loss_sum = torch.zeros((), device=selected_device)
        validation_snapshot_count = 0
        for start in range(0, len(validation), validation_batch_size):
            snapshots = validation[start:start + validation_batch_size]
            predictions, targets = _graph_prediction_batch(model, snapshots, selected_device)
            # Keep the original per-snapshot weighting exactly, even if a future
            # manifest contains snapshots with different node counts.
            batch_loss = _mean_snapshot_mse(predictions, targets)
            validation_loss_sum = validation_loss_sum + batch_loss * len(snapshots)
            validation_snapshot_count += len(snapshots)
            for snapshot, snapshot_predictions in zip(snapshots, predictions.cpu(), strict=True):
                for node, prediction in zip(snapshot.nodes, snapshot_predictions, strict=True):
                    records.append({"ticker_id": node.ticker_id, "target_date": snapshot.target_date,
                                    "prediction_norm": float(prediction), "target_raw": node.y_raw})
        validation_loss = (validation_loss_sum / validation_snapshot_count).item()
    evaluation = evaluate_records(records, store)
    _write_json(output / "results.json", {"config_name": name, "graph_hash": graph.content_hash("val"),
                                            "train_losses": losses, "validation_loss": validation_loss,
                                            "runtime": _graph_runtime_metadata(str(device), selected_device),
                                            "validation_metrics": evaluation["metrics"],
                                            "nonpositive_prediction_rate": evaluation["nonpositive_prediction_rate"]})
    return {"graph_hash": graph.content_hash("val"), "validation_loss": validation_loss,
            "validation_metrics": evaluation["metrics"]}


def _graph_prediction(
    model: GraphAblationModel, snapshot: Any, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return model(
        torch.from_numpy(snapshot.x_price.copy()).to(device, non_blocking=non_blocking),
        torch.from_numpy(snapshot.x_news.copy()).to(device, non_blocking=non_blocking),
        torch.from_numpy(snapshot.news_mask.copy()).to(device, non_blocking=non_blocking),
        torch.tensor([node.ticker_id for node in snapshot.nodes], dtype=torch.long, device=device),
        torch.from_numpy(snapshot.adjacency.copy()).to(device, non_blocking=non_blocking),
    ), torch.tensor([node.y_norm for node in snapshot.nodes], dtype=torch.float32, device=device)


def _graph_prediction_batch(
    model: GraphAblationModel, snapshots: list[Any], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run validation snapshots as one graph batch without changing train updates."""

    if not snapshots:
        raise ValueError("graph snapshot batch cannot be empty")
    first = snapshots[0]
    if any(snapshot.x_price.shape != first.x_price.shape
           or snapshot.x_news.shape != first.x_news.shape
           or snapshot.news_mask.shape != first.news_mask.shape
           or snapshot.adjacency.shape != first.adjacency.shape
           for snapshot in snapshots[1:]):
        predictions = []
        targets = []
        for snapshot in snapshots:
            prediction, target = _graph_prediction(model, snapshot, device)
            predictions.append(prediction)
            targets.append(target)
        return torch.stack(predictions), torch.stack(targets)
    non_blocking = device.type == "cuda"
    return model(
        torch.from_numpy(np.stack([snapshot.x_price for snapshot in snapshots])).to(device, non_blocking=non_blocking),
        torch.from_numpy(np.stack([snapshot.x_news for snapshot in snapshots])).to(device, non_blocking=non_blocking),
        torch.from_numpy(np.stack([snapshot.news_mask for snapshot in snapshots])).to(
            device, non_blocking=non_blocking),
        torch.tensor([[node.ticker_id for node in snapshot.nodes] for snapshot in snapshots],
                     dtype=torch.long, device=device),
        torch.from_numpy(np.stack([snapshot.adjacency for snapshot in snapshots])).to(
            device, non_blocking=non_blocking),
    ), torch.tensor([[node.y_norm for node in snapshot.nodes] for snapshot in snapshots],
                    dtype=torch.float32, device=device)


def _mean_snapshot_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return equal-weighted snapshot MSE, independent of validation batch size."""

    if predictions.shape != targets.shape or predictions.ndim != 2 or not predictions.shape[0]:
        raise ValueError("predictions and targets must be non-empty [batch, nodes] tensors")
    return torch.stack([
        torch.nn.functional.mse_loss(prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
    ]).mean()


def _assert_matched_horizon(pooled_horizon: int, graph_horizon: int) -> None:
    """Reject a graph run whose pooled and graph manifests use different horizons.

    Content hashes are computed independently for the two manifests and are never
    compared to each other, so a horizon mismatch is a silent leakage risk rather
    than a hash failure.  This guard enforces the single-horizon invariant explicitly.
    """

    if pooled_horizon != graph_horizon:
        raise ValueError(
            f"pooled horizon {pooled_horizon} must equal graph horizon {graph_horizon} "
            "within one graph run"
        )


def resolve_graph_device(requested: str) -> torch.device:
    """Resolve the graph runner's bounded device choice without silent CUDA fallback."""

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("graph device must be one of: auto, cpu, cuda")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested for graph screening but is unavailable")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu")


def _validate_graph_manifest(graph: GraphManifest) -> None:
    required_hashes = {"snapshots", "node_vocabulary", "adjacency", "tensors"}
    if not required_hashes.issubset(graph.hashes) or any(not graph.hashes[name] for name in required_hashes):
        raise ValueError("graph manifest lacks required provenance hashes")
    for split in ("train", "val"):
        if not graph.content_hash(split):
            raise ValueError(f"graph manifest cannot hash {split} split")


def _validate_graph_run_provenance(model: GraphAblationModel, graph: GraphManifest) -> None:
    _validate_graph_manifest(graph)
    if getattr(model, "graph_train_end_date", None) != graph.train_end_date:
        raise ValueError("graph model train boundary differs from graph manifest")
    if getattr(model, "graph_manifest_hash", None) != graph.content_hash("train"):
        raise ValueError("graph model manifest hash differs from graph manifest")


def _seed_graph_device(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _graph_runtime_metadata(requested: str, device: torch.device) -> dict[str, str | None]:
    return {"requested": requested, "selected": str(device), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda}


def np_concat_frames(frames: Any) -> Any:
    import pandas as pd

    return pd.concat(tuple(frames), ignore_index=True)


def _fit_graph_preprocessors(full_frames: Mapping[str, Any]) -> PreprocessorStore:
    common_dates = sorted(set.intersection(*[
        set(frame["date"].astype(str)) for frame in full_frames.values()
    ]))
    train_end = common_dates[int(len(common_dates) * 0.7) - 1]
    return PreprocessorStore({
        ticker_id: TickerPreprocessor.fit(
            frame.loc[frame["date"].astype(str) <= train_end], ["parkinson_volatility"], "parkinson_volatility",
        )
        for ticker_id, (_, frame) in enumerate(sorted(full_frames.items()))
    })


def build_screening_inputs(
    smoke: bool, max_tickers: int | None, phase: str = "pooled", batch_size: int = 256,
    horizon: int = 5, regime: str = "pooled",
) -> ScreeningInputs:
    """Apply ticker filtering before fitting preprocessing or constructing the shared manifest."""

    if max_tickers is not None and max_tickers < 1:
        raise ValueError("max_tickers must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if regime not in {"pooled", "common-date"}:
        raise ValueError("regime must be one of: pooled, common-date")
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
    manifest = build_pooled_manifest(splits, store, horizon=horizon)
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
    if regime == "common-date":
        # Reuse the graph common-date axis to keep, per ticker, only TRAIN anchors whose window
        # sits on globally-common trading dates.  Restriction runs AFTER scaler fitting and
        # (optional) news attachment, so the per-ticker scalers, winsor bounds, and the news
        # panel are byte-identical to the pooled regime.  Validation/test stay the full pooled
        # held-out sets, so both regimes are scored on an identical evaluation set and only the
        # training-sample SET differs.
        full_frames = {
            ticker: np_concat_frames(splits.frames[ticker][name] for name in ("train", "val", "test"))
            for ticker in splits.frames
        }
        manifest = restrict_train_to_common_dates(manifest, common_trading_dates(full_frames, store))
    loaders = {
        split: DataLoader(
            _ManifestDataset(manifest.samples[split], store),
            batch_size=batch_size,
            shuffle=False,
            # Windows worker processes add substantial overhead for this in-memory
            # manifest, so keep the conservative single-process loader.
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ("train", "val")
    }
    return ScreeningInputs(manifest, store, loaders, {
        "enabled": smoke, "max_tickers": requested, "selected_tickers": selected,
        "rows_per_split": _SMOKE_ROWS_PER_SPLIT if smoke else None, "horizon": horizon,
        "regime": regime,
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
    parser.add_argument("--regime", choices=("pooled", "common-date"), default="pooled",
                        help="A1 training-sample set: full pooled history or common-date intersection")
    parser.add_argument("--horizon", type=int, default=5, choices=(1, 5, 10, 22),
                        help="forecast horizon in trading days (5 is the primary target)")
    parser.add_argument("--p3-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--graph-batch-size", type=int, default=32,
                        help="validation graph snapshots per batch")
    parser.add_argument("--graph-train-batch-size", type=int, default=32,
                        help="training graph snapshots per optimizer step (mini-batched, unshuffled)")
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "results" / "pooled_news_gnn_pilot")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-tickers", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.phase == "graph" and args.regime != "pooled":
        # The graph manifest is common-date by construction; a non-default regime here would
        # be silently ignored, so reject it rather than run a misleadingly-labelled ablation.
        parser.error("--regime applies only to pooled P0-P3 phases; --phase graph is common-date by construction")
    return args


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
