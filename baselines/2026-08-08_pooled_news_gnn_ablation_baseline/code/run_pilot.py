"""Validation-only P0-P3 pooled screening runner."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    build_masked_graph_manifest,
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
    train_batch_size: int = 256, dropout: float = 0.0,
) -> Path:
    """Train a fresh P3 only on graph-bound samples and attest its exact provenance.

    ``dropout`` selects the regularization used for the frozen graph backbone.  The screening
    P0-P3 comparison uses ``dropout=0.2``; the graph runner passes that so G0/G1 wrap a
    screening-configuration P3 rather than a bespoke dropout-free backbone.  The training set stays
    the leakage-safe graph-bound subset (``target_date <= graph.train_end_date``) so the frozen
    backbone never sees data from the graph val/test period.
    """

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
                                 use_gate=True, dropout=dropout).to(selected_device)
    model.train()
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
    device: torch.device | None = None, train_batch_size: int = 256, dropout: float = 0.0,
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
    model = PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=True, dropout=dropout)
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

    if args.epochs < 1 or args.epochs > 10:
        raise ValueError("screening epochs must be between 1 and 10")
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


_MAX_GRAPH_EPOCHS = 50


def _validate_graph_epochs(epochs: int) -> None:
    """Bound graph G0/G1 training length.

    The pooled P0-P3 screening keeps the 1-10 experimentation cap. The graph ablation trains
    only the small message-passing head and is run to convergence, so it permits longer,
    user-approved runs (e.g. 15 epochs) while still rejecting runaway lengths.
    """

    if epochs < 1 or epochs > _MAX_GRAPH_EPOCHS:
        raise ValueError(f"graph screening epochs must be between 1 and {_MAX_GRAPH_EPOCHS}")


def run_graph_screening(args: argparse.Namespace) -> Path:
    """Run the bounded, matched G0/G1 graph ablation without touching P0-P3 semantics."""

    _validate_graph_epochs(args.epochs)
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
    graph_mode = getattr(args, "graph", "intersection")
    adjacency = getattr(args, "adjacency", "dense")
    top_k = getattr(args, "top_k", 8)
    corr_threshold = getattr(args, "corr_threshold", 0.7)
    graph = _build_graph_manifest_for_mode(
        graph_mode, graph_pooled, graph_store, full_frames, panel, graph_horizon,
        adjacency, top_k, corr_threshold,
    )
    _validate_graph_manifest(graph)
    out = Path(args.output_dir) / f"h{horizon}"
    out.mkdir(parents=True, exist_ok=True)
    # Screening-configuration backbone: G0/G1 wrap a P3 trained with the standard screening depth
    # + dropout (``--backbone-epochs`` / ``--backbone-dropout``, defaults 5 / 0.2), not a bespoke
    # 1+1-epoch dropout-0 backbone.  Training stays on the leakage-safe graph-bound subset.
    warm_epochs = max(1, args.backbone_epochs - 1)
    warm_start = args.p3_checkpoint or build_graph_bound_p3_warm_start(
        graph_pooled, graph, out, args.seed, graph_store, epochs=warm_epochs, device=device,
        train_batch_size=args.batch_size, dropout=args.backbone_dropout,
    )
    graph_safe = build_graph_safe_p3_checkpoint(graph_pooled, graph, out, args.seed, warm_start,
                                                 graph_store, epochs=1, device=device,
                                                 train_batch_size=args.batch_size,
                                                 dropout=args.backbone_dropout)
    graph_hash = graph.content_hash("train")
    models = {
        name: GraphAblationModel.from_p3_checkpoint(
            str(graph_safe), use_gnn=name == "G1", graph_train_end_date=graph.train_end_date,
            graph_manifest_hash=graph_hash,
        )
        for name in ("G0", "G1")
    }
    # G0 and G1 load the identical graph-safe P3 weights, so their FROZEN encoder + gate produce
    # bit-identical node embeddings.  Compute that ``base`` cache ONCE and reuse it for both
    # models and all epochs (proven equivalent to per-epoch recompute by the equivalence test).
    use_cache = not args.no_base_cache
    shared_base = None
    if use_cache:
        _assert_shared_frozen_encoder(models["G0"], models["G1"])
        models["G0"].to(device)
        shared_base = _build_shared_graph_base(models["G0"], graph, device,
                                               args.graph_train_batch_size, args.graph_batch_size)
    results = {
        name: _run_one_graph_model(
            models[name], graph, graph_store, name, args.epochs, args.seed, out / name, device,
            validation_batch_size=args.graph_batch_size, train_batch_size=args.graph_train_batch_size,
            base_cache=shared_base, use_base_cache=use_cache,
        )
        for name in ("G0", "G1")
    }
    snapshot_dates = {snapshot.target_date for snapshot in graph.snapshots}
    split_snapshot_counts = {
        split: sum(1 for snapshot in graph.snapshots if snapshot.split == split)
        for split in ("train", "val", "test")
    }
    payload = {"phase": "graph", "graph_mode": graph_mode,
               "adjacency": _adjacency_config(adjacency, top_k, corr_threshold),
               "edge_density": _edge_density_stats(graph), "horizon": horizon,
               "graph_hashes": dict(graph.hashes), "graph_train_hash": graph_hash,
               "snapshot_count": len(graph.snapshots), "distinct_date_count": len(snapshot_dates),
               "split_snapshot_counts": split_snapshot_counts,
               "graph_safe_p3_checkpoint": str(graph_safe), "runtime": _graph_runtime_metadata(args.device, device),
               "results": results,
               "paired_delta": results["G1"]["validation_loss"] - results["G0"]["validation_loss"]}
    _write_json(out / "graph_validation_comparison.json", payload)
    return out / "graph_validation_comparison.json"


def _edge_density_stats(graph: GraphManifest) -> dict[str, float | int]:
    """Off-diagonal edge counts per PRESENT node row, aggregated over all snapshots."""

    per_row: list[int] = []
    for snapshot in graph.snapshots:
        adjacency = snapshot.adjacency
        node_count = adjacency.shape[0]
        presence = (snapshot.presence_mask if snapshot.presence_mask is not None
                    else np.ones(node_count, dtype=np.int8))
        present = np.flatnonzero(presence)
        for index in present:
            row = adjacency[index, present].copy()
            row[np.flatnonzero(present == index)] = 0.0  # drop the self-loop
            per_row.append(int(np.count_nonzero(row)))
    counts = np.asarray(per_row, dtype=float)
    return {
        "avg_offdiag_nonzeros_per_present_row": float(counts.mean()),
        "max_offdiag_nonzeros_per_present_row": int(counts.max()),
        "min_offdiag_nonzeros_per_present_row": int(counts.min()),
        "present_row_count": int(counts.size),
    }


def _graph_validation_loss(
    model: GraphAblationModel, validation: list[Any], device: torch.device, validation_batch_size: int,
    base: list[torch.Tensor] | None = None,
) -> float:
    """Present-node-weighted validation MSE for one epoch's learning-curve point."""

    model.eval()
    loss_sum = torch.zeros((), device=device)
    snapshot_count = 0
    with torch.no_grad():
        for start in range(0, len(validation), validation_batch_size):
            snapshots = validation[start:start + validation_batch_size]
            predictions, targets, presence = _graph_predict(
                model, snapshots, device,
                None if base is None else base[start:start + validation_batch_size],
            )
            loss_sum = loss_sum + _mean_snapshot_mse(predictions, targets, presence) * len(snapshots)
            snapshot_count += len(snapshots)
    return (loss_sum / snapshot_count).item()


def _plot_learning_curve(train_losses: list[float], validation_losses: list[float], path: Path) -> None:
    """Plot train vs validation loss for convergence / overfitting inspection (best-effort)."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(epochs, train_losses, marker="o", label="train")
        axis.plot(epochs, validation_losses, marker="s", label="validation")
        axis.set_xlabel("epoch")
        axis.set_ylabel("MSE (normalized scale)")
        axis.set_title(path.parent.name)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=120)
        plt.close(figure)
    except Exception:  # noqa: BLE001 - plotting is best-effort; a backend or IO error must not fail training
        return


def _adjacency_config(adjacency: str, top_k: int, corr_threshold: float) -> dict[str, Any]:
    """Payload record of the adjacency mode and its active hyperparameter."""

    config: dict[str, Any] = {"mode": adjacency}
    if adjacency == "knn":
        config["top_k"] = top_k
    elif adjacency == "threshold":
        config["corr_threshold"] = corr_threshold
    return config


def _build_graph_manifest_for_mode(
    graph_mode: str, graph_pooled: PooledManifest, graph_store: PreprocessorStore,
    full_frames: Mapping[str, Any], panel: Any, horizon: int,
    adjacency: str, top_k: int, corr_threshold: float,
) -> GraphManifest:
    """Build the masked (variable-node union) or intersection (fixed-node) graph manifest.

    Both paths receive the same edge-sparsity config so ``--adjacency`` applies identically;
    masked is availability-aware over PRESENT tickers, intersection is the synchronized-date
    fixed-node graph.
    """

    if graph_mode == "masked":
        return build_masked_graph_manifest(graph_pooled, graph_store, adjacency=adjacency,
                                           top_k=top_k, corr_threshold=corr_threshold)
    return build_graph_manifest(full_frames, panel, graph_store, horizon=horizon,
                                adjacency=adjacency, top_k=top_k, corr_threshold=corr_threshold)


def _run_one_graph_model(
    model: GraphAblationModel, graph: GraphManifest, store: PreprocessorStore, name: str,
    epochs: int, seed: int, output: Path, device: str | torch.device = "cpu",
    validation_batch_size: int = 32, train_batch_size: int = 32,
    base_cache: Mapping[str, list[torch.Tensor]] | None = None, use_base_cache: bool = True,
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
    # Frozen-encoder ``base`` cache (proposed updates #1 + #2): computed once per seed over PRESENT
    # nodes and reused across every epoch (and, when supplied via ``base_cache``, across G0/G1).
    # ``use_base_cache=False`` recomputes the encoder each batch (the pre-cache path) and is used
    # by the equivalence + speedup measurements only.
    encode_seconds = 0.0
    if base_cache is not None:
        train_base, val_base = base_cache["train"], base_cache["val"]
    elif use_base_cache:
        encode_start = time.perf_counter()
        train_base = _precompute_graph_base(model, train, selected_device, train_batch_size)
        val_base = _precompute_graph_base(model, validation, selected_device, validation_batch_size)
        encode_seconds = time.perf_counter() - encode_start
    else:
        train_base = val_base = None
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, weight_decay=1e-5)
    losses: list[float] = []
    validation_losses: list[float] = []
    train_start = time.perf_counter()
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
            predictions, targets, presence = _graph_predict(
                model, snapshots, selected_device,
                None if train_base is None else train_base[start:start + train_batch_size],
            )
            loss = _mean_snapshot_mse(predictions, targets, presence)
            loss.backward()
            if any(parameter.grad is not None for parameter in model.price_encoder.parameters()):
                raise RuntimeError("frozen graph encoder received gradients")
            optimizer.step()
            epoch_loss_sum = epoch_loss_sum + loss.detach() * len(snapshots)
            epoch_snapshot_count += len(snapshots)
        losses.append((epoch_loss_sum / epoch_snapshot_count).item())
        # Per-epoch validation loss lets the learning curve expose convergence and any
        # train-vs-validation divergence (overfitting) rather than a single end-of-run point.
        validation_losses.append(
            _graph_validation_loss(model, validation, selected_device, validation_batch_size, val_base)
        )
    train_seconds = time.perf_counter() - train_start
    _plot_learning_curve(losses, validation_losses, output / "learning_curve.png")
    model.eval()
    validation_loss, val_records, val_eval = _evaluate_graph_split(
        model, validation, val_base, store, selected_device, validation_batch_size)
    _write_graph_predictions(val_records, val_eval, output / "predictions.json")
    result_payload: dict[str, Any] = {
        "config_name": name, "graph_hash": graph.content_hash("val"),
        "train_losses": losses, "validation_losses": validation_losses,
        "validation_loss": validation_loss,
        "runtime": _graph_runtime_metadata(str(device), selected_device),
        "timing": {"base_cache": base_cache is not None or use_base_cache,
                   "encode_seconds": encode_seconds, "train_seconds": train_seconds,
                   "seconds_per_epoch": train_seconds / epochs},
        "train_snapshot_count": len(train), "validation_snapshot_count": len(validation),
        "present_validation_node_count": len(val_records),
        "validation_metrics": val_eval["metrics"],
        "nonpositive_prediction_rate": val_eval["nonpositive_prediction_rate"],
    }
    # Held-out TEST evaluation (for the paper's val+test table).  Backbone is graph-bound (train),
    # message-passing head is train-only, so test is a clean out-of-sample forecast.
    test = [snapshot for snapshot in graph.snapshots if snapshot.split == "test"]
    outcome = {"graph_hash": graph.content_hash("val"), "validation_loss": validation_loss,
               "validation_metrics": val_eval["metrics"]}
    if test:
        test_base = None
        if base_cache is not None and "test" in base_cache:
            test_base = base_cache["test"]
        elif use_base_cache:
            test_base = _precompute_graph_base(model, test, selected_device, validation_batch_size)
        test_loss, test_records, test_eval = _evaluate_graph_split(
            model, test, test_base, store, selected_device, validation_batch_size)
        _write_graph_predictions(test_records, test_eval, output / "predictions_test.json")
        result_payload.update({
            "test_loss": test_loss, "test_snapshot_count": len(test),
            "present_test_node_count": len(test_records),
            "test_metrics": test_eval["metrics"],
            "test_nonpositive_prediction_rate": test_eval["nonpositive_prediction_rate"],
        })
        outcome.update({"test_loss": test_loss, "test_metrics": test_eval["metrics"]})
    _write_json(output / "results.json", result_payload)
    return outcome


def _evaluate_graph_split(
    model: GraphAblationModel, snapshots: list[Any], base: list[torch.Tensor] | None,
    store: PreprocessorStore, device: torch.device, batch_size: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Present-node-weighted loss + raw-scale ``evaluate_records`` metrics for one split."""

    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        loss_sum = torch.zeros((), device=device)
        snapshot_count = 0
        for start in range(0, len(snapshots), batch_size):
            batch = snapshots[start:start + batch_size]
            predictions, targets, presence = _graph_predict(
                model, batch, device, None if base is None else base[start:start + batch_size])
            # Keep the original per-snapshot weighting exactly, even if a future
            # manifest contains snapshots with different node counts.
            loss_sum = loss_sum + _mean_snapshot_mse(predictions, targets, presence) * len(batch)
            snapshot_count += len(batch)
            presence_cpu = presence.cpu()
            for snapshot, snapshot_predictions, snapshot_presence in zip(
                batch, predictions.cpu(), presence_cpu, strict=True
            ):
                for index, (node, prediction) in enumerate(
                    zip(snapshot.nodes, snapshot_predictions, strict=True)
                ):
                    # Masked absent nodes are excluded from the raw-scale evaluation.
                    if not bool(snapshot_presence[index]):
                        continue
                    records.append({"ticker_id": node.ticker_id, "target_date": snapshot.target_date,
                                    "prediction_norm": float(prediction), "target_raw": node.y_raw})
        split_loss = (loss_sum / snapshot_count).item()
    return split_loss, records, evaluate_records(records, store)


def _snapshot_presence(snapshot: Any, device: torch.device) -> torch.Tensor | None:
    if snapshot.presence_mask is None:
        return None
    return torch.from_numpy(snapshot.presence_mask.copy()).to(device).to(dtype=torch.bool)


def _graph_prediction(
    model: GraphAblationModel, snapshot: Any, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    presence = _snapshot_presence(snapshot, device)
    prediction = model(
        torch.from_numpy(snapshot.x_price.copy()).to(device, non_blocking=non_blocking),
        torch.from_numpy(snapshot.x_news.copy()).to(device, non_blocking=non_blocking),
        torch.from_numpy(snapshot.news_mask.copy()).to(device, non_blocking=non_blocking),
        torch.tensor([node.ticker_id for node in snapshot.nodes], dtype=torch.long, device=device),
        torch.from_numpy(snapshot.adjacency.copy()).to(device, non_blocking=non_blocking),
        presence,
    )
    target = torch.tensor([node.y_norm for node in snapshot.nodes], dtype=torch.float32, device=device)
    resolved = presence if presence is not None else torch.ones_like(target, dtype=torch.bool)
    return prediction, target, resolved


def _graph_prediction_batch(
    model: GraphAblationModel, snapshots: list[Any], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        presences = []
        for snapshot in snapshots:
            prediction, target, presence = _graph_prediction(model, snapshot, device)
            predictions.append(prediction)
            targets.append(target)
            presences.append(presence)
        return torch.stack(predictions), torch.stack(targets), torch.stack(presences)
    non_blocking = device.type == "cuda"
    presence_mask = None
    if any(snapshot.presence_mask is not None for snapshot in snapshots):
        presence_mask = torch.from_numpy(
            np.stack([snapshot.presence_mask for snapshot in snapshots])).to(
            device, non_blocking=non_blocking).to(dtype=torch.bool)
    predictions = model(
        torch.from_numpy(np.stack([snapshot.x_price for snapshot in snapshots])).to(device, non_blocking=non_blocking),
        torch.from_numpy(np.stack([snapshot.x_news for snapshot in snapshots])).to(device, non_blocking=non_blocking),
        torch.from_numpy(np.stack([snapshot.news_mask for snapshot in snapshots])).to(
            device, non_blocking=non_blocking),
        torch.tensor([[node.ticker_id for node in snapshot.nodes] for snapshot in snapshots],
                     dtype=torch.long, device=device),
        torch.from_numpy(np.stack([snapshot.adjacency for snapshot in snapshots])).to(
            device, non_blocking=non_blocking),
        presence_mask,
    )
    targets = torch.tensor([[node.y_norm for node in snapshot.nodes] for snapshot in snapshots],
                           dtype=torch.float32, device=device)
    resolved = presence_mask if presence_mask is not None else torch.ones_like(targets, dtype=torch.bool)
    return predictions, targets, resolved


def _stacked_snapshot_inputs(
    snapshots: list[Any], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Batch adjacency / presence / ticker_ids / normalized targets without the raw price+news.

    Used by the cached path: the frozen ``base`` is already computed, so only the trainable
    head's inputs (adjacency, presence, ticker IDs for positivity, targets) are stacked here.
    """

    non_blocking = device.type == "cuda"
    presence_mask = None
    if any(snapshot.presence_mask is not None for snapshot in snapshots):
        presence_mask = torch.from_numpy(
            np.stack([snapshot.presence_mask for snapshot in snapshots])).to(
            device, non_blocking=non_blocking).to(dtype=torch.bool)
    ticker_ids = torch.tensor([[node.ticker_id for node in snapshot.nodes] for snapshot in snapshots],
                              dtype=torch.long, device=device)
    adjacency = torch.from_numpy(np.stack([snapshot.adjacency for snapshot in snapshots])).to(
        device, non_blocking=non_blocking)
    targets = torch.tensor([[node.y_norm for node in snapshot.nodes] for snapshot in snapshots],
                           dtype=torch.float32, device=device)
    return adjacency, presence_mask, ticker_ids, targets


def _precompute_graph_base(
    model: GraphAblationModel, snapshots: list[Any], device: torch.device, batch_size: int,
) -> list[torch.Tensor]:
    """Frozen-encoder ``base`` per snapshot, computed once (present-only) and reused across epochs.

    Returns one ``[nodes, hidden]`` tensor per snapshot in ``snapshots`` order, resident on
    ``device``.  The encoders are frozen + dropout-free, so these embeddings are bit-identical to
    the per-epoch recompute (asserted by the equivalence test) while removing the redundant work.
    """

    model.eval()
    non_blocking = device.type == "cuda"
    bases: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(snapshots), batch_size):
            chunk = snapshots[start:start + batch_size]
            presence = None
            if any(snapshot.presence_mask is not None for snapshot in chunk):
                presence = torch.from_numpy(
                    np.stack([snapshot.presence_mask for snapshot in chunk])).to(
                    device, non_blocking=non_blocking).to(dtype=torch.bool)
            base = model.encode_base(
                torch.from_numpy(np.stack([snapshot.x_price for snapshot in chunk])).to(
                    device, non_blocking=non_blocking),
                torch.from_numpy(np.stack([snapshot.x_news for snapshot in chunk])).to(
                    device, non_blocking=non_blocking),
                torch.from_numpy(np.stack([snapshot.news_mask for snapshot in chunk])).to(
                    device, non_blocking=non_blocking),
                torch.tensor([[node.ticker_id for node in snapshot.nodes] for snapshot in chunk],
                             dtype=torch.long, device=device),
                presence,
            )
            bases.extend(base[index] for index in range(base.shape[0]))
    return bases


def _graph_predict(
    model: GraphAblationModel, snapshots: list[Any], device: torch.device,
    base: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict one snapshot batch, using cached ``base`` when supplied (else encode on the fly)."""

    if base is None:
        return _graph_prediction_batch(model, snapshots, device)
    if len(base) != len(snapshots):
        raise ValueError("cached base must align one-to-one with the snapshot batch")
    adjacency, presence_mask, ticker_ids, targets = _stacked_snapshot_inputs(snapshots, device)
    predictions = model.apply_graph_head(torch.stack(base), adjacency, ticker_ids, presence_mask)
    resolved = presence_mask if presence_mask is not None else torch.ones_like(targets, dtype=torch.bool)
    return predictions, targets, resolved


def _build_shared_graph_base(
    model: GraphAblationModel, graph: GraphManifest, device: torch.device,
    train_batch_size: int, validation_batch_size: int,
) -> dict[str, list[torch.Tensor]]:
    """Precompute the train/val frozen ``base`` cache once, shared by G0 and G1."""

    train = [snapshot for snapshot in graph.snapshots if snapshot.split == "train"]
    validation = [snapshot for snapshot in graph.snapshots if snapshot.split == "val"]
    test = [snapshot for snapshot in graph.snapshots if snapshot.split == "test"]
    cache = {
        "train": _precompute_graph_base(model, train, device, train_batch_size),
        "val": _precompute_graph_base(model, validation, device, validation_batch_size),
    }
    if test:
        cache["test"] = _precompute_graph_base(model, test, device, validation_batch_size)
    return cache


def _assert_shared_frozen_encoder(first: GraphAblationModel, second: GraphAblationModel) -> None:
    """Reject sharing a base cache between two models whose frozen encoder/gate differ.

    The cache is only valid across G0/G1 because both load identical graph-safe P3 weights.  This
    guards against a future path where they diverge (which would silently poison the shared cache).
    """

    for name, left, right in (
        ("price_encoder", first.price_encoder, second.price_encoder),
        ("news_encoder", first.news_encoder, second.news_encoder),
    ):
        left_state, right_state = left.state_dict(), right.state_dict()
        if left_state.keys() != right_state.keys() or any(
            not torch.equal(left_state[key], right_state[key]) for key in left_state
        ):
            raise ValueError(f"shared base cache requires identical frozen {name} for G0 and G1")
    if not torch.equal(first.gate_logits, second.gate_logits):
        raise ValueError("shared base cache requires identical frozen gate for G0 and G1")


def _mean_snapshot_mse(
    predictions: torch.Tensor, targets: torch.Tensor, presence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return equal-weighted snapshot MSE, independent of validation batch size.

    When ``presence`` is given, each snapshot's MSE is averaged over its PRESENT nodes only,
    so masked absent nodes never enter the loss.
    """

    if predictions.shape != targets.shape or predictions.ndim != 2 or not predictions.shape[0]:
        raise ValueError("predictions and targets must be non-empty [batch, nodes] tensors")
    if presence is None:
        return torch.stack([
            torch.nn.functional.mse_loss(prediction, target)
            for prediction, target in zip(predictions, targets, strict=True)
        ]).mean()
    if presence.shape != predictions.shape:
        raise ValueError("presence must match the [batch, nodes] prediction shape")
    present = presence.to(dtype=torch.bool)
    per_snapshot = []
    for prediction, target, mask in zip(predictions, targets, present, strict=True):
        if not mask.any():
            raise ValueError("each snapshot must contain at least one present node")
        per_snapshot.append(torch.nn.functional.mse_loss(prediction[mask], target[mask]))
    return torch.stack(per_snapshot).mean()


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
            frame.loc[frame["date"].astype(str) <= train_end], ["parkinson_variance"], "parkinson_variance",
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
            splits.frames[ticker]["train"], ["parkinson_variance"], "parkinson_variance"
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
    parser.add_argument("--graph", choices=("intersection", "masked"), default="intersection",
                        help="graph construction: fixed-node date intersection (default) or "
                             "availability-aware masked variable-node union")
    parser.add_argument("--adjacency", choices=("dense", "knn", "threshold"), default="dense",
                        help="edge sparsity on the present-node subgraph: dense full correlation "
                             "(default), GAT-hybrid-style knn top-k, or |corr|>tau threshold")
    parser.add_argument("--top-k", type=int, default=8,
                        help="knn neighbours per present node (mutual, undirected)")
    parser.add_argument("--corr-threshold", type=float, default=0.7,
                        help="threshold adjacency keeps edges with |corr| > this value")
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
    parser.add_argument("--backbone-epochs", type=int, default=5,
                        help="graph backbone (frozen P3) total training epochs (screening default 5)")
    parser.add_argument("--backbone-dropout", type=float, default=0.2,
                        help="graph backbone (frozen P3) training dropout (screening default 0.2)")
    parser.add_argument("--no-base-cache", action="store_true",
                        help="disable the frozen-encoder base cache (recompute per epoch); for "
                             "the cache speedup/equivalence measurement only")
    args = parser.parse_args(argv)
    if args.phase == "graph" and args.regime != "pooled":
        # The graph manifest is common-date by construction; a non-default regime here would
        # be silently ignored, so reject it rather than run a misleadingly-labelled ablation.
        parser.error("--regime applies only to pooled P0-P3 phases; --phase graph is common-date by construction")
    if args.phase != "graph" and args.adjacency != "dense":
        # Sparse adjacency only shapes the graph edge set; it would be silently ignored on the
        # pooled P0-P3 path, so reject it rather than run a misleadingly-labelled screening.
        parser.error("--adjacency applies only to --phase graph")
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


def _write_graph_predictions(
    records: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any], path: Path
) -> None:
    """Persist per-observation raw target/prediction pairs for a Diebold-Mariano test.

    ``evaluation['predictions_raw']`` / ``['targets_raw']`` are produced in the same order as
    ``records`` (present validation nodes), so the aligned rows let a later DM test build the
    per-observation loss differential between G0 and G1 on the identical (date, node) set.
    """
    rows = [
        {"ticker_id": int(record["ticker_id"]), "target_date": str(record["target_date"]),
         "target_raw": float(target), "prediction_raw": float(prediction)}
        for record, target, prediction in zip(
            records, evaluation["targets_raw"], evaluation["predictions_raw"], strict=True
        )
    ]
    _write_json(path, rows)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "graph":
        run_graph_screening(arguments)
    else:
        run_pooled_screening(arguments)
