"""Train and evaluate the FINAL / PROPOSED model (G1) on the held-out test split.

G1 = the P3 backbone (pooled Price + News + per-ticker gate LSTM) plus a cross-stock
graph message-passing layer. It is the proposed model of the Track-B study; P0-P3 and
G0 are ablations.

The original pilot (``run_pilot.py``) is a *validation-only* screening runner: its graph
phase trains G0/G1 and reports validation metrics, but never saves a standalone G1
checkpoint and never scores the test split. This module adds exactly those two things,
reusing the pilot's own (reviewed) building blocks so no modelling logic is duplicated:

* ``train_g1`` builds the graph-safe P3 initialisation, trains G1, saves ``g1_final.pt``
  (and keeps the ``graph_safe_p3.pt`` backbone next to it), then scores the test split.
* ``infer_g1`` rebuilds the deterministic graph context, loads ``g1_final.pt``, and scores
  the test split.

Both train and infer require the project dataset under ``data/`` (price CSVs + the news
panel parquet). ``view`` in ``reproduce.py`` needs neither and works standalone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

import run_pilot as rp
from data import PooledManifest, attach_news, build_graph_manifest, build_pooled_manifest
from models import GraphAblationModel
from train import evaluate_records

_METRIC_ORDER = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


class GraphContext:
    """Deterministic graph manifest + preprocessors + graph-safe P3 backbone."""

    def __init__(self, graph: Any, store: Any, graph_safe: Path, device: torch.device,
                 selected: tuple[str, ...], horizon: int) -> None:
        self.graph = graph
        self.store = store
        self.graph_safe = graph_safe
        self.device = device
        self.selected = selected
        self.horizon = horizon

    @property
    def graph_hash(self) -> str:
        return self.graph.content_hash("train")


def build_graph_context(
    output_dir: Path, seed: int, horizon: int, device: str, smoke: bool,
    max_tickers: int | None, batch_size: int,
) -> GraphContext:
    """Reproduce ``run_pilot.run_graph_screening`` setup up to the graph-safe P3 backbone."""

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = rp.build_screening_inputs(smoke, max_tickers, "P3", batch_size, horizon=horizon)
    raw = rp.load_and_split_price_data(rp._ROOT / "data" / "processed")
    selected = tuple(inputs.smoke_filter["selected_tickers"])
    raw = rp._select_tickers(raw, selected)
    full_frames = {
        ticker: rp.np_concat_frames(raw.frames[ticker][name] for name in ("train", "val", "test"))
        for ticker in selected
    }
    panel = rp.load_runner_news_panel(
        rp._ROOT / "data" / "features" / "dual_group_news_panel.parquet",
        selected,
        rp._train_news_cutoffs(inputs.manifest),
    )
    graph_store = rp._fit_graph_preprocessors(full_frames)
    graph = build_graph_manifest(full_frames, panel, graph_store, horizon=horizon)
    rp._validate_graph_manifest(graph)
    resolved_device = rp.resolve_graph_device(device)
    graph_pooled = build_pooled_manifest(raw, graph_store, horizon=horizon)
    rp._assert_matched_horizon(horizon, horizon)
    graph_pooled = PooledManifest(
        {split: tuple(attach_news(graph_pooled.samples[split], panel, panel.feature_cols))
         for split in ("train", "val", "test")},
        graph_pooled.exclusions, graph_pooled.ticker_to_id, graph_pooled.preprocessing_hash,
    )
    warm_start = rp.build_graph_bound_p3_warm_start(
        graph_pooled, graph, output_dir, seed, graph_store, epochs=1,
        device=resolved_device, train_batch_size=batch_size,
    )
    graph_safe = rp.build_graph_safe_p3_checkpoint(
        graph_pooled, graph, output_dir, seed, warm_start, graph_store, epochs=1,
        device=resolved_device, train_batch_size=batch_size,
    )
    return GraphContext(graph, graph_store, graph_safe, resolved_device, selected, horizon)


def _new_g1(context: GraphContext) -> GraphAblationModel:
    model = GraphAblationModel.from_p3_checkpoint(
        str(context.graph_safe), use_gnn=True,
        graph_train_end_date=context.graph.train_end_date,
        graph_manifest_hash=context.graph_hash,
    )
    model.configure_positivity(context.store)
    return model


def _evaluate_split(
    model: GraphAblationModel, context: GraphContext, split: str, batch_size: int,
) -> dict[str, float]:
    """Score G1 on one split, matching the pilot's per-snapshot evaluation contract."""

    snapshots = [snapshot for snapshot in context.graph.snapshots if snapshot.split == split]
    if not snapshots:
        raise ValueError(f"graph manifest has no {split} snapshots to evaluate")
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(snapshots), batch_size):
            batch = snapshots[start:start + batch_size]
            predictions, _ = rp._graph_prediction_batch(model, batch, context.device)
            for snapshot, snapshot_predictions in zip(batch, predictions.cpu(), strict=True):
                for node, prediction in zip(snapshot.nodes, snapshot_predictions, strict=True):
                    records.append({
                        "ticker_id": node.ticker_id,
                        "target_date": snapshot.target_date,
                        "prediction_norm": float(prediction),
                        "target_raw": node.y_raw,
                    })
    return {name: float(value) for name, value in evaluate_records(records, context.store)["metrics"].items()}


def train_g1(
    output_dir: Path, seed: int = 42, epochs: int = 10, horizon: int = 5, device: str = "auto",
    smoke: bool = False, max_tickers: int | None = None, batch_size: int = 256,
    train_batch_size: int = 32, val_batch_size: int = 32,
) -> dict[str, Any]:
    """Train the final G1 model, save ``g1_final.pt``, and score the test split."""

    if epochs < 1 or epochs > 10:
        raise ValueError("G1 epochs must be between 1 and 10 (screening/confirmation budget)")
    output_dir = Path(output_dir)
    context = build_graph_context(output_dir, seed, horizon, device, smoke, max_tickers, batch_size)
    model = _new_g1(context)
    rp._seed_graph_device(seed, context.device)
    model.to(context.device)
    train_snapshots = [snapshot for snapshot in context.graph.snapshots if snapshot.split == "train"]
    if not train_snapshots:
        raise ValueError("graph manifest has no train snapshots")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, weight_decay=1e-5)
    train_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        loss_sum = torch.zeros((), device=context.device)
        snapshot_count = 0
        for start in range(0, len(train_snapshots), train_batch_size):
            batch = train_snapshots[start:start + train_batch_size]
            optimizer.zero_grad()
            predictions, targets = rp._graph_prediction_batch(model, batch, context.device)
            loss = rp._mean_snapshot_mse(predictions, targets)
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.detach() * len(batch)
            snapshot_count += len(batch)
        epoch_loss = (loss_sum / snapshot_count).item()
        train_losses.append(epoch_loss)
        print(f"  [G1] epoch {epoch + 1}/{epochs}  train_mse={epoch_loss:.6e}")

    checkpoint_path = output_dir / "g1_final.pt"
    torch.save({
        "config": "G1",
        "model_state": model.state_dict(),
        "seed": seed,
        "horizon": horizon,
        "epochs": epochs,
        "graph_train_end_date": context.graph.train_end_date,
        "graph_manifest_hash": context.graph_hash,
        "graph_safe_p3": str(context.graph_safe),
        "selected_tickers": list(context.selected),
        "train_losses": train_losses,
    }, checkpoint_path)

    val_metrics = _evaluate_split(model, context, "val", val_batch_size)
    test_metrics = _evaluate_split(model, context, "test", val_batch_size)
    payload = {
        "config": "G1",
        "role": "final_proposed_model",
        "seed": seed,
        "horizon": horizon,
        "epochs": epochs,
        "checkpoint": str(checkpoint_path),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "g1_final_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_metrics("G1 (final) validation", val_metrics)
    _print_metrics("G1 (final) TEST", test_metrics)
    print(f"\nSaved G1 checkpoint -> {checkpoint_path}")
    print(f"Saved metrics       -> {output_dir / 'g1_final_metrics.json'}")
    return payload


def infer_g1(
    checkpoint_path: Path, device: str = "auto", val_batch_size: int = 32,
) -> dict[str, float]:
    """Load a saved G1 checkpoint and score the test split. Requires the project dataset."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"G1 checkpoint not found: {checkpoint_path}\n"
            "Run `python reproduce.py train` once to produce it."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != "G1":
        raise ValueError("checkpoint is not a G1 final-model checkpoint")
    seed = int(checkpoint["seed"])
    horizon = int(checkpoint["horizon"])
    max_tickers = len(checkpoint.get("selected_tickers") or []) or None
    context = build_graph_context(
        checkpoint_path.parent, seed, horizon, device, smoke=False,
        max_tickers=max_tickers, batch_size=256,
    )
    model = _new_g1(context)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(context.device)
    test_metrics = _evaluate_split(model, context, "test", val_batch_size)
    _print_metrics("G1 (final) TEST", test_metrics)
    return test_metrics


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title} metrics:")
    for name in _METRIC_ORDER:
        value = metrics[name]
        suffix = "%" if name == "directional_accuracy" else ""
        print(f"  {name:<22} {value:.6f}{suffix}")
