"""Leakage-safe pooled P1-P3 screening training and raw-scale evaluation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import RandomSampler

from models import PooledPriceLSTM, PooledPriceNewsLSTM
from scaling import PreprocessorStore
from src.common.evaluation import evaluate_predictions, qlike_loss


def evaluate_records(
    records: Iterable[Mapping[str, Any]], store: PreprocessorStore, epsilon: float = 1e-8
) -> dict[str, Any]:
    """Evaluate normalized predictions against untouched raw targets by explicit ticker ID."""

    values = list(records)
    if not values:
        raise ValueError("evaluation requires at least one record")
    ticker_ids = np.asarray([record["ticker_id"] for record in values], dtype=np.int64)
    predictions_norm = np.asarray([record["prediction_norm"] for record in values], dtype=float)
    targets_raw = np.asarray([record["target_raw"] for record in values], dtype=float)
    if not np.isfinite(predictions_norm).all() or not np.isfinite(targets_raw).all():
        raise ValueError("evaluation records must be finite")
    predictions_raw = store.inverse_targets(ticker_ids, predictions_norm)
    nonpositive_rate = float(np.mean(predictions_raw <= 0.0))
    if nonpositive_rate > 0.01:
        raise ValueError(f"nonpositive prediction rate {nonpositive_rate:.2%} exceeds 1%")

    # Reuse the shared five raw-scale metrics, then replace its flattened direction metric.
    metrics = evaluate_predictions(targets_raw, predictions_raw)
    metrics["qlike"] = float(qlike_loss(targets_raw, predictions_raw, epsilon=epsilon))
    grouped: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
    for record, prediction, target in zip(values, predictions_raw, targets_raw, strict=True):
        grouped[int(record["ticker_id"])].append((str(record["target_date"]), float(target), float(prediction)))
    per_ticker: dict[str, dict[str, float]] = {}
    directional_values: list[float] = []
    directional_weights: list[int] = []
    for ticker_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0])
        if len(rows) < 2:
            continue
        targets = np.asarray([row[1] for row in rows])
        predictions = np.asarray([row[2] for row in rows])
        score = float(np.mean(np.sign(np.diff(targets)) == np.sign(np.diff(predictions))) * 100)
        per_ticker[str(ticker_id)] = {"directional_accuracy": score, "observations": float(len(rows))}
        directional_values.append(score)
        directional_weights.append(len(rows) - 1)
    if not directional_values:
        raise ValueError("directional accuracy requires at least one ticker with two targets")
    metrics["directional_accuracy"] = float(np.mean(directional_values))
    return {
        "targets_raw": targets_raw.tolist(),
        "predictions_raw": predictions_raw.tolist(),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "nonpositive_prediction_rate": nonpositive_rate,
        "eligible_ticker_count": len(directional_values),
        "directional_accuracy_weighted": float(np.average(directional_values, weights=directional_weights)),
        "per_ticker": per_ticker,
    }


def evaluate_by_ticker(
    model: nn.Module, loader: Any, store: PreprocessorStore, epsilon: float = 1e-8
) -> dict[str, Any]:
    """Run one loader and evaluate its stored raw targets without batch-position inference."""

    device = next(model.parameters()).device
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            predictions = _forward(model, batch, device).detach().cpu().numpy()
            ticker_ids = _as_numpy(batch["ticker_id"]).astype(np.int64)
            targets_raw = _as_numpy(batch["y_raw"]).astype(float)
            dates = batch["target_date"]
            for ticker_id, target_date, prediction, target_raw in zip(
                ticker_ids, dates, predictions, targets_raw, strict=True
            ):
                records.append(
                    {
                        "ticker_id": int(ticker_id),
                        "target_date": str(target_date),
                        "prediction_norm": float(prediction),
                        "target_raw": float(target_raw),
                    }
                )
    return evaluate_records(records, store, epsilon=epsilon)


def run_training(
    config_name: str,
    loaders: Mapping[str, Any],
    store: PreprocessorStore,
    output_dir: Path | str,
    epochs: int,
    seed: int,
) -> Path:
    """Train P1-P3 with validation-only screening selection and durable local artifacts."""

    if epochs < 1 or epochs > 10:
        raise ValueError("screening epochs must be between 1 and 10")
    if config_name not in {"P1", "P2", "P3"}:
        raise ValueError("config_name must be P1, P2, or P3")
    if not {"train", "val"}.issubset(loaders):
        raise ValueError("loaders must contain train and val")
    for name, loader in loaders.items():
        if isinstance(loader.sampler, RandomSampler):
            raise ValueError(f"{name} loader must use shuffle=False")
    _set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "preprocessors.json", store.to_dict())
    _write_json(out / "sample_manifest.json", _manifest_payload(loaders))

    first_batch = next(iter(loaders["train"]))
    price_dim = int(first_batch["x_price"].shape[-1])
    news_dim = int(first_batch["x_news"].shape[-1])
    num_tickers = max(store.preprocessors) + 1
    if config_name == "P1":
        model: nn.Module = PooledPriceLSTM(price_dim)
    else:
        model = PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=config_name == "P3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.MSELoss()
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_loss = float("inf")
    best_path = out / "best.pt"
    best_metrics: dict[str, Any] | None = None
    for _epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            optimizer.zero_grad()
            prediction = _forward(model, batch, device)
            target = batch["y_norm"].to(device=device, dtype=torch.float32)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        train_losses.append(float(np.mean(losses)))
        val_loss = _normalized_loss(model, loaders["val"], criterion, device)
        val_losses.append(val_loss)
        validation = evaluate_by_ticker(model, loaders["val"], store)
        if val_loss < best_loss:
            best_loss, best_metrics = val_loss, validation
            torch.save({"config_name": config_name, "seed": seed, "model_state": model.state_dict()}, best_path)
        if _epoch in {5, 10}:
            _plot_losses(train_losses, val_losses, out / f"learning_curve_epoch_{_epoch}.png")
    curve_name = "learning_curve_partial.png" if epochs < 5 else "learning_curve_final.png"
    _plot_losses(train_losses, val_losses, out / curve_name)
    result_path = out / "results.json"
    _write_json(
        result_path,
        {
            "config_name": config_name,
            "seed": seed,
            "epochs": epochs,
            "train_losses": train_losses,
            "validation_losses": val_losses,
            "best_validation_loss": best_loss,
            "validation_metrics": best_metrics["metrics"] if best_metrics else {},
        },
    )
    return result_path


def _forward(model: nn.Module, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    price = batch["x_price"].to(device=device, dtype=torch.float32)
    if isinstance(model, PooledPriceLSTM):
        return model(price)
    return model(
        price,
        batch["x_news"].to(device=device, dtype=torch.float32),
        batch["news_mask"].to(device=device),
        batch["ticker_id"].to(device=device, dtype=torch.long),
    )


def _normalized_loss(model: nn.Module, loader: Any, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            target = batch["y_norm"].to(device=device, dtype=torch.float32)
            losses.append(float(criterion(_forward(model, batch, device), target).item()))
    if not losses or not np.isfinite(losses).all():
        raise ValueError("validation loss must be finite")
    return float(np.mean(losses))


def _manifest_payload(loaders: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split, loader in loaders.items():
        dataset = loader.dataset
        records = getattr(dataset, "records", [])
        payload[split] = [
            {"ticker_id": int(row["ticker_id"]), "target_date": str(row["target_date"])}
            for row in records
        ]
    return payload


def _plot_losses(train_losses: list[float], val_losses: list[float], path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axis.plot(range(1, len(val_losses) + 1), val_losses, label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized MSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
