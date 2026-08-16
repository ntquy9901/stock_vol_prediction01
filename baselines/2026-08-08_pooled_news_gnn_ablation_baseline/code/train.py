"""Leakage-safe pooled P1-P3 screening training and raw-scale evaluation."""

from __future__ import annotations

import json
import random
from hashlib import sha256
from os import replace
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
    resume_from: Path | str | None = None,
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
    preprocessors = store.to_dict()
    preprocessors_hash = _canonical_hash(preprocessors)
    manifest, manifest_hash = _manifest_payload(loaders, preprocessors_hash)
    state: Mapping[str, Any] | None = None
    if resume_from is not None:
        state = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        _validate_resume_state(state, config_name, seed, manifest_hash, preprocessors_hash)
        if epochs <= int(state["completed_epoch"]):
            raise ValueError("requested epochs must exceed completed_epoch when resuming")

    _set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "preprocessors.json", preprocessors)
    _write_json(out / "sample_manifest.json", manifest)

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
    state_path = out / "training_state.pt"
    best_metrics: dict[str, Any] | None = None
    completed_epoch = 0
    if state is not None:
        completed_epoch = int(state["completed_epoch"])
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        train_losses = [float(value) for value in state["train_losses"]]
        val_losses = [float(value) for value in state["val_losses"]]
        best_loss = float(state["best_loss"])
        best_metrics = state["best_metrics"]
        _restore_rng_state(state["rng_state"])
    _save_training_state(state_path, config_name, seed, completed_epoch, model, optimizer, train_losses,
                         val_losses, best_loss, best_metrics, manifest_hash, preprocessors_hash)
    try:
        for _epoch in range(completed_epoch + 1, epochs + 1):
            model.train()
            loss_total = 0.0
            sample_count = 0
            for batch in loaders["train"]:
                optimizer.zero_grad()
                prediction = _forward(model, batch, device)
                target = batch["y_norm"].to(device=device, dtype=torch.float32,
                                             non_blocking=device.type == "cuda")
                loss = criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_count = int(target.shape[0])
                loss_total += float(loss.item()) * batch_count
                sample_count += batch_count
            if sample_count == 0:
                raise ValueError("training loader must contain at least one sample")
            train_losses.append(loss_total / sample_count)
            val_loss = _normalized_loss(model, loaders["val"], criterion, device)
            val_losses.append(val_loss)
            validation = evaluate_by_ticker(model, loaders["val"], store)
            if val_loss < best_loss:
                best_loss, best_metrics = val_loss, validation
                torch.save({"config_name": config_name, "seed": seed, "model_state": model.state_dict()}, best_path)
            _save_training_state(state_path, config_name, seed, _epoch, model, optimizer, train_losses,
                                 val_losses, best_loss, best_metrics, manifest_hash, preprocessors_hash)
            if _epoch in {5, 10}:
                _plot_losses(train_losses, val_losses, out / f"learning_curve_epoch_{_epoch}.png")
    finally:
        _plot_losses(train_losses, val_losses, out / "learning_curve_partial.png")
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
    non_blocking = device.type == "cuda"
    price = batch["x_price"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if isinstance(model, PooledPriceLSTM):
        return model(price)
    return model(
        price,
        batch["x_news"].to(device=device, dtype=torch.float32, non_blocking=non_blocking),
        batch["news_mask"].to(device=device, non_blocking=non_blocking),
        batch["ticker_id"].to(device=device, dtype=torch.long, non_blocking=non_blocking),
    )


def _normalized_loss(model: nn.Module, loader: Any, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["y_norm"].to(device=device, dtype=torch.float32,
                                         non_blocking=device.type == "cuda")
            loss = criterion(_forward(model, batch, device), target)
            batch_count = int(target.shape[0])
            loss_total += float(loss.item()) * batch_count
            sample_count += batch_count
    if sample_count == 0 or not np.isfinite(loss_total):
        raise ValueError("validation loss must be finite")
    return loss_total / sample_count


def _manifest_payload(
    loaders: Mapping[str, Any], preprocessing_hash: str
) -> tuple[dict[str, Any], str]:
    """Build a compact artifact and a tensor-content resume identity.

    The JSON artifact intentionally records only sample keys and canonical content
    hashes.  The checkpoint identity is derived from those full in-memory tensor
    hashes, rather than from the keys alone.
    """

    splits: dict[str, dict[str, Any]] = {}
    for split, loader in loaders.items():
        ordered_keys = _ordered_keys(loader)
        hashes = _split_content_hashes(loader)
        splits[split] = {"count": len(ordered_keys), "ordered_keys": ordered_keys, "hashes": hashes}
    payload = {"version": 2, "preprocessing_hash": preprocessing_hash, "splits": splits}
    return payload, _canonical_hash(payload)


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


def _save_training_state(
    path: Path, config_name: str, seed: int, completed_epoch: int, model: nn.Module,
    optimizer: torch.optim.Optimizer, train_losses: list[float], val_losses: list[float],
    best_loss: float, best_metrics: dict[str, Any] | None, manifest_hash: str, preprocessors_hash: str,
) -> None:
    torch.save({"config_name": config_name, "seed": seed, "completed_epoch": completed_epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses, "val_losses": val_losses, "best_loss": best_loss,
                "best_metrics": best_metrics, "manifest_hash": manifest_hash,
                "preprocessors_hash": preprocessors_hash, "rng_state": _rng_state()}, path)


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _ordered_keys(loader: Any) -> list[dict[str, Any]]:
    samples = getattr(getattr(loader, "dataset", None), "samples", None)
    if samples is not None:
        return [
            {
                "ticker_id": int(sample.key.ticker_id),
                "ticker": str(sample.key.ticker),
                "target_date": str(sample.key.target_date),
                "input_dates": list(sample.input_dates),
            }
            for sample in samples
        ]

    keys: list[dict[str, Any]] = []
    for batch in loader:
        ticker_ids = _as_numpy(batch["ticker_id"]).astype(np.int64)
        for ticker_id, target_date in zip(ticker_ids, batch["target_date"], strict=True):
            keys.append({
                "ticker_id": int(ticker_id), "ticker": str(ticker_id),
                "target_date": str(target_date), "input_dates": [],
            })
    return keys


def _split_content_hashes(loader: Any) -> dict[str, str]:
    values: dict[str, list[str]] = {
        "price": [], "news": [], "mask": [], "raw_targets": [], "model_targets": [],
    }
    for batch in loader:
        values["price"].extend(_canonical_sample_hashes(batch["x_price"]))
        values["news"].extend(_canonical_sample_hashes(batch["x_news"]))
        values["mask"].extend(_canonical_sample_hashes(batch["news_mask"]))
        values["raw_targets"].extend(_canonical_sample_hashes(batch["y_raw"]))
        values["model_targets"].extend(_canonical_sample_hashes(batch["y_norm"]))
    return {name: _canonical_hash(hashes) for name, hashes in values.items()}


def _canonical_sample_hashes(value: Any) -> list[str]:
    array = _as_numpy(value)
    return [_canonical_tensor_hash(sample) for sample in array]


def _canonical_tensor_hash(value: Any) -> str:
    array = np.ascontiguousarray(_as_numpy(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = sha256()
    digest.update(header)
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_resume_state(
    state: Mapping[str, Any],
    config_name: str,
    seed: int,
    manifest_hash: str,
    preprocessors_hash: str,
) -> None:
    if state.get("config_name") != config_name:
        raise ValueError("resume config_name does not match")
    if state.get("seed") != seed:
        raise ValueError("resume seed does not match")
    if state.get("manifest_hash") != manifest_hash:
        raise ValueError("resume manifest hash does not match")
    if state.get("preprocessors_hash") != preprocessors_hash:
        raise ValueError("resume preprocessor hash does not match")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace(temporary, path)
