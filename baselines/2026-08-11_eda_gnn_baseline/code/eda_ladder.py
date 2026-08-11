"""EDA-recommended GNN ablation ladder E0 -> E1 -> E2 -> E3 (+ E3off / G1corr controls).

One leakage-safe basis (5 EDA features, graph-bound train, same val/test observations as the pilot):
  * E0     = HAR pooled linear regression (3 HAR features).
  * E1     = price LSTM, HAR + MarketPK (4 features, no graph).
  * E2     = price LSTM, HAR + MarketPK + volume_zscore_20 (5 features, no graph).
  * E3     = price graph model, 5 features + directed volume->PK Top-5 edge (recommended config).
  * E3off  = the trained E3 read out with message passing disabled (nested "remove the graph").
  * G1corr = price graph model, 5 features + correlation kNN-8 edge (controlled current-G1 edge).

Run (GPU venv):  python .../code/eda_ladder.py <TS> [device]
Writes results/eda_gnn_seed{seed}_<TS>/h5/<rung>/ with metrics + per-observation test dumps.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

R = str(Path(__file__).resolve().parents[3])
CODE = str(Path(__file__).resolve().parent)
PILOT = str(Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code")
for _p in (CODE, PILOT, R):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data import (  # noqa: E402
    PooledManifest, PooledSample, build_masked_graph_manifest, build_pooled_manifest,
    load_and_split_price_data,
)
from edges import build_vol2pk_adjacency, swap_adjacency  # noqa: E402
from eda_model import PriceGraphModel  # noqa: E402
from features import EXTRA_FEATURE_COLUMNS, augment_split_frames, build_extended_store  # noqa: E402
from models import PooledPriceLSTM  # noqa: E402
from run_pilot import (  # noqa: E402
    _ManifestDataset, _mean_snapshot_mse, _stacked_snapshot_inputs, _write_graph_predictions,
    _write_json, resolve_graph_device, run_har_reference,
)
from train import _set_seed, evaluate_records  # noqa: E402

HORIZON = 5
SEQ = 22
EPOCHS = 20
SEEDS = (42, 123, 2026)
VOL2PK_TOP_K = 5
CORR_TOP_K = 8
GRAPH_BATCH = 32
HIDDEN = 64
DROPOUT = 0.2
_PROCESSED = Path(R) / "data" / "processed"
_PRICE_DIR = Path(R) / "data" / "raw" / "prices"
_METRIC_KEYS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


def _log(stamp: Path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    with stamp.with_suffix(".progress").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _slice_samples(samples: tuple[PooledSample, ...], width: int) -> tuple[PooledSample, ...]:
    """Narrow each sample's price features to the first ``width`` columns (nested ladder slice)."""

    sliced = []
    for sample in samples:
        sliced.append(PooledSample(
            key=sample.key, x_price_raw=np.ascontiguousarray(sample.x_price_raw[:, :width]),
            x_news=sample.x_news, news_mask=sample.news_mask, y_raw=sample.y_raw,
            y_model_raw=sample.y_model_raw, y_eval_raw=sample.y_eval_raw, input_dates=sample.input_dates,
        ))
    return tuple(sliced)


def _pooled_loader(samples, store, device: torch.device, batch: int = 256) -> DataLoader:
    return DataLoader(_ManifestDataset(samples, store), batch_size=batch, shuffle=False,
                      num_workers=0, pin_memory=device.type == "cuda")


def _present_obs(snapshots) -> set[tuple[int, str]]:
    obs: set[tuple[int, str]] = set()
    for snapshot in snapshots:
        presence = (snapshot.presence_mask if snapshot.presence_mask is not None
                    else np.ones(len(snapshot.nodes), dtype=np.int8))
        for index, node in enumerate(snapshot.nodes):
            if presence[index]:
                obs.add((int(node.ticker_id), snapshot.target_date))
    return obs


def build_basis(stamp: Path):
    """Seed-independent: augmented features, 5-feature manifest, corr + vol2pk masked graphs."""

    raw = load_and_split_price_data(_PROCESSED)
    augmented = augment_split_frames(raw, _PRICE_DIR)
    store = build_extended_store(augmented, EXTRA_FEATURE_COLUMNS)
    pooled = build_pooled_manifest(augmented, store, seq_length=SEQ, horizon=HORIZON)
    _log(stamp, "pooled 5-feature manifest built; building masked corr knn-8 graph ...")
    graph_corr = build_masked_graph_manifest(pooled, store, adjacency="knn", top_k=CORR_TOP_K)
    vol2pk = build_vol2pk_adjacency(augmented, graph_corr.ticker_to_id, graph_corr.train_end_date,
                                    top_k=VOL2PK_TOP_K)
    graph_vol2pk = swap_adjacency(graph_corr, vol2pk)
    allowed = tuple(s for s in pooled.samples["train"] if s.key.target_date <= graph_corr.train_end_date)
    if not allowed:
        raise ValueError("graph-bound train set is empty")
    # One-basis invariant: the graph val/test present observations equal the pooled val/test samples.
    for split in ("val", "test"):
        graph_obs = _present_obs([s for s in graph_corr.snapshots if s.split == split])
        pooled_obs = {(int(s.key.ticker_id), s.key.target_date) for s in pooled.samples[split]}
        if graph_obs != pooled_obs:
            raise ValueError(f"{split} observation set differs between graph and pooled manifests")
    edge_count = int((np.asarray(vol2pk) != 0).sum())
    _log(stamp, f"basis ready: snapshots={len(graph_corr.snapshots)} train_samples={len(allowed)} "
                f"val_obs={len(pooled.samples['val'])} test_obs={len(pooled.samples['test'])} "
                f"vol2pk_edges={edge_count}")
    return {"pooled": pooled, "store": store, "graph_corr": graph_corr, "graph_vol2pk": graph_vol2pk,
            "allowed": allowed, "vol2pk_edges": edge_count}


def _har_predictions(results_path: Path) -> list[dict[str, Any]]:
    """Convert a run_har_reference results.json into per-observation DM rows."""

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    keys = payload["ordered_validation_keys"]  # "ticker_id:target_date" strings
    rows = []
    for key, target, prediction in zip(keys, payload["targets_raw"], payload["predictions_raw"], strict=True):
        ticker_id, target_date = key.split(":", 1)
        rows.append({"ticker_id": int(ticker_id), "target_date": str(target_date),
                     "target_raw": float(target), "prediction_raw": float(prediction)})
    return rows


def run_e0(pooled: PooledManifest, allowed, store, out: Path) -> dict[str, Any]:
    """E0: pooled HAR regression on the 3 HAR features, scored on val and test observations."""

    train3 = _slice_samples(allowed, 3)
    val3 = _slice_samples(pooled.samples["val"], 3)
    test3 = _slice_samples(pooled.samples["test"], 3)
    val_manifest = PooledManifest({"train": train3, "val": val3, "test": test3},
                                  pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash)
    test_manifest = PooledManifest({"train": train3, "val": test3, "test": test3},
                                   pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash)
    val_path = run_har_reference(val_manifest, store, out / "val")
    test_path = run_har_reference(test_manifest, store, out / "test")
    val = json.loads(val_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    _write_json(out / "predictions_test.json", _har_predictions(test_path))
    return {"validation_metrics": val["validation_metrics"], "test_metrics": test["validation_metrics"]}


def _train_pooled_lstm(width: int, allowed, pooled, store, out: Path, seed: int,
                       device: torch.device) -> dict[str, Any]:
    """E1/E2: standalone price LSTM at ``width`` features; 20 epochs, best-val checkpoint."""

    _set_seed(seed)
    train_loader = _pooled_loader(_slice_samples(allowed, width), store, device)
    val_loader = _pooled_loader(_slice_samples(pooled.samples["val"], width), store, device)
    test_loader = _pooled_loader(_slice_samples(pooled.samples["test"], width), store, device)
    model = PooledPriceLSTM(width, hidden_dim=HIDDEN, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            price = batch["x_price"].to(device, dtype=torch.float32)
            target = batch["y_norm"].to(device, dtype=torch.float32)
            loss = criterion(model(price), target)
            if not torch.isfinite(loss):
                raise ValueError("non-finite pooled training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss = _pooled_val_loss(model, val_loader, criterion, device)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val_eval, _ = _pooled_eval(model, val_loader, store, device)
    test_eval, test_records = _pooled_eval(model, test_loader, store, device)
    out.mkdir(parents=True, exist_ok=True)
    _write_graph_predictions(test_records, test_eval, out / "predictions_test.json")
    return {"validation_metrics": val_eval["metrics"], "test_metrics": test_eval["metrics"]}


def _pooled_eval(model, loader, store, device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one pooled loader; return evaluation plus the ordered (id, date) records for a DM dump."""

    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            price = batch["x_price"].to(device, dtype=torch.float32)
            predictions = model(price).detach().cpu().numpy()
            ticker_ids = np.asarray(batch["ticker_id"]).astype(np.int64)
            targets_raw = np.asarray(batch["y_raw"]).astype(float)
            for ticker_id, target_date, prediction, target_raw in zip(
                ticker_ids, batch["target_date"], predictions, targets_raw, strict=True
            ):
                records.append({"ticker_id": int(ticker_id), "target_date": str(target_date),
                                "prediction_norm": float(prediction), "target_raw": float(target_raw)})
    return evaluate_records(records, store), records


def _pooled_val_loss(model, loader, criterion, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            price = batch["x_price"].to(device, dtype=torch.float32)
            target = batch["y_norm"].to(device, dtype=torch.float32)
            total += float(criterion(model(price), target).item()) * int(target.shape[0])
            count += int(target.shape[0])
    if count == 0:
        raise ValueError("empty validation loader")
    return total / count


def _stack_price(batch, device):
    return torch.from_numpy(np.stack([snapshot.x_price for snapshot in batch])).to(device)


def _graph_forward(model, batch, device, apply_mp: bool):
    adjacency, presence, ticker_ids, targets = _stacked_snapshot_inputs(batch, device)
    predictions = model(_stack_price(batch, device), adjacency, ticker_ids, presence,
                        apply_message_passing=apply_mp)
    resolved = presence if presence is not None else torch.ones_like(targets, dtype=torch.bool)
    return predictions, targets, resolved


def _graph_eval(model, snapshots, store, device, apply_mp: bool = True):
    model.eval()
    records: list[dict[str, Any]] = []
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(snapshots), GRAPH_BATCH):
            batch = snapshots[start:start + GRAPH_BATCH]
            predictions, targets, resolved = _graph_forward(model, batch, device, apply_mp)
            total += _mean_snapshot_mse(predictions, targets, resolved).item() * len(batch)
            count += len(batch)
            for snapshot, pred, present in zip(batch, predictions.cpu(), resolved.cpu(), strict=True):
                for index, (node, value) in enumerate(zip(snapshot.nodes, pred, strict=True)):
                    if not bool(present[index]):
                        continue
                    records.append({"ticker_id": node.ticker_id, "target_date": snapshot.target_date,
                                    "prediction_norm": float(value), "target_raw": node.y_raw})
    return total / count, records, evaluate_records(records, store)


def _train_graph(graph, store, num_tickers, out: Path, seed: int, device: torch.device):
    """E3 / G1corr: train PriceGraphModel end-to-end; 20 epochs, best-val checkpoint."""

    _set_seed(seed)
    width = int(graph.snapshots[0].x_price.shape[-1])
    model = PriceGraphModel(width, num_tickers, use_gnn=True, hidden_dim=HIDDEN, dropout=DROPOUT)
    model.configure_positivity(store).to(device)
    train_snaps = [s for s in graph.snapshots if s.split == "train"]
    val_snaps = [s for s in graph.snapshots if s.split == "val"]
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_snaps))
        for start in range(0, len(order), GRAPH_BATCH):
            batch = [train_snaps[i] for i in order[start:start + GRAPH_BATCH]]
            optimizer.zero_grad()
            predictions, targets, resolved = _graph_forward(model, batch, device, True)
            loss = _mean_snapshot_mse(predictions, targets, resolved)
            if not torch.isfinite(loss):
                raise ValueError("non-finite graph training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss, _, _ = _graph_eval(model, val_snaps, store, device, apply_mp=True)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def run_graph_rung(graph, store, num_tickers, out: Path, seed: int, device: torch.device,
                   emit_graph_off: bool) -> dict[str, Any]:
    model = _train_graph(graph, store, num_tickers, out, seed, device)
    val_snaps = [s for s in graph.snapshots if s.split == "val"]
    test_snaps = [s for s in graph.snapshots if s.split == "test"]
    passes = [("on", True)] + ([("off", False)] if emit_graph_off else [])
    result: dict[str, Any] = {}
    for name, apply_mp in passes:
        _, _, val_eval = _graph_eval(model, val_snaps, store, device, apply_mp=apply_mp)
        _, test_records, test_eval = _graph_eval(model, test_snaps, store, device, apply_mp=apply_mp)
        target_dir = out if apply_mp else out.parent / (out.name + "_off")
        target_dir.mkdir(parents=True, exist_ok=True)
        _write_graph_predictions(test_records, test_eval, target_dir / "predictions_test.json")
        result[name] = {"validation_metrics": val_eval["metrics"], "test_metrics": test_eval["metrics"]}
    return result


def run_seed(basis, out: Path, seed: int, device: torch.device, stamp: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pooled, store, allowed = basis["pooled"], basis["store"], basis["allowed"]
    num_tickers = max(pooled.ticker_to_id.values()) + 1
    _log(stamp, f"seed={seed} E0 (HAR)")
    e0 = run_e0(pooled, allowed, store, out / "E0")
    _log(stamp, f"seed={seed} E1 (HAR+MarketPK)")
    e1 = _train_pooled_lstm(4, allowed, pooled, store, out / "E1", seed, device)
    _log(stamp, f"seed={seed} E2 (HAR+MarketPK+vol_z)")
    e2 = _train_pooled_lstm(5, allowed, pooled, store, out / "E2", seed, device)
    _log(stamp, f"seed={seed} E3 (+ directed vol2pk graph)")
    e3 = run_graph_rung(basis["graph_vol2pk"], store, num_tickers, out / "E3", seed, device,
                        emit_graph_off=True)
    _log(stamp, f"seed={seed} G1corr (correlation knn-8 edge control)")
    g1 = run_graph_rung(basis["graph_corr"], store, num_tickers, out / "G1corr", seed, device,
                        emit_graph_off=False)
    ladder = {"seed": seed, "horizon": HORIZON, "epochs": EPOCHS, "vol2pk_top_k": VOL2PK_TOP_K,
              "corr_top_k": CORR_TOP_K, "vol2pk_edges": basis["vol2pk_edges"],
              "rungs": {"E0": e0, "E1": e1, "E2": e2, "E3": e3["on"], "E3off": e3["off"],
                        "G1corr": g1["on"]}}
    _write_json(out / "ladder_metrics.json", ladder)
    _log(stamp, f"seed={seed} DONE")


def main(ts: str, device_name: str = "cuda") -> None:
    stamp = Path(R) / "temp" / f"eda_ladder_{ts}_h{HORIZON}"
    device = resolve_graph_device(device_name)
    _log(stamp, f"device={device} building seed-independent basis ...")
    basis = build_basis(stamp)
    for seed in SEEDS:
        out = Path(R) / "results" / f"eda_gnn_seed{seed}_{ts}" / f"h{HORIZON}"
        t0 = time.perf_counter()
        run_seed(basis, out, seed, device, stamp)
        _log(stamp, f"seed={seed} elapsed {time.perf_counter() - t0:.1f}s")
    stamp.with_suffix(".ALL_DONE").write_text("done", encoding="utf-8")
    _log(stamp, "ALL_DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cuda")
