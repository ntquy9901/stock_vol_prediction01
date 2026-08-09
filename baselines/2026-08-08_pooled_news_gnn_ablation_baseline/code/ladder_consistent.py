"""Consistent-basis Track-B ablation ladder P0 -> P1 -> P2 -> P3 -> G1 (single source of truth).

Every rung is trained and evaluated on ONE basis so the ablation is truly nested:
  * masked manifest (k-NN-8 sparse adjacency), leakage-safe graph-bound train set
    (``target_date <= graph.train_end_date``), the SAME per-ticker scalers (``graph_store``),
    the SAME held-out validation / test observations, positivity floor, horizon 5.

Rungs (component toggles on the same backbone-training regime, screening config dropout 0.2):
  * P0 = pooled HAR regression evaluated on the same observations.
  * P1 = price-only LSTM (news OFF, gate OFF, graph OFF).
  * P2 = price + news LSTM (gate OFF, graph OFF).
  * P3 = price + news + per-ticker gate, graph OFF -- the SAME trained G1 model read out with the
    message-passing residual disabled (this replaces the old separate "G0" row).
  * G1 = P3 + GAT/message-passing (masked k-NN-8 adjacency).

Because P3 is the identical trained G1 model with only the graph residual removed, 'remove the GAT
from G1 = P3' is exact by construction; the per-seed ``nesting_check.json`` records that the two
graph-off passes are bit-identical and that the graph residual is non-trivial.

Run (GPU venv):  python .../code/ladder_consistent.py <TS> [device]
Writes per-seed run dirs ``results/ladder_consistent_seed{seed}_<TS>/h5``.
Aggregate with ``docs/reports/ladder_consistent_dump.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

R = str(Path(__file__).resolve().parents[3])
CODE = str(Path(__file__).resolve().parent)
for _p in (CODE, R):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data import PooledManifest, attach_news, build_masked_graph_manifest  # noqa: E402
from models import GraphAblationModel, PooledPriceLSTM, PooledPriceNewsLSTM  # noqa: E402
from run_pilot import (  # noqa: E402
    _ManifestDataset, _ROOT, _adjacency_config, _build_shared_graph_base, _edge_density_stats,
    _mean_snapshot_mse, _run_one_graph_model, _select_tickers, _stacked_snapshot_inputs,
    _train_news_cutoffs, _write_graph_predictions, _write_json, build_graph_bound_p3_warm_start,
    build_graph_safe_p3_checkpoint, build_pooled_manifest, build_screening_inputs,
    load_and_split_price_data, load_runner_news_panel, np_concat_frames, resolve_graph_device,
    run_har_reference,
)
from train import evaluate_by_ticker, run_training  # noqa: E402

HORIZON = 5
BACKBONE_EPOCHS = 5      # frozen P3 backbone (warm-start + 1 safe epoch), dropout 0.2
BACKBONE_DROPOUT = 0.2
POOLED_EPOCHS = 5        # P1/P2 screening depth (run_training caps at 10)
GRAPH_EPOCHS = 15        # G1 message-passing head (frozen backbone) to convergence
BATCH = 256
GRAPH_TRAIN_BATCH = 32
GRAPH_VAL_BATCH = 32
SEEDS = (42, 123, 2026)
ADJACENCY = "knn"
TOP_K = 8
_METRIC_KEYS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


def _log(stamp: Path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with stamp.with_suffix(".progress").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _loader(samples, store, device: torch.device) -> DataLoader:
    return DataLoader(_ManifestDataset(samples, store), batch_size=BATCH, shuffle=False,
                      num_workers=0, pin_memory=device.type == "cuda")


def _manifest_with(pooled: PooledManifest, train, val) -> PooledManifest:
    """A pooled manifest with a swapped train/val sample set (test kept for provenance)."""

    return PooledManifest({"train": tuple(train), "val": tuple(val), "test": pooled.samples["test"]},
                          pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash)


def _present_obs(snapshots) -> set[tuple[int, str]]:
    obs: set[tuple[int, str]] = set()
    for snapshot in snapshots:
        presence = (snapshot.presence_mask if snapshot.presence_mask is not None
                    else np.ones(len(snapshot.nodes), dtype=np.int8))
        for index, node in enumerate(snapshot.nodes):
            if presence[index]:
                obs.add((int(node.ticker_id), snapshot.target_date))
    return obs


def run_har_rung(pooled: PooledManifest, allowed, store, out: Path) -> dict[str, Any]:
    """P0: fit one pooled HAR regression on the graph-bound train set; score val and test."""

    val_path = run_har_reference(_manifest_with(pooled, allowed, pooled.samples["val"]), store, out / "val")
    test_path = run_har_reference(_manifest_with(pooled, allowed, pooled.samples["test"]), store, out / "test")
    val = json.loads(val_path.read_text(encoding="utf-8"))["validation_metrics"]
    test = json.loads(test_path.read_text(encoding="utf-8"))["validation_metrics"]
    return {"validation_metrics": val, "test_metrics": test}


def run_pooled_rung(config_name: str, pooled: PooledManifest, allowed, store, out: Path, seed: int,
                    device: torch.device) -> dict[str, Any]:
    """P1 / P2: train a standalone LSTM on the graph-bound train set; score val (best) and test."""

    loaders = {"train": _loader(allowed, store, device), "val": _loader(pooled.samples["val"], store, device)}
    result_path = run_training(config_name, loaders, store, out, POOLED_EPOCHS, seed)
    val_metrics = json.loads(result_path.read_text(encoding="utf-8"))["validation_metrics"]
    price_dim = int(allowed[0].x_price_raw.shape[1])
    news_dim = int(allowed[0].x_news.shape[1])
    num_tickers = max(pooled.ticker_to_id.values()) + 1
    if config_name == "P1":
        model: torch.nn.Module = PooledPriceLSTM(price_dim)
    else:
        model = PooledPriceNewsLSTM(price_dim, news_dim, num_tickers, use_gate=False)
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model_state"])
    model.to(device)
    test_eval = evaluate_by_ticker(model, _loader(pooled.samples["test"], store, device), store)
    return {"validation_metrics": val_metrics, "test_metrics": test_eval["metrics"]}


def _evaluate_graph_off(model: GraphAblationModel, snapshots, base, store, device: torch.device,
                        batch_size: int) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """Present-node metrics for the trained model with the message-passing residual DISABLED.

    Mirrors ``run_pilot._evaluate_graph_split`` but calls ``apply_graph_head(..., apply_message_
    passing=False)`` on the shared frozen ``base`` -- the pure backbone+head+positivity (P3) path.
    """

    from train import evaluate_records

    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        loss_sum = torch.zeros((), device=device)
        count = 0
        for start in range(0, len(snapshots), batch_size):
            batch = snapshots[start:start + batch_size]
            adjacency, presence_mask, ticker_ids, targets = _stacked_snapshot_inputs(batch, device)
            predictions = model.apply_graph_head(
                torch.stack(base[start:start + batch_size]), adjacency, ticker_ids, presence_mask,
                apply_message_passing=False)
            resolved = presence_mask if presence_mask is not None else torch.ones_like(targets, dtype=torch.bool)
            loss_sum = loss_sum + _mean_snapshot_mse(predictions, targets, resolved) * len(batch)
            count += len(batch)
            for snapshot, snapshot_pred, snapshot_present in zip(batch, predictions.cpu(), resolved.cpu(), strict=True):
                for index, (node, prediction) in enumerate(zip(snapshot.nodes, snapshot_pred, strict=True)):
                    if not bool(snapshot_present[index]):
                        continue
                    records.append({"ticker_id": node.ticker_id, "target_date": snapshot.target_date,
                                    "prediction_norm": float(prediction), "target_raw": node.y_raw})
        split_loss = (loss_sum / count).item()
    return split_loss, records, evaluate_records(records, store)


def run_graph_rungs(pooled: PooledManifest, graph, graph_store, out: Path, seed: int,
                    device: torch.device) -> dict[str, Any]:
    """P3 (graph OFF) + G1 (graph ON) from ONE trained model, with a nesting-identity check."""

    graph_hash = graph.content_hash("train")
    warm = build_graph_bound_p3_warm_start(pooled, graph, out, seed, graph_store,
                                            epochs=max(1, BACKBONE_EPOCHS - 1), device=device,
                                            train_batch_size=BATCH, dropout=BACKBONE_DROPOUT)
    safe = build_graph_safe_p3_checkpoint(pooled, graph, out, seed, warm, graph_store, epochs=1,
                                          device=device, train_batch_size=BATCH, dropout=BACKBONE_DROPOUT)
    g1 = GraphAblationModel.from_p3_checkpoint(str(safe), use_gnn=True,
                                               graph_train_end_date=graph.train_end_date,
                                               graph_manifest_hash=graph_hash)
    g1.to(device)
    shared_base = _build_shared_graph_base(g1, graph, device, GRAPH_TRAIN_BATCH, GRAPH_VAL_BATCH)
    g1_outcome = _run_one_graph_model(g1, graph, graph_store, "G1", GRAPH_EPOCHS, seed, out / "G1",
                                      device, validation_batch_size=GRAPH_VAL_BATCH,
                                      train_batch_size=GRAPH_TRAIN_BATCH, base_cache=shared_base)
    # P3 readout: the identical trained g1 with the graph residual removed.
    val = [snapshot for snapshot in graph.snapshots if snapshot.split == "val"]
    test = [snapshot for snapshot in graph.snapshots if snapshot.split == "test"]
    p3_val_loss, p3_val_records, p3_val_eval = _evaluate_graph_off(
        g1, val, shared_base["val"], graph_store, device, GRAPH_VAL_BATCH)
    p3_test_loss, p3_test_records, p3_test_eval = _evaluate_graph_off(
        g1, test, shared_base["test"], graph_store, device, GRAPH_VAL_BATCH)
    (out / "P3").mkdir(parents=True, exist_ok=True)
    _write_graph_predictions(p3_val_records, p3_val_eval, out / "P3" / "predictions.json")
    _write_graph_predictions(p3_test_records, p3_test_eval, out / "P3" / "predictions_test.json")
    # Nesting evidence: a second graph-off pass is bit-identical (deterministic readout) and the
    # graph residual moves predictions (G1 != P3).
    _, recheck_records, _ = _evaluate_graph_off(g1, val, shared_base["val"], graph_store, device, GRAPH_VAL_BATCH)
    p3_val_pred = np.asarray([row["prediction_norm"] for row in p3_val_records], dtype=float)
    recheck_pred = np.asarray([row["prediction_norm"] for row in recheck_records], dtype=float)
    g1_val_pred = np.asarray([row["prediction_raw"] for row in
                              json.loads((out / "G1" / "predictions.json").read_text(encoding="utf-8"))], dtype=float)
    p3_val_pred_raw = np.asarray(p3_val_eval["predictions_raw"], dtype=float)
    nesting = {
        "graph_off_readout_determinism_max_abs_diff": float(np.abs(p3_val_pred - recheck_pred).max()),
        "graph_effect_val_mean_abs_pred_diff_raw": float(np.abs(g1_val_pred - p3_val_pred_raw).mean()),
        "graph_effect_val_max_abs_pred_diff_raw": float(np.abs(g1_val_pred - p3_val_pred_raw).max()),
        "n_val_obs": len(p3_val_records),
    }
    _write_json(out / "nesting_check.json", nesting)
    p3 = {"validation_metrics": p3_val_eval["metrics"], "test_metrics": p3_test_eval["metrics"],
          "validation_loss": p3_val_loss, "test_loss": p3_test_loss}
    g1_row = {"validation_metrics": g1_outcome["validation_metrics"],
              "test_metrics": g1_outcome.get("test_metrics"),
              "validation_loss": g1_outcome["validation_loss"], "test_loss": g1_outcome.get("test_loss")}
    return {"P3": p3, "G1": g1_row, "nesting_check": nesting,
            "graph_train_hash": graph_hash, "snapshot_count": len(graph.snapshots)}


def run_seed(pooled: PooledManifest, graph, graph_store, allowed, out: Path, seed: int,
             device: torch.device, stamp: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    _log(stamp, f"seed={seed} P0 (HAR)")
    p0 = run_har_rung(pooled, allowed, graph_store, out / "P0")
    _log(stamp, f"seed={seed} P1 (price-only)")
    p1 = run_pooled_rung("P1", pooled, allowed, graph_store, out / "P1", seed, device)
    _log(stamp, f"seed={seed} P2 (price+news)")
    p2 = run_pooled_rung("P2", pooled, allowed, graph_store, out / "P2", seed, device)
    _log(stamp, f"seed={seed} P3+G1 (graph off/on)")
    graph_rungs = run_graph_rungs(pooled, graph, graph_store, out, seed, device)
    ladder = {"seed": seed, "horizon": HORIZON, "adjacency": _adjacency_config(ADJACENCY, TOP_K, 0.7),
              "edge_density": _edge_density_stats(graph),
              "rungs": {"P0": p0, "P1": p1, "P2": p2, "P3": graph_rungs["P3"], "G1": graph_rungs["G1"]},
              "nesting_check": graph_rungs["nesting_check"],
              "graph_train_hash": graph_rungs["graph_train_hash"],
              "snapshot_count": graph_rungs["snapshot_count"]}
    _write_json(out / "ladder_metrics.json", ladder)
    determinism = graph_rungs["nesting_check"]["graph_off_readout_determinism_max_abs_diff"]
    _log(stamp, f"seed={seed} DONE nesting_det={determinism:.2e}")


def build_basis(device: torch.device, stamp: Path):
    """Seed-independent: pooled+news manifest, k-NN-8 masked graph, graph-bound train set."""

    inputs = build_screening_inputs(False, None, "P3", horizon=HORIZON)
    raw = load_and_split_price_data(_ROOT / "data" / "processed")
    selected = tuple(inputs.smoke_filter["selected_tickers"])
    raw = _select_tickers(raw, selected)
    full_frames = {t: np_concat_frames(raw.frames[t][s] for s in ("train", "val", "test")) for t in selected}
    panel = load_runner_news_panel(_ROOT / "data" / "features" / "dual_group_news_panel.parquet",
                                   selected, _train_news_cutoffs(inputs.manifest))
    from run_pilot import _fit_graph_preprocessors
    graph_store = _fit_graph_preprocessors(full_frames)
    pooled = build_pooled_manifest(raw, graph_store, horizon=HORIZON)
    pooled = PooledManifest(
        {s: tuple(attach_news(pooled.samples[s], panel, panel.feature_cols)) for s in ("train", "val", "test")},
        pooled.exclusions, pooled.ticker_to_id, pooled.preprocessing_hash)
    _log(stamp, "pooled+news built; building masked knn-8 manifest ...")
    graph = build_masked_graph_manifest(pooled, graph_store, adjacency=ADJACENCY, top_k=TOP_K)
    allowed = tuple(s for s in pooled.samples["train"] if s.key.target_date <= graph.train_end_date)
    if not allowed:
        raise ValueError("graph-bound train set is empty")
    # One-basis invariant: the masked graph val/test present-node observations are exactly the
    # pooled val/test samples P0-P2 are scored on.
    for split, snap_split in (("val", "val"), ("test", "test")):
        graph_obs = _present_obs([s for s in graph.snapshots if s.split == snap_split])
        pooled_obs = {(int(s.key.ticker_id), s.key.target_date) for s in pooled.samples[split]}
        if graph_obs != pooled_obs:
            raise ValueError(f"{split} observation set differs between graph and pooled manifests")
    _log(stamp, f"basis ready: snapshots={len(graph.snapshots)} train_samples={len(allowed)} "
                f"val_obs={len(pooled.samples['val'])} test_obs={len(pooled.samples['test'])}")
    return pooled, graph, graph_store, allowed


def main(ts: str, device_name: str) -> None:
    stamp = Path(R) / "temp" / f"ladder_{ts}"
    device = resolve_graph_device(device_name)
    _log(stamp, f"device={device} building seed-independent basis ...")
    pooled, graph, graph_store, allowed = build_basis(device, stamp)
    for seed in SEEDS:
        out = Path(R) / "results" / f"ladder_consistent_seed{seed}_{ts}" / "h5"
        t0 = time.perf_counter()
        run_seed(pooled, graph, graph_store, allowed, out, seed, device, stamp)
        _log(stamp, f"seed={seed} elapsed {time.perf_counter() - t0:.1f}s")
    stamp.with_suffix(".ALL_DONE").write_text("done", encoding="utf-8")
    _log(stamp, "ALL_DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cuda")
