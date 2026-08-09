"""Quick experiment: GAT on the price-only P1 backbone (news OFF, gate OFF) vs classical HAR.

Model = the pooled price LSTM backbone (P1 configuration, 3 HAR-scale node features) + k-NN-8
masked message passing (the same graph residual G1 uses), but WITHOUT the news branch and WITHOUT
the per-ticker gate.  This is 'GAT on P1' = P1 + graph -- leaner than G1 (graph on the full
news+gate P3 backbone).

Same consistent basis as ``ladder_consistent.py`` (reuses ``build_basis``): masked knn-8 manifest,
leakage-safe graph-bound train (``target_date <= graph.train_end_date``), shared per-ticker
scalers, positivity floor, identical val/test observations, horizon 5.  P0 (classical HAR) is
recomputed on that exact basis so the comparison is apples-to-apples with the ladder's P0.

Run (GPU venv):  python .../code/gat_price_quick.py <TS> [device] [seed[,seed,...]]
Writes ``results/gat_price_quick_seed{seed}_<TS>/h5`` and a combined ``gat_price_quick_<TS>.json``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

R = str(Path(__file__).resolve().parents[3])
CODE = str(Path(__file__).resolve().parent)
for _p in (CODE, R):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from ladder_consistent import (  # noqa: E402
    ADJACENCY, BACKBONE_DROPOUT, GRAPH_TRAIN_BATCH, GRAPH_VAL_BATCH, HORIZON, POOLED_EPOCHS,
    TOP_K, build_basis, run_har_rung,
)
from models import PooledPriceLSTM, PriceGraphAblationModel  # noqa: E402
from run_pilot import (  # noqa: E402
    _adjacency_config, _build_shared_graph_base, _canonical_sample_hash, _edge_density_stats,
    _pooled_training_batches, _run_one_graph_model, _seed_graph_device, _write_json,
    resolve_graph_device,
)

GRAPH_EPOCHS = 20   # user-approved for this quick run
_METRIC_KEYS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
# Lower-is-better for MSE/RMSE/MAE/QLIKE; higher-is-better for R2 and directional accuracy.
_HIGHER_IS_BETTER = {"r2", "directional_accuracy"}


def _log(stamp: Path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with stamp.with_suffix(".progress").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def build_graph_safe_price_checkpoint(
    allowed, graph, out: Path, seed: int, store, epochs: int, device: torch.device,
    batch_size: int, dropout: float, price_dim: int,
) -> Path:
    """Train a price-only P1 LSTM on the leakage-safe graph-bound train set; attest provenance.

    Mirrors the graph-safe P3 checkpoint contract (records the graph train boundary + manifest
    hash so ``PriceGraphAblationModel.from_p1_checkpoint`` accepts it), but the backbone is the
    price-only ``PooledPriceLSTM`` (no news, no gate).
    """

    if not allowed:
        raise ValueError("no graph-bound train samples for the price backbone")
    _seed_graph_device(seed, device)
    model = PooledPriceLSTM(price_dim, dropout=dropout).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    for _ in range(epochs):
        for x_price, _x_news, _news_mask, _ticker_ids, targets in _pooled_training_batches(
            allowed, store, device, batch_size,
        ):
            optimizer.zero_grad()
            prediction = model(x_price)
            torch.nn.functional.mse_loss(prediction, targets).backward()
            optimizer.step()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "graph_safe_p1.pt"
    torch.save({
        "config_name": "P1",
        "model_state": model.state_dict(),
        "graph_safe": True,
        "seed": seed,
        "max_training_target_date": max(sample.key.target_date for sample in allowed),
        "graph_train_end_date": graph.train_end_date,
        "training_sample_count": len(allowed),
        "training_sample_hash": _canonical_sample_hash(allowed),
        "graph_manifest_hash": graph.content_hash("train"),
    }, path)
    return path


def _compare(model_metrics: dict[str, float], har_metrics: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Per-metric GAT-price vs HAR verdict on identical observations."""

    verdicts: dict[str, dict[str, Any]] = {}
    for key in _METRIC_KEYS:
        gat = float(model_metrics[key])
        har = float(har_metrics[key])
        higher_better = key in _HIGHER_IS_BETTER
        if gat == har:
            verdict = "tie"
        elif (gat > har) == higher_better:
            verdict = "beats_HAR"
        else:
            verdict = "worse_than_HAR"
        verdicts[key] = {"gat_price": gat, "har": har, "delta_gat_minus_har": gat - har,
                         "higher_is_better": higher_better, "verdict": verdict}
    return verdicts


def run_seed(pooled, graph, graph_store, allowed, out: Path, seed: int, device: torch.device,
             stamp: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    price_dim = int(allowed[0].x_price_raw.shape[1])
    num_tickers = max(pooled.ticker_to_id.values()) + 1
    _log(stamp, f"seed={seed} P0 (HAR) on the shared basis")
    p0 = run_har_rung(pooled, allowed, graph_store, out / "P0")
    _log(stamp, f"seed={seed} training price-only P1 backbone ({POOLED_EPOCHS} epochs)")
    checkpoint = build_graph_safe_price_checkpoint(
        allowed, graph, out / "backbone", seed, graph_store, POOLED_EPOCHS, device,
        batch_size=256, dropout=BACKBONE_DROPOUT, price_dim=price_dim)
    model = PriceGraphAblationModel.from_p1_checkpoint(
        str(checkpoint), use_gnn=True, num_tickers=num_tickers,
        graph_train_end_date=graph.train_end_date, graph_manifest_hash=graph.content_hash("train"))
    model.to(device)
    _log(stamp, f"seed={seed} training GAT head ({GRAPH_EPOCHS} epochs, knn-{TOP_K})")
    shared_base = _build_shared_graph_base(model, graph, device, GRAPH_TRAIN_BATCH, GRAPH_VAL_BATCH)
    outcome = _run_one_graph_model(
        model, graph, graph_store, "GAT_P1", GRAPH_EPOCHS, seed, out / "GAT_P1", device,
        validation_batch_size=GRAPH_VAL_BATCH, train_batch_size=GRAPH_TRAIN_BATCH,
        base_cache=shared_base)
    seed_result = {
        "seed": seed,
        "P0_HAR": {"validation_metrics": p0["validation_metrics"], "test_metrics": p0["test_metrics"]},
        "GAT_P1": {"validation_metrics": outcome["validation_metrics"],
                   "test_metrics": outcome.get("test_metrics")},
        "comparison": {
            "validation": _compare(outcome["validation_metrics"], p0["validation_metrics"]),
            "test": _compare(outcome["test_metrics"], p0["test_metrics"]),
        },
    }
    _write_json(out / "gat_price_vs_har.json", seed_result)
    _log(stamp, f"seed={seed} DONE")
    return seed_result


def main(ts: str, device_name: str, seeds: tuple[int, ...]) -> None:
    stamp = Path(R) / "temp" / f"gat_price_quick_{ts}_h{HORIZON}"
    device = resolve_graph_device(device_name)
    _log(stamp, f"device={device} horizon={HORIZON} adjacency={ADJACENCY} top_k={TOP_K} "
                f"seeds={seeds} building shared basis ...")
    pooled, graph, graph_store, allowed = build_basis(device, stamp)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        out = Path(R) / "results" / f"gat_price_quick_seed{seed}_{ts}" / f"h{HORIZON}"
        t0 = time.perf_counter()
        per_seed.append(run_seed(pooled, graph, graph_store, allowed, out, seed, device, stamp))
        _log(stamp, f"seed={seed} elapsed {time.perf_counter() - t0:.1f}s")
    combined = {
        "timestamp": ts, "horizon": HORIZON, "graph_epochs": GRAPH_EPOCHS,
        "backbone_epochs": POOLED_EPOCHS, "seeds": list(seeds),
        "adjacency": _adjacency_config(ADJACENCY, TOP_K, 0.7),
        "edge_density": _edge_density_stats(graph), "snapshot_count": len(graph.snapshots),
        "basis": "masked manifest, leakage-safe graph-bound train, shared per-ticker scalers, "
                 "positivity floor, identical val/test observations",
        "model": "GAT on price-only P1 backbone (news OFF, gate OFF) + knn-8 message passing",
        "per_seed": per_seed,
    }
    combined_path = Path(R) / "docs" / "reports" / f"gat_price_quick_{ts}.json"
    _write_json(combined_path, combined)
    stamp.with_suffix(".ALL_DONE").write_text("done", encoding="utf-8")
    _log(stamp, f"ALL_DONE combined={combined_path}")


if __name__ == "__main__":
    ts_arg = sys.argv[1]
    device_arg = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    seeds_arg = tuple(int(s) for s in sys.argv[3].split(",")) if len(sys.argv) > 3 else (42,)
    main(ts_arg, device_arg, seeds_arg)
