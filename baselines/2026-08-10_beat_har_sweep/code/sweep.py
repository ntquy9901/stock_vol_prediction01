"""Beat-HAR sweep orchestrator: C1/C2/C3/C5/C6 graph-head training on the consistent fair basis.

Reuses the pilot's build_basis, graph-safe P3 backbone, frozen-encoder base cache, and raw-scale
evaluator READ-ONLY. Each config retrains only the graph-stage message-passing + head (the P3 backbone
and the val/test observations are IDENTICAL to the consistent ladder) under its loss / adjacency / head
variation, so the comparison to HAR is on the same fair basis. See design/design.md.

Usage (GPU venv):
  python .../code/sweep.py <CONFIG> <TS> [device]
where CONFIG in {C1,C2,C3,C5,C6}. Writes results/beat_har_<CONFIG>_<TS>/seed{seed}/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_CODE = str(Path(__file__).resolve().parent)
_PILOT = str(Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code")
_ROOT = str(Path(__file__).resolve().parents[3])
for _p in (_CODE, _PILOT, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from torch import nn  # noqa: E402

from adjacency_ops import LearnedAdjacency, mask_learned_adjacency, mask_static_adjacency  # noqa: E402
from qlike_torch import snapshot_qlike_loss  # noqa: E402
from spillover import directed_spillover_adjacency, load_train_volatility_panel  # noqa: E402

from ladder_consistent import build_basis  # noqa: E402
from models import GraphAblationModel, _ResidualMessagePassing  # noqa: E402
from run_pilot import (  # noqa: E402
    _build_shared_graph_base, _mean_snapshot_mse, _plot_learning_curve, _seed_graph_device,
    _write_graph_predictions, _write_json, build_graph_bound_p3_warm_start,
    build_graph_safe_p3_checkpoint,
)
from train import evaluate_records  # noqa: E402

SEEDS = (42, 123, 2026)
GRAPH_EPOCHS = 20
BACKBONE_WARM_EPOCHS = 4
BACKBONE_SAFE_EPOCHS = 1
BACKBONE_DROPOUT = 0.2
BACKBONE_BATCH = 256
GRAPH_TRAIN_BATCH = 32
GRAPH_VAL_BATCH = 32
SPILLOVER_VAR_LAG = 1
SPILLOVER_FEVD_HORIZON = 10

CONFIGS: dict[str, dict[str, Any]] = {
    "C1": {"loss": "qlike", "adjacency": "knn", "head": "monolithic"},
    "C2": {"loss": "qlike", "adjacency": "knn", "head": "har_residual"},
    "C3": {"loss": "qlike", "adjacency": "spillover", "head": "monolithic", "omit_self": False},
    "C5": {"loss": "qlike", "adjacency": "spillover", "head": "monolithic", "omit_self": True,
           "k_sweep": (4, 8, 12, 16)},
    "C6": {"loss": "qlike", "adjacency": "learned", "head": "monolithic"},
}
_METRIC_KEYS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


# ---------------------------------------------------------------------------- backbone + base cache

def build_backbone(pooled, graph, graph_store, out: Path, seed: int, device) -> GraphAblationModel:
    """The SAME graph-safe MSE P3 backbone the consistent ladder wraps (warm-start + 1 safe epoch)."""

    graph_hash = graph.content_hash("train")
    warm = build_graph_bound_p3_warm_start(pooled, graph, out, seed, graph_store,
                                            epochs=BACKBONE_WARM_EPOCHS, device=device,
                                            train_batch_size=BACKBONE_BATCH, dropout=BACKBONE_DROPOUT)
    safe = build_graph_safe_p3_checkpoint(pooled, graph, out, seed, warm, graph_store,
                                          epochs=BACKBONE_SAFE_EPOCHS, device=device,
                                          train_batch_size=BACKBONE_BATCH, dropout=BACKBONE_DROPOUT)
    model = GraphAblationModel.from_p3_checkpoint(str(safe), use_gnn=True,
                                                  graph_train_end_date=graph.train_end_date,
                                                  graph_manifest_hash=graph_hash)
    model.to(device)
    return model


# ---------------------------------------------------------------------------- adjacency per split

def _spillover_static(graph, graph_store) -> np.ndarray:
    """Frozen directed spillover matrix from the TRAIN volatility panel (train-only, leakage-safe)."""

    tickers_ordered = [t for t, _ in sorted(graph.ticker_to_id.items(), key=lambda kv: kv[1])]
    panel = load_train_volatility_panel(Path(_ROOT) / "data" / "processed", tickers_ordered,
                                        graph.train_end_date)
    return directed_spillover_adjacency(panel, var_lag=SPILLOVER_VAR_LAG,
                                        fevd_horizon=SPILLOVER_FEVD_HORIZON)


def _static_adjacency_by_snapshot(snapshots, static, omit_self, top_k) -> list[np.ndarray]:
    return [mask_static_adjacency(static, snap.presence_mask, omit_self=omit_self, top_k=top_k)
            for snap in snapshots]


# ---------------------------------------------------------------------------- C2 residual head

class HARResidualHead(nn.Module):
    """C2: final = HAR(frozen) + residual(graph). Zero-init residual ⇒ initial prediction == HAR."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message_passing = _ResidualMessagePassing(hidden_dim)
        self.residual = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, base, adjacency, presence, har_norm):
        residual = self.residual(base + self.message_passing(base, adjacency, presence)).squeeze(-1)
        return har_norm + residual


def _fit_pooled_har(graph, graph_store) -> LinearRegression:
    """Pooled HAR OLS on graph-bound train nodes (last-timestep 3 HAR features → normalized target)."""

    features, targets = [], []
    for snap in graph.snapshots:
        if snap.split != "train":
            continue
        present = np.flatnonzero(snap.presence_mask)
        for node_idx in present:
            features.append(snap.x_price[node_idx, -1, :])
            targets.append(snap.nodes[node_idx].y_norm)
    return LinearRegression().fit(np.asarray(features, dtype=float), np.asarray(targets, dtype=float))


def _har_norm_by_snapshot(snapshots, har_model) -> list[np.ndarray]:
    rows = []
    for snap in snapshots:
        node_count = len(snap.nodes)
        har = np.zeros(node_count, dtype=np.float32)
        present = np.flatnonzero(snap.presence_mask)
        if present.size:
            feats = np.stack([snap.x_price[i, -1, :] for i in present]).astype(float)
            har[present] = har_model.predict(feats).astype(np.float32)
        rows.append(har)
    return rows


# ---------------------------------------------------------------------------- training / eval

def _stack_targets(snapshots, device):
    presence = torch.from_numpy(np.stack([s.presence_mask for s in snapshots])).to(device).bool()
    ticker_ids = torch.tensor([[n.ticker_id for n in s.nodes] for s in snapshots],
                              dtype=torch.long, device=device)
    targets = torch.tensor([[n.y_norm for n in s.nodes] for s in snapshots],
                           dtype=torch.float32, device=device)
    return presence, ticker_ids, targets


def _batch_adjacency(cfg, snapshots, static_adj_list, learned_module, presence, device, start):
    """Return the [B,N,N] adjacency tensor for one batch under the config's adjacency mode."""

    if cfg["adjacency"] == "knn":
        return torch.from_numpy(np.stack([s.adjacency for s in snapshots])).to(device)
    if cfg["adjacency"] == "spillover":
        return torch.from_numpy(np.stack(static_adj_list[start:start + len(snapshots)])).to(device)
    if cfg["adjacency"] == "learned":
        return mask_learned_adjacency(learned_module(), presence)
    raise ValueError(f"unknown adjacency mode {cfg['adjacency']!r}")


def _predict_batch(cfg, model, residual_head, base_slice, adjacency, ticker_ids, presence, har_slice):
    base = torch.stack(base_slice)
    if cfg["head"] == "har_residual":
        floored_off = model._positivity_configured
        model._positivity_configured = False
        try:
            resid_pred = residual_head(base, adjacency, presence, har_slice)
        finally:
            model._positivity_configured = floored_off
        # apply the same positivity floor the monolithic head uses, on the combined prediction
        return model._apply_positivity(resid_pred, ticker_ids)
    return model.apply_graph_head(base, adjacency, ticker_ids, presence)


def _loss(cfg, model, predictions, targets, ticker_ids, presence):
    if cfg["loss"] == "mse":
        return _mean_snapshot_mse(predictions, targets, presence)
    mean = model.target_mean[ticker_ids]
    std = model.target_std[ticker_ids]
    return snapshot_qlike_loss(predictions, targets, mean, std, presence)


def _run_split(cfg, model, residual_head, snapshots, base_list, static_adj_list, learned_module,
               har_list, device, batch_size, train: bool, optimizer=None):
    total = torch.zeros((), device=device)
    count = 0
    records: list[dict[str, Any]] = []
    for start in range(0, len(snapshots), batch_size):
        batch = snapshots[start:start + batch_size]
        presence, ticker_ids, targets = _stack_targets(batch, device)
        har_slice = None
        if har_list is not None:
            har_slice = torch.from_numpy(np.stack(har_list[start:start + len(batch)])).to(device)
        adjacency = _batch_adjacency(cfg, batch, static_adj_list, learned_module, presence, device, start)
        if train:
            optimizer.zero_grad()
        predictions = _predict_batch(cfg, model, residual_head, base_list[start:start + len(batch)],
                                     adjacency, ticker_ids, presence, har_slice)
        loss = _loss(cfg, model, predictions, targets, ticker_ids, presence)
        if train:
            loss.backward()
            if any(p.grad is not None for p in model.price_encoder.parameters()):
                raise RuntimeError("frozen encoder received gradients")
            optimizer.step()
        total = total + loss.detach() * len(batch)
        count += len(batch)
        if not train:
            for snap, snap_pred, snap_present in zip(batch, predictions.detach().cpu(),
                                                     presence.cpu(), strict=True):
                for idx, (node, pred) in enumerate(zip(snap.nodes, snap_pred, strict=True)):
                    if bool(snap_present[idx]):
                        records.append({"ticker_id": node.ticker_id, "target_date": snap.target_date,
                                        "prediction_norm": float(pred), "target_raw": node.y_raw})
    return (total / count).item(), records


def train_and_eval(cfg_name: str, cfg: dict[str, Any], pooled, graph, graph_store, model,
                   base_cache, out: Path, seed: int, device, top_k: int | None = None) -> dict[str, Any]:
    _seed_graph_device(seed, device)
    model.configure_positivity(graph_store)
    train_snaps = [s for s in graph.snapshots if s.split == "train"]
    val_snaps = [s for s in graph.snapshots if s.split == "val"]
    test_snaps = [s for s in graph.snapshots if s.split == "test"]

    static = None
    static_by_split = {"train": None, "val": None, "test": None}
    if cfg["adjacency"] == "spillover":
        static = _spillover_static(graph, graph_store)
        for split, snaps in (("train", train_snaps), ("val", val_snaps), ("test", test_snaps)):
            static_by_split[split] = _static_adjacency_by_snapshot(
                snaps, static, cfg.get("omit_self", False), top_k)

    residual_head = None
    har_by_split = {"train": None, "val": None, "test": None}
    if cfg["head"] == "har_residual":
        har_model = _fit_pooled_har(graph, graph_store)
        for split, snaps in (("train", train_snaps), ("val", val_snaps), ("test", test_snaps)):
            har_by_split[split] = _har_norm_by_snapshot(snaps, har_model)
        residual_head = HARResidualHead(model.head[0].in_features).to(device)

    learned_module = None
    if cfg["adjacency"] == "learned":
        learned_module = LearnedAdjacency(len(graph.ticker_to_id)).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if residual_head is not None:
        trainable = [p for p in residual_head.parameters()]  # C2 trains only HAR-residual + its MP
    if learned_module is not None:
        trainable = trainable + list(learned_module.parameters())
    optimizer = torch.optim.Adam(trainable, weight_decay=1e-5)

    train_losses, val_losses = [], []
    for _ in range(GRAPH_EPOCHS):
        model.train()
        if residual_head is not None:
            residual_head.train()
        tl, _ = _run_split(cfg, model, residual_head, train_snaps, base_cache["train"],
                           static_by_split["train"], learned_module, har_by_split["train"],
                           device, GRAPH_TRAIN_BATCH, train=True, optimizer=optimizer)
        train_losses.append(tl)
        model.eval()
        if residual_head is not None:
            residual_head.eval()
        with torch.no_grad():
            vl, _ = _run_split(cfg, model, residual_head, val_snaps, base_cache["val"],
                               static_by_split["val"], learned_module, har_by_split["val"],
                               device, GRAPH_VAL_BATCH, train=False)
        val_losses.append(vl)
    _plot_learning_curve(train_losses, val_losses, out / "learning_curve.png")

    model.eval()
    if residual_head is not None:
        residual_head.eval()
    result: dict[str, Any] = {"config": cfg_name, "seed": seed, "graph_epochs": GRAPH_EPOCHS,
                              "top_k": top_k, "train_losses": train_losses, "val_losses": val_losses}
    with torch.no_grad():
        for split, snaps, cache_key in (("validation", val_snaps, "val"), ("test", test_snaps, "test")):
            _, records = _run_split(cfg, model, residual_head, snaps, base_cache[cache_key],
                                    static_by_split[cache_key], learned_module,
                                    har_by_split[cache_key], device, GRAPH_VAL_BATCH, train=False)
            evaluation = evaluate_records(records, graph_store)
            fname = "predictions.json" if split == "validation" else "predictions_test.json"
            _write_graph_predictions(records, evaluation, out / fname)
            result[f"{split}_metrics"] = {k: float(evaluation["metrics"][k]) for k in _METRIC_KEYS}
    if static is not None:
        np.save(out / "spillover_static.npy", static)
    _write_json(out / "results.json", result)
    return result


# ---------------------------------------------------------------------------- driver

def _log(stamp: Path, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    with stamp.with_suffix(".progress").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main(config_name: str, ts: str, device_name: str = "cuda") -> None:
    if config_name not in CONFIGS:
        raise ValueError(f"config must be one of {sorted(CONFIGS)}")
    from run_pilot import resolve_graph_device
    device = resolve_graph_device(device_name)
    cfg = CONFIGS[config_name]
    stamp = Path(_ROOT) / "temp" / f"beat_har_{config_name}_{ts}"
    _log(stamp, f"device={device} config={config_name} building basis ...")
    pooled, graph, graph_store, _allowed = build_basis(device, stamp)
    _log(stamp, f"basis ready snapshots={len(graph.snapshots)}")

    for seed in SEEDS:
        base_out = Path(_ROOT) / "results" / f"beat_har_{config_name}_{ts}" / f"seed{seed}"
        base_out.mkdir(parents=True, exist_ok=True)
        _log(stamp, f"seed={seed} building backbone + base cache ...")
        model = build_backbone(pooled, graph, graph_store, base_out / "backbone", seed, device)
        base_cache = _build_shared_graph_base(model, graph, device, GRAPH_TRAIN_BATCH, GRAPH_VAL_BATCH)
        k_sweep = cfg.get("k_sweep")
        if k_sweep:
            for top_k in k_sweep:
                out = base_out / f"k{top_k}"
                out.mkdir(parents=True, exist_ok=True)
                # reload a fresh backbone head/mp for each k so runs are independent
                fresh = build_backbone(pooled, graph, graph_store, base_out / "backbone", seed, device)
                res = train_and_eval(config_name, cfg, pooled, graph, graph_store, fresh, base_cache,
                                     out, seed, device, top_k=top_k)
                _log(stamp, f"seed={seed} k={top_k} test_qlike={res['test_metrics']['qlike']:.4f}")
        else:
            res = train_and_eval(config_name, cfg, pooled, graph, graph_store, model, base_cache,
                                 base_out, seed, device)
            _log(stamp, f"seed={seed} test_qlike={res['test_metrics']['qlike']:.4f} "
                        f"test_rmse={res['test_metrics']['rmse']:.7f}")
    stamp.with_suffix(".DONE").write_text("done", encoding="utf-8")
    _log(stamp, f"{config_name} ALL_DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cuda")
