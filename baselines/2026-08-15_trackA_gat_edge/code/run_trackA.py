# code/run_trackA.py
"""Task 4: reuse the combo basis (5 node features + news + directed vol->PK Top-5 graph),
train the Track-A-style GAT model, and evaluate HAR vs NODE (graph off) vs GNN (graph on).

Run (GPU venv):  python <.../code/run_trackA.py> <TS> [device] [epochs] [seeds...]
Writes results/trackA_gat_seed{seed}_<TS>/ladder_metrics.json + models/trackA_gat_seed{seed}_<TS>.pt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
PILOT = ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
EDA = ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code"
COMBO = ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code"
for _p in (str(CODE), str(PILOT), str(EDA), str(COMBO), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from combo_ladder import build_basis  # noqa: E402
from eda_ladder import run_e0  # noqa: E402
from model import POSITIVITY_EPSILON, TrackAGatModel  # noqa: E402
from run_pilot import _write_graph_predictions, _write_json, resolve_graph_device  # noqa: E402
from train import evaluate_records  # noqa: E402
from train_resume import load_checkpoint, train_with_resume  # noqa: E402


def build_trackA_basis(stamp: Path) -> dict[str, Any]:
    """Reuse the combo (5-feature + news + directed vol->PK) basis; add HAR (P0) reference.

    ``stamp`` doubles as the seed's own results directory: the combo basis builder logs progress
    to ``stamp.progress`` and HAR (P0) dumps land under ``stamp/P0/``.
    """

    stamp = Path(stamp)
    pooled, graph, store, allowed, _edge_count = build_basis(stamp)
    num_tickers = max(graph.ticker_to_id.values()) + 1
    scaler_mean = torch.zeros(num_tickers, dtype=torch.float32)
    scaler_std = torch.ones(num_tickers, dtype=torch.float32)
    for ticker_id in range(num_tickers):
        scaler = store.get(ticker_id).target_scaler
        scaler_mean[ticker_id] = float(scaler.mean[0])
        scaler_std[ticker_id] = float(scaler.std[0])

    snaps: list[dict[str, Any]] = []
    for snapshot in graph.snapshots:
        presence = (snapshot.presence_mask if snapshot.presence_mask is not None
                    else np.ones(len(snapshot.nodes), dtype=np.int8))
        snaps.append({
            "price": torch.from_numpy(np.asarray(snapshot.x_price, dtype=np.float32)),
            "news": torch.from_numpy(np.asarray(snapshot.x_news, dtype=np.float32)),
            "news_mask": torch.from_numpy(np.asarray(snapshot.news_mask, dtype=np.float32)),
            "ticker_ids": torch.arange(len(snapshot.nodes), dtype=torch.long),
            "adjacency": torch.from_numpy(np.asarray(snapshot.adjacency, dtype=np.float32)),
            "target": torch.tensor([node.y_norm for node in snapshot.nodes], dtype=torch.float32),
            "target_raw": [float(node.y_raw) for node in snapshot.nodes],
            "presence_mask": torch.from_numpy(np.asarray(presence, dtype=np.float32)),
            "split": snapshot.split,
            "target_date": snapshot.target_date,
        })
    price_dim = int(graph.snapshots[0].x_price.shape[-1])
    news_dim = int(graph.snapshots[0].x_news.shape[-1])
    har_out = stamp / "P0"
    har_raw = run_e0(pooled, allowed, store, har_out)
    har_floor = _floor_hit_from_dump(har_out / "predictions_test.json")
    har = {"val": har_raw["validation_metrics"], "test": har_raw["test_metrics"],
           "floor_hit_fraction": har_floor}
    return {"snaps": snaps, "num_tickers": num_tickers, "scaler_mean": scaler_mean,
            "scaler_std": scaler_std, "price_dim": price_dim, "news_dim": news_dim, "har": har}


def _floor_hit_from_dump(path: Path, eps: float = POSITIVITY_EPSILON) -> float:
    """Floor-hit fraction computed from a ``_write_graph_predictions`` dump (used for HAR)."""

    if not path.exists():
        return 0.0
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row["prediction_raw"] <= 2.0 * eps)
    return hits / len(rows)


class _StoreShim:
    """Minimal PreprocessorStore-compatible view built from the model's scaler tensors.

    Functionally equivalent to the real PreprocessorStore for target inverse-transform: the
    scaler tensors are copied verbatim from ``store.get(ticker_id).target_scaler`` in
    ``build_trackA_basis``, so this also lets the smoke test's fake basis (no real store) run
    ``evaluate_records`` unmodified.
    """

    def __init__(self, scaler_mean: torch.Tensor, scaler_std: torch.Tensor) -> None:
        self._mean = scaler_mean.numpy()
        self._std = scaler_std.numpy()

    def get(self, ticker_id: int) -> "_StoreShim._Preprocessor":
        return _StoreShim._Preprocessor(float(self._mean[ticker_id]), float(self._std[ticker_id]))

    def inverse_targets(self, ticker_ids, y_norm) -> np.ndarray:
        ids = np.asarray(ticker_ids)
        normalized = np.asarray(y_norm, dtype=float)
        return normalized * self._std[ids] + self._mean[ids]

    class _Preprocessor:
        def __init__(self, mean: float, std: float) -> None:
            self.target_scaler = _StoreShim._Scaler(mean, std)

    class _Scaler:
        def __init__(self, mean: float, std: float) -> None:
            self.mean = np.asarray([mean])
            self.std = np.asarray([std])


def _snapshot_records(model, snaps: list[dict[str, Any]], device: torch.device,
                      apply_graph: bool) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for snap in snaps:
            pred = model(snap["price"].unsqueeze(0).to(device), snap["news"].unsqueeze(0).to(device),
                         snap["news_mask"].unsqueeze(0).to(device),
                         snap["ticker_ids"].unsqueeze(0).to(device),
                         snap["adjacency"].unsqueeze(0).to(device), apply_graph=apply_graph)
            pred = pred.squeeze(0).cpu().numpy()
            presence = snap["presence_mask"].numpy()
            ticker_ids = snap["ticker_ids"].numpy()
            target_raw = snap["target_raw"]
            target_date = snap.get("target_date", "")
            for index in range(len(ticker_ids)):
                if not presence[index]:
                    continue
                records.append({"ticker_id": int(ticker_ids[index]), "target_date": target_date,
                                "prediction_norm": float(pred[index]),
                                "target_raw": float(target_raw[index])})
    return records


def _floor_hit_fraction(records: list[dict[str, Any]], store, eps: float = POSITIVITY_EPSILON) -> float:
    """Share of predictions pinned at the positivity floor (denormalized <= 2*eps)."""

    if not records:
        return 0.0
    hits = 0
    for record in records:
        scaler = store.get(int(record["ticker_id"])).target_scaler
        raw = record["prediction_norm"] * float(scaler.std[0]) + float(scaler.mean[0])
        if raw <= 2.0 * eps:
            hits += 1
    return hits / len(records)


def _evaluate_rung(model, snaps: list[dict[str, Any]], store, device: torch.device,
                   apply_graph: bool, out: Path, dump: bool) -> dict[str, Any]:
    records = _snapshot_records(model, snaps, device, apply_graph)
    evaluation = evaluate_records(records, store)
    if dump:
        out.mkdir(parents=True, exist_ok=True)
        _write_graph_predictions(records, evaluation, out / "predictions_test.json")
    return {"metrics": evaluation["metrics"], "floor_hit_fraction": _floor_hit_fraction(records, store)}


def run_seed(seed: int, epochs: int, ts: str, out_base: Path, device: torch.device,
            resume: bool = False) -> dict[str, Any]:
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    basis = build_trackA_basis(out_base)
    snaps = basis["snaps"]
    train_snaps = [s for s in snaps if s["split"] == "train"]
    val_snaps = [s for s in snaps if s["split"] == "val"]
    test_snaps = [s for s in snaps if s["split"] == "test"]

    model = TrackAGatModel(basis["price_dim"], basis["news_dim"], basis["num_tickers"])
    model.configure_positivity(basis["scaler_mean"], basis["scaler_std"])
    ckpt_path = out_base / "ckpt.pt"
    train_with_resume(model, train_snaps, val_snaps, ckpt_path, epochs, device, seed, resume=resume)
    model.load_state_dict(load_checkpoint(ckpt_path)["best_state"])
    model.to(device)

    store = _StoreShim(basis["scaler_mean"], basis["scaler_std"])

    def _rung(apply_graph: bool, name: str) -> dict[str, Any]:
        val_eval = _evaluate_rung(model, val_snaps, store, device, apply_graph, out_base / name, dump=False)
        test_eval = _evaluate_rung(model, test_snaps, store, device, apply_graph, out_base / name, dump=True)
        return {"validation_metrics": val_eval["metrics"], "test_metrics": test_eval["metrics"],
                "floor_hit_fraction": test_eval["floor_hit_fraction"]}

    node = _rung(False, "NODE")
    gnn = _rung(True, "GNN")
    har = basis["har"]
    rungs = {
        "HAR": {"validation_metrics": har["val"], "test_metrics": har["test"],
                "floor_hit_fraction": har.get("floor_hit_fraction", 0.0)},
        "NODE": node,
        "GNN": gnn,
    }
    ladder = {"seed": seed, "epochs": epochs, "rungs": rungs,
              "graph_effect": gnn["test_metrics"]["qlike"] - node["test_metrics"]["qlike"]}
    _write_json(out_base / "ladder_metrics.json", ladder)
    return ladder


def main(ts: str, device_name: str = "cuda", seeds: tuple[int, ...] = (42,), epochs: int = 15,
         resume: bool = False) -> None:
    device = resolve_graph_device(device_name)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        out_base = ROOT / "results" / f"trackA_gat_seed{seed}_{ts}"
        t0 = time.perf_counter()
        ladder = run_seed(seed, epochs, ts, out_base, device, resume=resume)
        ckpt = out_base / "ckpt.pt"
        target = models_dir / f"trackA_gat_seed{seed}_{ts}.pt"
        if ckpt.exists():
            target.write_bytes(ckpt.read_bytes())
        print(f"seed={seed} elapsed={time.perf_counter() - t0:.1f}s "
              f"graph_effect={ladder['graph_effect']:.6f}")


if __name__ == "__main__":
    _ts = sys.argv[1]
    _device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    _epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    _seeds = tuple(int(s) for s in sys.argv[4:]) if len(sys.argv) > 4 else (42,)
    main(_ts, _device, _seeds, _epochs)
