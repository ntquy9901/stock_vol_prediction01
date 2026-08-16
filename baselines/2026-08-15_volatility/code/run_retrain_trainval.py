"""Retrain-on-(train+val) variant of the leave-one-out ablation, scored on the held-out test.

Same rungs as ``run_ablation.py`` plus the ``lstm_only`` anchor, but each deep model is RETRAINED
on train+val (not train-only) with a FIXED epoch budget and NO early stopping (patience=0), then read
out once on the held-out test split. HAR (P0) is likewise refit on train+val. Test is never used for
selection, so there is no leakage; the only change vs the leave-one-out run is that the extra val
observations join the training set and there is no held-out early-stopping set.

Rungs (all retrained on train+val, evaluated on test):
  HAR         = pooled HAR linear regression refit on train+val (external baseline).
  FULL        = LSTM(5 node feats) + vol->PK GAT graph + news + per-ticker gate     [graph on]
  minus_graph = FULL with the whole GAT branch removed (use_graph=False)            [graph off]
  minus_gate  = FULL without the per-ticker gate (news, graph on)                   [graph on]
  minus_news  = FULL without the news branch (graph on)                             [graph on]
  lstm_only   = price-only backbone: no news, no gate, no graph                     [graph off]

Run (GPU venv):  python <.../code/run_retrain_trainval.py> <TS> [device] [seed] [epochs] [horizons...]
Writes results/volatility_retrain_h{h}_seed{seed}_<TS>/retrain_metrics.json + per-rung test dumps.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code",
           _ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code",
           _ROOT / "baselines" / "2026-08-14_pooled_news_edanode_gnn" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import combo_ladder  # noqa: E402
import features  # noqa: E402
from eda_ladder import run_e0  # noqa: E402
from model import VolatilityModel  # noqa: E402
from run_pilot import _write_json, resolve_graph_device  # noqa: E402
from run_volatility import ROOT, _StoreShim, _evaluate_rung  # noqa: E402
from train_resume import load_checkpoint, train_with_resume  # noqa: E402

# Deep rung spec: (name, use_news, use_gate, use_graph, apply_graph). apply_graph is inert when
# use_graph=False; it is set False for the two no-graph rungs to keep readouts explicit.
_RUNGS = (
    ("FULL", True, True, True, True),
    ("minus_graph", True, True, False, False),
    ("minus_gate", True, False, True, True),
    ("minus_news", False, False, True, True),
    ("lstm_only", False, False, False, False),
)


def _build_deep_basis(graph, store) -> dict[str, Any]:
    """Snapshot dicts + per-ticker scaler tensors + dims, mirroring run_volatility.build_volatility_basis."""

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
    return {"snaps": snaps, "num_tickers": num_tickers, "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "price_dim": int(graph.snapshots[0].x_price.shape[-1]),
            "news_dim": int(graph.snapshots[0].x_news.shape[-1])}


def _train(basis, train_snaps, ckpt: Path, epochs, seed, device,
           use_news: bool, use_gate: bool, use_graph: bool):
    """Retrain one variant on train+val with a fixed epoch budget and no early stopping.

    train+val is passed as both the training and the validation set: with patience=0 the run never
    early-stops (it uses the whole `epochs` budget) and best_state tracks the train+val MSE, which
    decreases monotonically, so the loaded checkpoint is effectively the fully-trained final model.
    """

    model = VolatilityModel(basis["price_dim"], basis["news_dim"], basis["num_tickers"],
                           use_news=use_news, use_gate=use_gate, use_graph=use_graph)
    model.configure_positivity(basis["scaler_mean"], basis["scaler_std"])
    train_with_resume(model, train_snaps, train_snaps, ckpt, epochs, device, seed,
                      apply_graph=True, patience=0)  # patience=0 -> no early stop, full budget
    model.load_state_dict(load_checkpoint(ckpt)["best_state"])
    return model.to(device)


def run_horizon(horizon: int, seed: int, epochs: int, ts: str, device: torch.device) -> dict[str, Any]:
    combo_ladder.HORIZON = horizon                              # combo build_basis reads this global
    features._VOLUME_WINDOW = 22                                # unified 22-day window (as build_volatility_basis)
    out_base = ROOT / "results" / f"volatility_retrain_h{horizon}_seed{seed}_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)
    pooled, graph, store, _allowed, _edges = combo_ladder.build_basis(out_base)
    basis = _build_deep_basis(graph, store)
    shim = _StoreShim(basis["scaler_mean"], basis["scaler_std"])
    train_snaps = [s for s in basis["snaps"] if s["split"] in ("train", "val")]  # retrain set
    test_snaps = [s for s in basis["snaps"] if s["split"] == "test"]            # held-out eval

    rungs: dict[str, dict[str, Any]] = {}
    for name, use_news, use_gate, use_graph, apply_graph in _RUNGS:
        model = _train(basis, train_snaps, out_base / f"{name}.pt", epochs, seed, device,
                       use_news=use_news, use_gate=use_gate, use_graph=use_graph)
        test_eval = _evaluate_rung(model, test_snaps, shim, device, apply_graph,
                                   out_base / name, dump=True)
        rungs[name] = {"test_metrics": test_eval["metrics"],
                       "floor_hit_fraction": test_eval["floor_hit_fraction"],
                       "trained_on": "train+val", "epochs": epochs}

    har = run_e0(pooled, list(pooled.samples["train"]) + list(pooled.samples["val"]),
                 store, out_base / "P0")
    rungs["HAR"] = {"test_metrics": har["test_metrics"], "trained_on": "train+val", "epochs": epochs}

    result = {"horizon": horizon, "seed": seed, "epochs": epochs, "trained_on": "train+val",
              "rungs": rungs}
    _write_json(out_base / "retrain_metrics.json", result)
    return result


def main(ts: str, device_name: str = "cuda", seed: int = 42, epochs: int = 9,
         horizons: tuple[int, ...] = (1, 5, 10, 22)) -> None:
    device = resolve_graph_device(device_name)
    previous = combo_ladder.HORIZON
    order = ("HAR", "FULL", "minus_graph", "minus_gate", "minus_news", "lstm_only")
    try:
        for h in horizons:
            t0 = time.perf_counter()
            result = run_horizon(h, seed, epochs, ts, device)
            ql = [round(result["rungs"][r]["test_metrics"]["qlike"], 4) for r in order]
            print(f"h={h} seed={seed} elapsed={time.perf_counter() - t0:.1f}s "
                  f"{order} test_qlike={ql}", flush=True)
    finally:
        combo_ladder.HORIZON = previous


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    _seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    _epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 9
    _horizons = tuple(int(x) for x in sys.argv[5:]) if len(sys.argv) > 5 else (1, 5, 10, 22)
    main(_ts, _device, _seed, _epochs, _horizons)
