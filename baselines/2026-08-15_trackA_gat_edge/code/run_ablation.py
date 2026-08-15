"""Full component ablation of the Track-A GAT model across horizons h in {1,5,10,22}.

Rungs (each deeper one adds one component; separately TRAINED where the component changes weights):
  HAR   = pooled HAR linear regression (external baseline).
  LSTM  = price LSTM on 5 node features only (no news, no gate, no graph)   [trained]
  NEWS  = + news branch (no gate, no graph)                                 [trained]
  NODE  = + per-ticker gate (no graph)  == full model read graph-off        [trained]
  GNN   = + directed vol->PK GAT graph  == same full model read graph-on    [NODE's checkpoint]

Reuses the combo basis (5 features + news + vol->PK edge) per horizon. 1 seed first (extend later).
Run: python <.../code/run_ablation.py> <TS> [device] [seed] [epochs] [horizons...]
Writes results/trackA_ablation_h{h}_seed{seed}_<TS>/ladder_metrics.json.
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

import torch  # noqa: E402

import combo_ladder  # noqa: E402
from model import TrackAGatModel  # noqa: E402
from run_pilot import _write_json, resolve_graph_device  # noqa: E402
from run_trackA import ROOT, _StoreShim, _evaluate_rung, build_trackA_basis  # noqa: E402
from train_resume import load_checkpoint, train_with_resume  # noqa: E402


def _train(basis, train_snaps, val_snaps, ckpt: Path, epochs, seed, device,
           use_news: bool, use_gate: bool):
    model = TrackAGatModel(basis["price_dim"], basis["news_dim"], basis["num_tickers"],
                           use_news=use_news, use_gate=use_gate)
    model.configure_positivity(basis["scaler_mean"], basis["scaler_std"])
    train_with_resume(model, train_snaps, val_snaps, ckpt, epochs, device, seed)
    model.load_state_dict(load_checkpoint(ckpt)["best_state"])
    return model.to(device)


def run_horizon(horizon: int, seed: int, epochs: int, ts: str, device: torch.device) -> dict[str, Any]:
    combo_ladder.HORIZON = horizon                              # combo build_basis reads this global
    out_base = ROOT / "results" / f"trackA_ablation_h{horizon}_seed{seed}_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)
    basis = build_trackA_basis(out_base)
    store = _StoreShim(basis["scaler_mean"], basis["scaler_std"])
    tr = [s for s in basis["snaps"] if s["split"] == "train"]
    va = [s for s in basis["snaps"] if s["split"] == "val"]
    te = [s for s in basis["snaps"] if s["split"] == "test"]

    def _metrics(model, name, apply_graph):
        v = _evaluate_rung(model, va, store, device, apply_graph, out_base / name, dump=False)
        t = _evaluate_rung(model, te, store, device, apply_graph, out_base / name, dump=True)
        return {"validation_metrics": v["metrics"], "test_metrics": t["metrics"],
                "floor_hit_fraction": t["floor_hit_fraction"]}

    lstm = _train(basis, tr, va, out_base / "lstm.pt", epochs, seed, device, use_news=False, use_gate=False)
    news = _train(basis, tr, va, out_base / "news.pt", epochs, seed, device, use_news=True, use_gate=False)
    full = _train(basis, tr, va, out_base / "full.pt", epochs, seed, device, use_news=True, use_gate=True)

    har = basis["har"]
    rungs = {
        "HAR": {"validation_metrics": har["val"], "test_metrics": har["test"],
                "floor_hit_fraction": har.get("floor_hit_fraction", 0.0)},
        "LSTM": _metrics(lstm, "LSTM", apply_graph=False),
        "NEWS": _metrics(news, "NEWS", apply_graph=False),
        "NODE": _metrics(full, "NODE", apply_graph=False),
        "GNN": _metrics(full, "GNN", apply_graph=True),
    }
    ladder = {"horizon": horizon, "seed": seed, "epochs": epochs, "rungs": rungs,
              "graph_effect": rungs["GNN"]["test_metrics"]["qlike"] - rungs["NODE"]["test_metrics"]["qlike"]}
    _write_json(out_base / "ladder_metrics.json", ladder)
    return ladder


def main(ts: str, device_name: str = "cuda", seed: int = 42, epochs: int = 15,
         horizons: tuple[int, ...] = (1, 5, 10, 22)) -> None:
    device = resolve_graph_device(device_name)
    previous = combo_ladder.HORIZON
    try:
        for h in horizons:
            t0 = time.perf_counter()
            ladder = run_horizon(h, seed, epochs, ts, device)
            ql = [round(ladder["rungs"][r]["test_metrics"]["qlike"], 4)
                  for r in ("HAR", "LSTM", "NEWS", "NODE", "GNN")]
            print(f"h={h} seed={seed} elapsed={time.perf_counter() - t0:.1f}s "
                  f"graph_effect={ladder['graph_effect']:.6f} HAR/LSTM/NEWS/NODE/GNN qlike={ql}",
                  flush=True)
    finally:
        combo_ladder.HORIZON = previous


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    _seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    _epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    _horizons = tuple(int(x) for x in sys.argv[5:]) if len(sys.argv) > 5 else (1, 5, 10, 22)
    main(_ts, _device, _seed, _epochs, _horizons)
