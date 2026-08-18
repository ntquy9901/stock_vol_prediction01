"""LEAVE-ONE-OUT ablation of the Track-A GAT model across horizons h in {1,5,10,22}.

Build the FULL model, then RETRAIN each variant with exactly one component removed (per CLAUDE.md
ablation rule) so all effects are on the same footing (no rung is a train/eval mismatch):
  HAR         = pooled HAR linear regression (external baseline).
  FULL        = LSTM(5 node feats) + vol->PK GAT graph + news + per-ticker gate     [trained]
  minus_graph = FULL retrained with the WHOLE GAT branch removed (no node/edge)     [trained]
  minus_gate  = FULL retrained without the per-ticker gate (news, graph-on)         [trained]
  minus_news  = FULL retrained without the news branch (graph-on)                   [trained]

Component contribution = QLIKE(FULL) - QLIKE(minus_X): negative means removing X hurt (X helped).
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
           use_news: bool, use_gate: bool, use_graph: bool):
    model = TrackAGatModel(basis["price_dim"], basis["news_dim"], basis["num_tickers"],
                           use_news=use_news, use_gate=use_gate, use_graph=use_graph)
    model.configure_positivity(basis["scaler_mean"], basis["scaler_std"])
    # no_graph drops the whole GAT branch (use_graph=False) so adjacency is never consulted; the
    # other variants keep the graph on. apply_graph is True here and inert when use_graph is False.
    # Pooled models converge ~epoch 5-6 then overfit -> early-stop (patience 3, min 6) under the cap.
    train_with_resume(model, train_snaps, val_snaps, ckpt, epochs, device, seed, apply_graph=True,
                      patience=3, min_epochs=6)
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

    # FULL has all components; each variant RETRAINS from scratch with exactly one removed, so the
    # three effects are on the same footing (all retrain-based, no train/eval mismatch on any rung).
    full = _train(basis, tr, va, out_base / "full.pt", epochs, seed, device,
                  use_news=True, use_gate=True, use_graph=True)
    no_graph = _train(basis, tr, va, out_base / "no_graph.pt", epochs, seed, device,
                      use_news=True, use_gate=True, use_graph=False)  # drop whole GAT branch
    no_gate = _train(basis, tr, va, out_base / "no_gate.pt", epochs, seed, device,
                     use_news=True, use_gate=False, use_graph=True)   # remove per-ticker gate
    no_news = _train(basis, tr, va, out_base / "no_news.pt", epochs, seed, device,
                     use_news=False, use_gate=False, use_graph=True)  # remove news (gate is a
    #        no-op without news: model.py applies the gate only inside the `if use_news` branch)

    har = basis["har"]
    rungs = {
        "HAR": {"validation_metrics": har["val"], "test_metrics": har["test"],
                "floor_hit_fraction": har.get("floor_hit_fraction", 0.0)},
        "FULL": _metrics(full, "FULL", apply_graph=True),
        "minus_graph": _metrics(no_graph, "minus_graph", apply_graph=True),  # GAT branch absent
        "minus_gate": _metrics(no_gate, "minus_gate", apply_graph=True),
        "minus_news": _metrics(no_news, "minus_news", apply_graph=True),
    }
    full_q = rungs["FULL"]["test_metrics"]["qlike"]
    effects = {  # QLIKE(FULL) - QLIKE(FULL minus X): negative => removing X hurt (X helped)
        "graph": full_q - rungs["minus_graph"]["test_metrics"]["qlike"],
        "gate": full_q - rungs["minus_gate"]["test_metrics"]["qlike"],
        "news": full_q - rungs["minus_news"]["test_metrics"]["qlike"],
    }
    ladder = {"horizon": horizon, "seed": seed, "epochs": epochs, "rungs": rungs,
              "leave_one_out_effects": effects}
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
            rung_order = ("HAR", "FULL", "minus_graph", "minus_gate", "minus_news")
            ql = [round(ladder["rungs"][r]["test_metrics"]["qlike"], 4) for r in rung_order]
            eff = {k: round(v, 6) for k, v in ladder["leave_one_out_effects"].items()}
            print(f"h={h} seed={seed} elapsed={time.perf_counter() - t0:.1f}s "
                  f"effects(FULL-minusX){eff} {rung_order} qlike={ql}", flush=True)
    finally:
        combo_ladder.HORIZON = previous


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    _seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    _epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    _horizons = tuple(int(x) for x in sys.argv[5:]) if len(sys.argv) > 5 else (1, 5, 10, 22)
    main(_ts, _device, _seed, _epochs, _horizons)
