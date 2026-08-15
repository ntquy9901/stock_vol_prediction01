"""LSTM-only reference rung (price-only: no news, no gate, no graph) for the leave-one-out study.

This is NOT a leave-one-out variant (it removes three components at once); it is the price-only
backbone anchor (analogous to a P1 rung) that measures what the whole deep stack adds over a plain
price LSTM. It reuses the SAME trackA basis per horizon and writes its metrics + test dump into the
SAME results/trackA_ablation_h{h}_seed{seed}_<TS>/ directory as the leave-one-out run, under an
`lstm_only/` subdir, so the two merge into one reporting table.

Run with the SAME <TS> as the matching leave-one-out run:
  python <.../code/run_lstm_only.py> <TS> [device] [seed] [epochs] [horizons...]
Writes results/trackA_ablation_h{h}_seed{seed}_<TS>/lstm_only_metrics.json.
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


def run_horizon(horizon: int, seed: int, epochs: int, ts: str, device: torch.device) -> dict[str, Any]:
    combo_ladder.HORIZON = horizon                              # combo build_basis reads this global
    out_base = ROOT / "results" / f"trackA_ablation_h{horizon}_seed{seed}_{ts}"
    out_base.mkdir(parents=True, exist_ok=True)
    basis = build_trackA_basis(out_base)
    store = _StoreShim(basis["scaler_mean"], basis["scaler_std"])
    tr = [s for s in basis["snaps"] if s["split"] == "train"]
    va = [s for s in basis["snaps"] if s["split"] == "val"]
    te = [s for s in basis["snaps"] if s["split"] == "test"]

    model = TrackAGatModel(basis["price_dim"], basis["news_dim"], basis["num_tickers"],
                           use_news=False, use_gate=False, use_graph=False)  # price-only backbone
    model.configure_positivity(basis["scaler_mean"], basis["scaler_std"])
    ckpt = out_base / "lstm_only.pt"
    train_with_resume(model, tr, va, ckpt, epochs, device, seed, apply_graph=True,
                      patience=3, min_epochs=6)  # apply_graph inert when use_graph=False
    model.load_state_dict(load_checkpoint(ckpt)["best_state"])
    model.to(device)

    v = _evaluate_rung(model, va, store, device, True, out_base / "lstm_only", dump=False)
    t = _evaluate_rung(model, te, store, device, True, out_base / "lstm_only", dump=True)
    metrics = {"validation_metrics": v["metrics"], "test_metrics": t["metrics"],
               "floor_hit_fraction": t["floor_hit_fraction"]}
    _write_json(out_base / "lstm_only_metrics.json",
                {"horizon": horizon, "seed": seed, "epochs": epochs, "rung": "LSTM_only", "metrics": metrics})
    return metrics


def main(ts: str, device_name: str = "cuda", seed: int = 42, epochs: int = 12,
         horizons: tuple[int, ...] = (1, 5, 10, 22)) -> None:
    device = resolve_graph_device(device_name)
    previous = combo_ladder.HORIZON
    try:
        for h in horizons:
            t0 = time.perf_counter()
            m = run_horizon(h, seed, epochs, ts, device)
            print(f"h={h} seed={seed} elapsed={time.perf_counter() - t0:.1f}s "
                  f"LSTM_only test qlike={m['test_metrics']['qlike']:.4f}", flush=True)
    finally:
        combo_ladder.HORIZON = previous


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    _seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    _epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    _horizons = tuple(int(x) for x in sys.argv[5:]) if len(sys.argv) > 5 else (1, 5, 10, 22)
    main(_ts, _device, _seed, _epochs, _horizons)
