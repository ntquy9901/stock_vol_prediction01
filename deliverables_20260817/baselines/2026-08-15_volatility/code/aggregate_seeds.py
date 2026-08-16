"""Aggregate the volatility ablation across seeds: per-rung mean +/- std of each metric, per horizon.

Reads results/volatility_ablation_h{h}_seed{s}_<TS>/ladder_metrics.json (rungs HAR/FULL/minus_*) and
lstm_only_metrics.json (rung LSTM-only), for every seed, and returns/prints the seed mean and std of
the six metrics on test and validation, plus the mean component effect(X)=QLIKE(FULL)-QLIKE(FULL-X).

Run: python <.../code/aggregate_seeds.py> <TS> <seeds_csv> [horizons_csv]
  e.g. aggregate_seeds.py 2026-08-15_085544_loo 42,123,2026
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
ROOT = CODE.resolve().parents[2]

import numpy as np  # noqa: E402

RUNGS = ("HAR", "FULL", "minus_graph", "minus_gate", "minus_news", "LSTM-only")
METRICS = ("mse", "rmse", "mae", "r2", "qlike")


def _run_dir(ts: str, horizon: int, seed: int) -> Path:
    return ROOT / "results" / f"volatility_ablation_h{horizon}_seed{seed}_{ts}"


def _rung_metrics(ts: str, horizon: int, seed: int) -> dict[str, dict[str, dict[str, float]]]:
    """Return {rung: {split: {metric: value}}} for one seed/horizon."""
    d = json.loads((_run_dir(ts, horizon, seed) / "ladder_metrics.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {r: d["rungs"][r] for r in RUNGS if r != "LSTM-only"}
    lo = json.loads((_run_dir(ts, horizon, seed) / "lstm_only_metrics.json").read_text(encoding="utf-8"))
    out["LSTM-only"] = lo["metrics"]
    return out


def aggregate(ts: str, seeds, horizon: int) -> dict[str, Any]:
    """Mean/std over seeds of each metric per rung per split, plus mean QLIKE effects."""
    per_seed = [_rung_metrics(ts, horizon, s) for s in seeds]
    agg: dict[str, Any] = {}
    for rung in RUNGS:
        agg[rung] = {}
        for split in ("test_metrics", "validation_metrics"):
            agg[rung][split] = {}
            for metric in METRICS:
                vals = [ps[rung][split][metric] for ps in per_seed]
                agg[rung][split][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    fq = agg["FULL"]["test_metrics"]["qlike"]["mean"]
    agg["effects_qlike_test"] = {
        "graph": fq - agg["minus_graph"]["test_metrics"]["qlike"]["mean"],
        "gate": fq - agg["minus_gate"]["test_metrics"]["qlike"]["mean"],
        "news": fq - agg["minus_news"]["test_metrics"]["qlike"]["mean"],
    }
    return agg


def main(ts: str, seeds, horizons=(1, 5, 10, 22)) -> None:
    for h in horizons:
        try:
            a = aggregate(ts, seeds, h)
        except FileNotFoundError as exc:
            print(f"## h{h}: MISSING ({exc})")
            continue
        print(f"\n## h{h}  (seeds {list(seeds)})  TEST mean(std)")
        print("| rung | MSE(1e-6) | RMSE(1e-3) | MAE(1e-4) | R2 | QLIKE |")
        print("|" + "---|" * 6)
        for r in RUNGS:
            m = a[r]["test_metrics"]
            print(f"| {r} | {m['mse']['mean']*1e6:.2f}({m['mse']['std']*1e6:.2f}) "
                  f"| {m['rmse']['mean']*1e3:.3f}({m['rmse']['std']*1e3:.3f}) "
                  f"| {m['mae']['mean']*1e4:.2f}({m['mae']['std']*1e4:.2f}) "
                  f"| {m['r2']['mean']:.4f}({m['r2']['std']:.4f}) "
                  f"| {m['qlike']['mean']:.4f}({m['qlike']['std']:.4f}) |")
        print(f"effects_qlike_test (mean): {a['effects_qlike_test']}")


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _seeds = [int(x) for x in sys.argv[2].split(",")]
    _horizons = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (1, 5, 10, 22)
    main(_ts, _seeds, _horizons)
