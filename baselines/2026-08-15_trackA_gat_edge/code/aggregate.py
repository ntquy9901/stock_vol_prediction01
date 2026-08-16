"""Aggregate the Track-A GAT seeds: metric means + Diebold-Mariano verdicts.

Reads results/trackA_gat_seed{seed}_<TS>/ (ladder_metrics.json + per-obs test dumps in P0/NODE/GNN)
and writes results/trackA_gat_<TS>_summary.json + a markdown table.

DM: loss_a - loss_b per observation; NEGATIVE favors A. Tested: GNN vs NODE (does the graph help?),
GNN vs HAR, NODE vs HAR. QLIKE floor 1e-8 identical across compared models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _p in (str(PILOT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from diebold_mariano import diebold_mariano  # noqa: E402

HORIZON = 5
RUNGS = ("HAR", "NODE", "GNN")
_DUMP_DIR = {"HAR": "P0", "NODE": "NODE", "GNN": "GNN"}  # per-obs dump subdir per rung
_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
_EPSILON = 1e-8


def _run_dir(ts: str, seed: int) -> Path:
    return ROOT / "results" / f"trackA_gat_seed{seed}_{ts}"


def _mean_std_metrics(ts: str, seeds) -> dict[str, dict[str, dict[str, float]]]:
    collected: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed in seeds:
        ladder = json.loads((_run_dir(ts, seed) / "ladder_metrics.json").read_text(encoding="utf-8"))
        for rung in RUNGS:
            metrics = ladder["rungs"][rung]
            for split in ("validation_metrics", "test_metrics"):
                for key in _METRICS:
                    collected.setdefault(rung, {}).setdefault(split, {}).setdefault(key, []).append(
                        float(metrics[split][key]))
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for rung, splits in collected.items():
        summary[rung] = {}
        for split, keys in splits.items():
            for key, values in keys.items():
                summary[rung].setdefault(split, {})[key] = float(np.mean(values))
                summary[rung][split][f"{key}_std"] = float(np.std(values))
    return summary


def _load_rows(path: Path) -> dict[tuple[int, str], tuple[float, float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(int(r["ticker_id"]), str(r["target_date"])): (float(r["target_raw"]),
            float(r["prediction_raw"])) for r in rows}


def _ensemble(ts: str, rung: str, seeds) -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    per_seed = [_load_rows(_run_dir(ts, seed) / _DUMP_DIR[rung] / "predictions_test.json")
                for seed in seeds]
    keys = sorted(set(per_seed[0]))
    for seed_rows in per_seed[1:]:
        if set(seed_rows) != set(keys):
            raise ValueError(f"{rung} seed dumps cover different observations")
        if any(seed_rows[key][0] != per_seed[0][key][0] for key in keys):
            raise ValueError(f"{rung} seed dumps disagree on raw targets")
    targets = np.asarray([per_seed[0][key][0] for key in keys], dtype=float)
    predictions = np.asarray([np.mean([sr[key][1] for sr in per_seed]) for key in keys], dtype=float)
    return keys, targets, predictions


def _qlike(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.maximum(target, _EPSILON)
    prediction = np.maximum(prediction, _EPSILON)
    ratio = target / prediction
    return ratio - np.log(ratio) - 1.0


def _dm_pair(ts: str, rung_a: str, rung_b: str, seeds) -> dict[str, Any]:
    keys_a, target_a, pred_a = _ensemble(ts, rung_a, seeds)
    keys_b, target_b, pred_b = _ensemble(ts, rung_b, seeds)
    if keys_a != keys_b or not np.allclose(target_a, target_b):
        raise ValueError(f"DM {rung_a} vs {rung_b}: misaligned observations or targets")
    out: dict[str, Any] = {"n": len(keys_a)}
    for name, loss_a, loss_b in (
        ("qlike", _qlike(target_a, pred_a), _qlike(target_b, pred_b)),
        ("se", (target_a - pred_a) ** 2, (target_b - pred_b) ** 2),
    ):
        try:
            result = diebold_mariano(loss_a, loss_b, h=HORIZON)
        except ValueError as exc:
            out[name] = {"error": str(exc), "mean_loss_diff": float(np.mean(loss_a - loss_b))}
            continue
        favors = "tie" if result.mean_diff == 0 else ("A" if result.mean_diff < 0 else "B")
        out[name] = {"dm_hln": result.dm_hln, "p_value": result.p_value,
                     "mean_loss_diff": result.mean_diff, "favors": favors}
    return out


def main(ts: str, seeds=(42, 123, 2026)) -> None:
    metrics = _mean_std_metrics(ts, seeds)
    comparisons = {
        "GNN_vs_NODE": _dm_pair(ts, "GNN", "NODE", seeds),   # does the graph help?
        "GNN_vs_HAR": _dm_pair(ts, "GNN", "HAR", seeds),
        "NODE_vs_HAR": _dm_pair(ts, "NODE", "HAR", seeds),
    }
    summary = {"timestamp": ts, "seeds": list(seeds), "horizon": HORIZON,
               "metrics_mean_std": metrics, "diebold_mariano": comparisons}
    out_path = ROOT / "results" / f"trackA_gat_{ts}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"\n# Track-A GAT summary (seeds {tuple(seeds)}, h{HORIZON}, TS={ts})\n")
    print("| rung | split | RMSE | QLIKE | R2 | DirAcc% |")
    print("|" + "---|" * 5)
    for rung in RUNGS:
        for split in ("test_metrics", "validation_metrics"):
            m = metrics[rung][split]
            print(f"| {rung} | {split.split('_')[0]} | {m['rmse']:.6f} | {m['qlike']:.4f} "
                  f"| {m['r2']:.4f} | {m['directional_accuracy']:.2f} |")
    print("\n## Diebold-Mariano (seed-ensemble test; negative dm favors A)\n")
    print("| A vs B | metric | dm_hln | p_value | favors | n |")
    print("|" + "---|" * 6)
    for name, result in comparisons.items():
        for metric in ("qlike", "se"):
            row = result[metric]
            if "error" in row:
                print(f"| {name} | {metric} | n/a | n/a | {row['error']} | {result['n']} |")
                continue
            print(f"| {name} | {metric} | {row['dm_hln']:.3f} | {row['p_value']:.4f} "
                  f"| {row['favors']} | {result['n']} |")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1])
