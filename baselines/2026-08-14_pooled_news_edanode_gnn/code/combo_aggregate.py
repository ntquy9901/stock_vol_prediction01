"""Aggregate the 3-seed combo ladder: metric means + Diebold-Mariano verdicts.

Reads results/pooled_news_edanode_seed{seed}_<TS>/h5/ (ladder_metrics.json + per-observation test
dumps for P0/P3/G1) and writes results/pooled_news_edanode_<TS>_summary.json + a markdown table.

DM convention (diebold_mariano): loss_a - loss_b per observation; a NEGATIVE statistic means model
A carries the smaller loss. We test G1 (the combo, A) vs P0 (HAR, B), G1 (A) vs P3 (graph-off, B),
and P3 (A) vs P0 (B). P1/P2 have no per-observation dump (B's pooled rung), reported by metric only.
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

SEEDS = (42, 123, 2026)
HORIZON = 5
RUNGS = ("P0", "P1", "P2", "P3", "G1")
_LABELS = {"P0": "P0 HAR", "P1": "P1 price5", "P2": "P2 +news", "P3": "P3 +gate (graph off)",
           "G1": "G1 combo (+vol->PK)"}
_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
_EPSILON = 1e-8


def _run_dir(ts: str, seed: int) -> Path:
    return ROOT / "results" / f"pooled_news_edanode_seed{seed}_{ts}" / f"h{HORIZON}"


def _mean_std_metrics(ts: str) -> dict[str, dict[str, dict[str, float]]]:
    collected: dict[str, dict[str, dict[str, list[float]]]] = {}
    for seed in SEEDS:
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


def _ensemble(ts: str, rung: str) -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    per_seed = [_load_rows(_run_dir(ts, seed) / rung / "predictions_test.json") for seed in SEEDS]
    keys = sorted(set(per_seed[0]))
    for seed_rows in per_seed[1:]:
        if set(seed_rows) != set(keys):
            raise ValueError(f"{rung} seed prediction dumps cover different observations")
        if any(seed_rows[key][0] != per_seed[0][key][0] for key in keys):
            raise ValueError(f"{rung} seed prediction dumps disagree on raw targets")
    targets = np.asarray([per_seed[0][key][0] for key in keys], dtype=float)
    predictions = np.asarray([np.mean([seed_rows[key][1] for seed_rows in per_seed]) for key in keys],
                             dtype=float)
    return keys, targets, predictions


def _qlike(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.maximum(target, _EPSILON)
    prediction = np.maximum(prediction, _EPSILON)
    ratio = target / prediction
    return ratio - np.log(ratio) - 1.0


def _dm_pair(ts: str, rung_a: str, rung_b: str) -> dict[str, Any]:
    keys_a, target_a, pred_a = _ensemble(ts, rung_a)
    keys_b, target_b, pred_b = _ensemble(ts, rung_b)
    if keys_a != keys_b or not np.allclose(target_a, target_b):
        raise ValueError(f"DM {rung_a} vs {rung_b}: misaligned observation sets or targets")
    out: dict[str, Any] = {"n": len(keys_a)}
    for name, loss_a, loss_b in (
        ("qlike", _qlike(target_a, pred_a), _qlike(target_b, pred_b)),
        ("se", (target_a - pred_a) ** 2, (target_b - pred_b) ** 2),
    ):
        # A degenerate pair (zero loss-differential variance) makes the DM statistic undefined;
        # record it instead of aborting the whole summary.
        try:
            result = diebold_mariano(loss_a, loss_b, h=HORIZON)
        except ValueError as exc:
            out[name] = {"error": str(exc), "mean_loss_diff": float(np.mean(loss_a - loss_b))}
            continue
        favors = "tie" if result.mean_diff == 0 else ("A" if result.mean_diff < 0 else "B")
        out[name] = {"dm_hln": result.dm_hln, "p_value": result.p_value,
                     "mean_loss_diff": result.mean_diff, "favors": favors}
    return out


def main(ts: str) -> None:
    metrics = _mean_std_metrics(ts)
    comparisons = {
        "G1_vs_P0": _dm_pair(ts, "G1", "P0"),
        "G1_vs_P3": _dm_pair(ts, "G1", "P3"),
        "P3_vs_P0": _dm_pair(ts, "P3", "P0"),
    }
    summary = {"timestamp": ts, "seeds": list(SEEDS), "horizon": HORIZON,
               "metrics_mean_std": metrics, "diebold_mariano": comparisons,
               "dm_note": "P1/P2 have no per-observation dump (pooled rung); reported by metric only."}
    out_path = ROOT / "results" / f"pooled_news_edanode_{ts}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"\n# Combo ladder summary (seeds {SEEDS}, h{HORIZON}, TS={ts})\n")
    print("| rung | split | RMSE | QLIKE | MAE | R2 | DirAcc% |")
    print("|" + "---|" * 6)
    for rung in RUNGS:
        for split in ("test_metrics", "validation_metrics"):
            m = metrics[rung][split]
            print(f"| {_LABELS[rung]} | {split.split('_')[0]} | {m['rmse']:.6f} | {m['qlike']:.4f} "
                  f"| {m['mae']:.6f} | {m['r2']:.4f} | {m['directional_accuracy']:.2f} |")
    print("\n## Diebold-Mariano (seed-ensemble test predictions; negative dm favors A)\n")
    print("| comparison (A vs B) | metric | dm_hln | p_value | favors | n |")
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


if __name__ == "__main__":
    main(sys.argv[1])
