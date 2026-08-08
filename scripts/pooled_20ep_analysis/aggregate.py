"""Aggregate the 20-epoch pooled pilot P0-P3 results across seeds with paired-t tests.

Reads each seed's ``validation_comparison.json`` (produced by ``run_pilot.py --phase
pooled``), computes per-config mean and sample standard deviation for the six
mandatory metrics, and runs paired-t tests (df = n_seeds - 1) for the news effect
(P2 vs P1), the gate effect (P3 vs P2), and each deep config against the HAR
reference (P0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

METRICS: tuple[str, ...] = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
CONFIGS: tuple[str, ...] = ("P0", "P1", "P2", "P3")


def load_seed_comparison(path: Path | str) -> dict[str, dict[str, float]]:
    """Return ``{config_name: {metric: value}}`` from one validation_comparison.json."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {row["config_name"]: row for row in payload["rows"]}
    missing = [config for config in CONFIGS if config not in rows]
    if missing:
        raise ValueError(f"{path} missing configs: {missing}")
    return {
        config: {metric: float(rows[config][metric]) for metric in METRICS}
        for config in CONFIGS
    }


def aggregate_metrics(
    seed_metrics: Sequence[Mapping[str, Mapping[str, float]]],
    configs: Sequence[str] = CONFIGS,
    metrics: Sequence[str] = METRICS,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return ``{config: {metric: (mean, sample_std)}}`` across seeds (ddof=1)."""

    if not seed_metrics:
        raise ValueError("seed_metrics must contain at least one seed")
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for config in configs:
        result[config] = {}
        for metric in metrics:
            values = np.asarray([seed[config][metric] for seed in seed_metrics], dtype=float)
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            result[config][metric] = (float(values.mean()), std)
    return result


def paired_t(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """Return (t_stat, p_value, mean_diff) of paired samples a-b (scipy ttest_rel)."""

    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    if a_array.shape != b_array.shape or a_array.size < 2:
        raise ValueError("paired_t requires two equal-length samples of length >= 2")
    outcome = stats.ttest_rel(a_array, b_array)
    return float(outcome.statistic), float(outcome.pvalue), float(np.mean(a_array - b_array))


def _series(seed_metrics: Sequence[Mapping[str, Mapping[str, float]]], config: str, metric: str):
    return [seed[config][metric] for seed in seed_metrics]


def build_report(seed_paths: Sequence[Path]) -> dict:
    """Load every seed, aggregate, and run the three paired-t comparison families."""

    seeds = [load_seed_comparison(path) for path in seed_paths]
    aggregated = aggregate_metrics(seeds)
    comparisons = {
        "news_effect_P2_vs_P1": ("P2", "P1"),
        "gate_effect_P3_vs_P2": ("P3", "P2"),
        "P1_vs_HAR_P0": ("P1", "P0"),
        "P2_vs_HAR_P0": ("P2", "P0"),
        "P3_vs_HAR_P0": ("P3", "P0"),
    }
    paired: dict[str, dict[str, dict[str, float]]] = {}
    for name, (left, right) in comparisons.items():
        paired[name] = {}
        for metric in METRICS:
            t_stat, p_value, mean_diff = paired_t(
                _series(seeds, left, metric), _series(seeds, right, metric)
            )
            paired[name][metric] = {"t": t_stat, "p": p_value, "mean_diff": mean_diff}
    return {"n_seeds": len(seeds), "aggregated": aggregated, "paired_t": paired}


def _format_report(report: dict) -> str:
    lines: list[str] = [f"n_seeds = {report['n_seeds']} (paired-t df = {report['n_seeds'] - 1})", ""]
    header = f"{'config':<7}" + "".join(f"{metric:>24}" for metric in METRICS)
    lines.append(header)
    for config in CONFIGS:
        cells = []
        for metric in METRICS:
            mean, std = report["aggregated"][config][metric]
            cells.append(f"{mean:.6g}+-{std:.3g}")
        lines.append(f"{config:<7}" + "".join(f"{cell:>24}" for cell in cells))
    lines.append("")
    lines.append("Paired-t (mean_diff, t, p):")
    for name, metrics in report["paired_t"].items():
        lines.append(f"  {name}")
        for metric, stat in metrics.items():
            lines.append(
                f"    {metric:<24} diff={stat['mean_diff']:+.6g}  t={stat['t']:+.4g}  p={stat['p']:.4g}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_paths", nargs="+", type=Path,
                        help="validation_comparison.json paths, one per seed")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.seed_paths)
    print(_format_report(report))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
