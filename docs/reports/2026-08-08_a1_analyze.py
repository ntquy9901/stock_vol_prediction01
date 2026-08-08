"""Aggregate the A1 pooled-vs-common-date cells into a comparison table.

Reads results/a1_{regime}_seed{seed}/h5/{config}/results.json for regime in
{pooled, commondate}, seed in {42,123,2026}, config in {P0,P1,P2,P3} and prints
3-seed mean+/-std per metric plus a paired t-test (pooled - commondate) across seeds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 2026)
CONFIGS = ("P0", "P1", "P2", "P3")
METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
REGIME_TAG = {"pooled": "pooled", "common-date": "commondate"}


def load_metrics(tag: str, seed: int, config: str) -> dict[str, float] | None:
    path = ROOT / f"results/a1_{tag}_seed{seed}/h5/{config}/results.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("validation_metrics", {})
    if not metrics:
        return None
    return {m: float(metrics[m]) for m in METRICS}


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def paired_t(pooled: list[float], common: list[float]) -> tuple[float, float]:
    diffs = [p - c for p, c in zip(pooled, common)]
    n = len(diffs)
    md, sd = mean_std(diffs)
    if n < 2 or sd == 0.0:
        return md, float("nan")
    t = md / (sd / math.sqrt(n))
    return md, t


def collect(regime: str) -> dict[str, dict[str, list[float]]]:
    tag = REGIME_TAG[regime]
    out: dict[str, dict[str, list[float]]] = {}
    for config in CONFIGS:
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in SEEDS:
            metrics = load_metrics(tag, seed, config)
            if metrics is None:
                continue
            for m in METRICS:
                per_metric[m].append(metrics[m])
        out[config] = per_metric
    return out


def main() -> None:
    pooled = collect("pooled")
    common = collect("common-date")

    print("=" * 100)
    print("A1: pooled vs common-date  (3-seed mean +/- std over seeds", SEEDS, ")")
    print("=" * 100)
    for config in CONFIGS:
        print(f"\n### {config}")
        header = f"{'metric':<24}{'pooled':<26}{'common-date':<26}{'delta(pool-comm)':<20}{'paired-t':>10}"
        print(header)
        print("-" * len(header))
        for m in METRICS:
            pv = pooled[config][m]
            cv = common[config][m]
            if not pv or not cv:
                print(f"{m:<24}{'(missing)':<26}{'(missing)':<26}")
                continue
            pm, ps = mean_std(pv)
            cm, cs = mean_std(cv)
            md, t = paired_t(pv, cv)
            tstr = "nan" if math.isnan(t) else f"{t:+.3f}"
            print(f"{m:<24}{pm:>10.6f}+/-{ps:<11.6f}{cm:>10.6f}+/-{cs:<11.6f}{md:>+18.6f}{tstr:>12}")

    # Compact verdict lines for QLIKE / RMSE / R2 on P1 and P2 vs P0.
    print("\n" + "=" * 100)
    print("VERDICT INPUTS (means over seeds)")
    print("=" * 100)
    for config in ("P0", "P1", "P2"):
        for m in ("qlike", "rmse", "r2", "directional_accuracy"):
            pv = pooled[config][m]
            cv = common[config][m]
            if pv and cv:
                pm, _ = mean_std(pv)
                cm, _ = mean_std(cv)
                print(f"{config:<4} {m:<22} pooled={pm:.6f}  common-date={cm:.6f}  n_pool={len(pv)} n_comm={len(cv)}")


if __name__ == "__main__":
    main()
