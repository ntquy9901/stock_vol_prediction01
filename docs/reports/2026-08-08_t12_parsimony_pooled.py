"""T1.2: parsimony verdict on the POOLED regime (reuse A1 pooled cells).

Reads results/a1_pooled_seed{42,123,2026}/h5/{P0,P1,P2,P3}/results.json and reports,
per the 6 mandatory metrics, 3-seed mean +/- std per cell plus three contrasts with a
paired t-test (df = 2) and per-seed sign consistency:

    news effect = P2 vs P1  (does adding news help?)
    gate effect = P3 vs P2  (does the per-ticker gate add anything?)
    vs HAR      = P1/P2/P3 vs P0  (does any deep model beat the HAR linear baseline?)

Cell semantics (from baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code):
    P0 = HAR linear reference, P1 = pooled price-only LSTM,
    P2 = pooled price+news LSTM, P3 = P2 + per-ticker news gate.

Caveat: n = 3 seeds, 5 epochs, horizon 5 -- a screening signal, not a final result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 2026)
CONFIGS = ("P0", "P1", "P2", "P3")
METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
# True = smaller is better (error metrics); False = larger is better.
LOWER_IS_BETTER = {
    "mse": True,
    "rmse": True,
    "mae": True,
    "r2": False,
    "qlike": True,
    "directional_accuracy": False,
}


def load_metrics(seed: int, config: str) -> dict[str, float]:
    path = ROOT / f"results/a1_pooled_seed{seed}/h5/{config}/results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["validation_metrics"]
    return {m: float(metrics[m]) for m in METRICS}


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t on diffs (a - b). Returns (mean_diff, t). t is nan if sd == 0."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    md, sd = mean_std(diffs)
    if n < 2 or sd == 0.0:
        return md, float("nan")
    return md, md / (sd / math.sqrt(n))


def sign_consistency(diffs: list[float], lower_is_better: bool) -> tuple[int, int]:
    """Count seeds where the diff (treatment - reference) is an improvement."""
    if lower_is_better:
        n_improve = sum(1 for d in diffs if d < 0)
    else:
        n_improve = sum(1 for d in diffs if d > 0)
    return n_improve, len(diffs)


def _cells() -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for cfg in CONFIGS:
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in SEEDS:
            metrics = load_metrics(seed, cfg)
            for m in METRICS:
                per_metric[m].append(metrics[m])
        out[cfg] = per_metric
    return out


def _contrast(cells: dict, treat: str, ref: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in METRICS:
        a, b = cells[treat][m], cells[ref][m]
        md, t = paired_t(a, b)
        diffs = [x - y for x, y in zip(a, b)]
        n_improve, n_total = sign_consistency(diffs, LOWER_IS_BETTER[m])
        out[m] = {"mean_diff": md, "t": t, "n_improve": n_improve, "n_total": n_total}
    return out


def build_report() -> dict:
    cells = _cells()
    contrasts = {
        "news_P2_vs_P1": _contrast(cells, "P2", "P1"),
        "gate_P3_vs_P2": _contrast(cells, "P3", "P2"),
        "P1_vs_HAR": _contrast(cells, "P1", "P0"),
        "P2_vs_HAR": _contrast(cells, "P2", "P0"),
        "P3_vs_HAR": _contrast(cells, "P3", "P0"),
    }
    return {"cells": cells, "contrasts": contrasts}


def _fmt_cell(values: list[float]) -> str:
    m, s = mean_std(values)
    return f"{m:.6f}+/-{s:.6f}"


def _print_contrast(title: str, contrast: dict) -> None:
    print(f"\n### {title}")
    header = (
        f"{'metric':<24}{'mean_diff(treat-ref)':<24}"
        f"{'paired-t(df=2)':<18}{'sign_consistency':<18}{'direction':<10}"
    )
    print(header)
    print("-" * len(header))
    for m in METRICS:
        c = contrast[m]
        tstr = "nan" if math.isnan(c["t"]) else f"{c['t']:+.3f}"
        better = "lower=good" if LOWER_IS_BETTER[m] else "higher=good"
        sc = f"{c['n_improve']}/{c['n_total']} improve"
        print(f"{m:<24}{c['mean_diff']:<+24.6e}{tstr:<18}{sc:<18}{better:<10}")


def main() -> None:
    report = build_report()
    cells = report["cells"]

    print("=" * 100)
    print("T1.2  POOLED regime parsimony  (3-seed mean +/- std over seeds", SEEDS, ")")
    print("P0=HAR linear  P1=price-only LSTM  P2=price+news LSTM  P3=P2+per-ticker gate")
    print("Caveat: n=3, 5 epochs, horizon 5 -- screening signal, not a final result.")
    print("=" * 100)

    print(f"\n{'metric':<24}" + "".join(f"{c:<26}" for c in CONFIGS))
    print("-" * (24 + 26 * len(CONFIGS)))
    for m in METRICS:
        row = f"{m:<24}"
        for cfg in CONFIGS:
            row += f"{_fmt_cell(cells[cfg][m]):<26}"
        print(row)

    _print_contrast("NEWS EFFECT  (P2 vs P1)", report["contrasts"]["news_P2_vs_P1"])
    _print_contrast("GATE EFFECT  (P3 vs P2)", report["contrasts"]["gate_P3_vs_P2"])
    _print_contrast("P1 vs HAR  (P1 - P0)", report["contrasts"]["P1_vs_HAR"])
    _print_contrast("P2 vs HAR  (P2 - P0)", report["contrasts"]["P2_vs_HAR"])
    _print_contrast("P3 vs HAR  (P3 - P0)", report["contrasts"]["P3_vs_HAR"])


if __name__ == "__main__":
    main()
