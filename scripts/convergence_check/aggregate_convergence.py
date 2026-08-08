"""Aggregate 6-metric mean+/-std and paired t-tests for the 4 variants x 3 seeds,
for the 2026-08-06 convergence check (docs/reports/2026-08-06_convergence_check_40epoch.md).

Reads a run-mapping JSON (variant -> seed -> results_dir), pulls test_metrics from each
run's results file (training_results.json for backbone/no_graph, results.json for
full/no_gate), and prints:
  - per-variant mean+/-std over the 3 seeds for the 6 mandatory metrics;
  - the three paired t-tests (n=3, df=2, two-sided 5% critical value 4.303):
      Ablation 1 (news branch): backbone - full
      Ablation 2 (graph):       no_graph - backbone
      Ablation 3 (gate):        full - no_gate
  - per-variant best_epoch per seed (convergence diagnosis).

Usage:
  python scripts/convergence_check/aggregate_convergence.py <mapping.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"]
TCRIT_N3 = 4.303  # two-sided 5%, df=2


def _load(run_dir: str) -> dict:
    p = ROOT / run_dir
    for fname in ("training_results.json", "results.json"):
        f = p / fname
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no results json in {p}")


def _best_epoch(rec: dict) -> int | None:
    ts = rec.get("training_summary")
    if ts and "best_epoch" in ts:
        return ts["best_epoch"]
    return rec.get("best_epoch")


def collect(mapping: dict) -> dict:
    """variant -> {'metrics': {metric: [seed-ordered values]}, 'best_epoch': {seed: e}}"""
    out = {}
    for variant, seeds in mapping.items():
        vals = {m: [] for m in METRICS}
        best = {}
        for seed in sorted(seeds, key=int):
            rec = _load(seeds[seed])
            tm = rec["test_metrics"]
            for m in METRICS:
                vals[m].append(float(tm[m]))
            best[seed] = _best_epoch(rec)
        out[variant] = {"metrics": vals, "best_epoch": best}
    return out


def paired_t(a: list, b: list) -> float:
    """Paired t across seeds for (a - b). Returns nan if std==0."""
    d = np.array(a) - np.array(b)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(d.mean() / (sd / np.sqrt(len(d))))


def main():
    mapping = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    data = collect(mapping)

    print("=" * 92)
    print("PER-VARIANT best_epoch PER SEED")
    print("=" * 92)
    for v, d in data.items():
        print(f"  {v:12} " + "  ".join(f"seed{s}={d['best_epoch'][s]}" for s in sorted(d['best_epoch'], key=int)))

    print("\n" + "=" * 92)
    print("PER-VARIANT MEAN +/- STD (n=3)")
    print("=" * 92)
    hdr = f"{'metric':<16}" + "".join(f"{v:>22}" for v in data)
    print(hdr)
    for m in METRICS:
        row = f"{m:<16}"
        for v in data:
            arr = np.array(data[v]["metrics"][m])
            row += f"{arr.mean():>12.6f}+/-{arr.std(ddof=1):<8.6f}"
        print(row)

    ablations = [
        ("Ablation1_news (backbone - full)", "backbone", "full"),
        ("Ablation2_graph (no_graph - backbone)", "no_graph", "backbone"),
        ("Ablation3_gate  (full - no_gate)", "full", "no_gate"),
    ]
    print("\n" + "=" * 92)
    print(f"PAIRED t-TESTS (n=3, df=2, |t|>{TCRIT_N3} passes 5%)")
    print("=" * 92)
    for name, a, b in ablations:
        if a not in data or b not in data:
            continue
        print(f"\n{name}")
        for m in METRICS:
            t = paired_t(data[a]["metrics"][m], data[b]["metrics"][m])
            mark = "SIG" if (t == t and abs(t) > TCRIT_N3) else "ns"
            print(f"    {m:<24} t={t:>8.3f}  {mark}")


if __name__ == "__main__":
    main()
