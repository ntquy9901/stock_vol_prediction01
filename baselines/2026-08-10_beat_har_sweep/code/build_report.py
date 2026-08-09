"""Assemble the consolidated beat-HAR sweep results table from per-config results + analysis.json.

Reads results/beat_har_<C>_<TS>/seed*/results.json (C1/C2/C3/C6) and .../seed*/k*/results.json (C5),
plus results/beat_har_sweep_<TS>/analysis.json (DM + paired-t vs P0), and prints a machine-checkable
JSON summary (mean/std over seeds of all 6 metrics, val+test, per config, vs the HAR bar). The prose
report is written by hand from this JSON so no number is invented.

Usage:  python .../code/build_report.py <TS>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
BAR = {"P0_qlike": 0.5676, "classicalHAR_qlike": 0.5793, "HARQ_rmse": 0.0022891, "HARQ_r2": 0.76682}


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def _collect(result_files: list[Path]) -> dict[str, dict]:
    val: dict[str, list[float]] = {m: [] for m in METRICS}
    test: dict[str, list[float]] = {m: [] for m in METRICS}
    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for m in METRICS:
            val[m].append(payload["validation_metrics"][m])
            test[m].append(payload["test_metrics"][m])
    return {"validation": {m: _mean_std(val[m]) for m in METRICS},
            "test": {m: _mean_std(test[m]) for m in METRICS}}


def summarize(ts: str) -> dict:
    results = _ROOT / "results"
    analysis_path = results / f"beat_har_sweep_{ts}" / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
    out: dict = {"ts": ts, "bar": BAR, "configs": {}}
    for config in ("C1", "C2", "C3", "C6", "C5"):
        matches = sorted(results.glob(f"beat_har_{config}_{ts}*"))
        if not matches:
            out["configs"][config] = {"status": "no_results"}
            continue
        cdir = matches[0]
        if config == "C5":
            best = (analysis.get("configs", {}).get("C5", {}) or {}).get("best_k")
            per_k = {}
            for k_dir in sorted({p.parent.name for p in cdir.glob("seed*/k*/results.json")}):
                files = sorted(cdir.glob(f"seed*/{k_dir}/results.json"))
                per_k[k_dir] = _collect(files)
            out["configs"][config] = {"per_k": per_k, "best_k": best}
        else:
            files = sorted(cdir.glob("seed*/results.json"))
            if not files:
                out["configs"][config] = {"status": "no_results"}
                continue
            out["configs"][config] = _collect(files)
        out["configs"][config]["significance"] = analysis.get("configs", {}).get(config, {})
    dest = results / f"beat_har_sweep_{ts}" / "report_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    summarize(sys.argv[1])
