"""CLI: compute the classical baseline suite and write the canonical JSON + MD report.

Usage:  python baselines/classical_baselines/code/run_classical_baselines.py <TS> [horizon]
Writes  docs/reports/classical_baselines_h<h>_<TS>.{json,md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from classical_baselines import _ROOT, run_all  # noqa: E402

_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")
_ORDER = ["Persistence", "EWMA", "HAR", "HARQ", "logHAR", "GARCH", "GJR-GARCH", "EGARCH"]


def to_ladder_schema(payload: dict) -> dict:
    """Mirror the ladder canonical schema so the paper can merge rows into one table."""

    results = payload["results"]
    rung = {"val": {}, "test": {}}
    for split in ("val", "test"):
        for name in _ORDER:
            if name not in results:
                continue
            rung[split][name] = {
                m: {"mean": float(results[name][split][m]), "std": 0.0} for m in _METRICS
            }
            rung[split][name]["n_obs"] = results[name][f"n_{split}_obs"]
    return {
        "timestamp": payload["timestamp"],
        "horizon": payload["horizon"],
        "basis": ("classical econometric baselines on the consistent Track-B ladder observation set "
                  "(identical val/test keys + targets + evaluate_records scorer as P0-G1)"),
        "n_val_obs": payload["n_val_obs"],
        "n_test_obs": payload["n_test_obs"],
        "n_tickers": payload["n_tickers"],
        "garch_excluded_tickers": payload["garch_excluded_tickers"],
        "garch_n_val_obs": payload["garch_n_val_obs"],
        "garch_n_test_obs": payload["garch_n_test_obs"],
        "notes": payload.get("notes", {}),
        "rung_metrics": rung,
    }


def _fmt(v: float, metric: str) -> str:
    return f"{v:.2f}" if metric == "directional_accuracy" else f"{v:.6g}"


def to_markdown(payload: dict) -> str:
    results = payload["results"]
    lines = [f"# Classical econometric volatility baselines (h{payload['horizon']})", ""]
    lines.append(payload["notes"]["basis_note"])
    lines.append("")
    lines.append(f"Observation set: val={payload['n_val_obs']}, test={payload['n_test_obs']}, "
                 f"tickers={payload['n_tickers']}. Metrics via the ladder `evaluate_records` scorer.")
    lines.append("")
    for split in ("val", "test"):
        lines.append(f"## {split.upper()} metrics")
        lines.append("")
        lines.append("| baseline | mse | rmse | mae | r2 | qlike | dir_acc |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in _ORDER:
            if name not in results:
                continue
            row = results[name][split]
            cells = " | ".join(_fmt(row[m], m) for m in _METRICS)
            lines.append(f"| {name} | {cells} |")
        lines.append("")
    lines.append("## Notes")
    for key, text in payload["notes"].items():
        if key != "basis_note":
            lines.append(f"- **{key}**: {text}")
    lines.append("")
    return "\n".join(lines)


def main(ts: str, horizon: int = 5) -> None:
    payload = run_all(horizon=horizon)
    payload["timestamp"] = ts
    payload["notes"] = {
        "basis_note": ("One basis for every baseline: leakage-safe chronological 70/15/15 split, "
                       "Parkinson-volatility target shift(-h), and the EXACT pooled val/test "
                       "observations (same keys + raw targets) scored by the same "
                       "`train.evaluate_records` as the deep-model ladder P0-G1."),
        "HARQ": ("HARQ uses a DAILY range-based realized-quarticity proxy RQ_d = sigma_d^2 because "
                 "the dataset is daily OHLCV (no intraday returns); the canonical BPQ-2016 5-min RQ "
                 "is not identified. This is an approximation, not the canonical HARQ."),
        "target_units": ("The processed `parkinson_variance` column is numerically the Parkinson "
                         "VARIANCE estimator sigma^2 = (ln(H/L))^2 / (4 ln 2) (verified corr=1.0 vs "
                         "raw OHLCV, median ~1.3e-4). Every baseline forecasts this daily realized-"
                         "variance quantity; EWMA smooths it directly and GARCH forecasts the "
                         "conditional return variance (same units)."),
        "GARCH_family": ("GARCH(1,1)/GJR/EGARCH fit per ticker on 100x close-to-close log returns; "
                         "params estimated on the train sample only (frozen). The h-step marginal "
                         "conditional variance (percent^2) is divided by 1e4 to recover the raw-"
                         "return variance, directly comparable to the Parkinson variance target."),
        "P0_anchor": ("P0 (pooled HAR on standardized features, from the ladder) is the deep-pipeline "
                      "HAR anchor; the HAR row here is a per-ticker OLS on raw volatility."),
        "GARCH_coverage": (
            "GARCH/GJR/EGARCH cover all {tot} of {tot} tickers on the full {gv} val / {gt} test "
            "observation set (exact ladder alignment), same as the vol-only baselines."
            if not payload["garch_excluded_tickers"] else
            "GARCH/GJR/EGARCH are scored on {gv} val / {gt} test observations ({nt} of {tot} "
            "tickers); tickers {ex} lack raw OHLCV so returns cannot be formed. Persistence/EWMA/"
            "HAR/HARQ/logHAR cover the full {v}/{t} observation set (exact ladder alignment).").format(
                gv=payload["garch_n_val_obs"], gt=payload["garch_n_test_obs"],
                nt=payload["n_tickers"] - len(payload["garch_excluded_tickers"]),
                tot=payload["n_tickers"], ex=payload["garch_excluded_tickers"],
                v=payload["n_val_obs"], t=payload["n_test_obs"]),
        "LPB_provenance": ("LPB raw OHLCV was recovered from the SSI iBoard API (2020-11-09..2026-"
                           "08-10); Parkinson variance recomputed from its High/Low reproduces "
                           "LPB_processed.csv (median |diff| 4.8e-6). SSI uses a different price-"
                           "adjustment convention than the other tickers (immaterial for return-"
                           "GARCH: log-returns are scale-invariant except on a few ex-dividend "
                           "days). This lifts the GARCH family from 32/33 to 33/33 tickers."),
        "GARCH_calendar_gap": ("18 LPB observations fall on holidays (Tet 2025, Apr/May 2025, New "
                               "Year 2026) present in the processed series but absent from the SSI "
                               "trading calendar; for those the GARCH forecast is carried forward "
                               "from the last trading origin (persistent conditional variance; "
                               "0.12% of test observations)."),
    }
    out_dir = _ROOT / "docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"classical_baselines_h{horizon}_{ts}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(to_ladder_schema(payload), indent=2), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(to_markdown(payload), encoding="utf-8")
    print(f"wrote {out_dir / stem}.json / .md", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
