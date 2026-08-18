"""Cross-seq Diebold-Mariano: same rung, different LSTM lookback (SEQ), on the shared held-out test.

Reuses the delivered DM machinery (``dm_report._load_rows`` / ``dm_report.LOSSES`` with the shared
QLIKE floor, and ``diebold_mariano`` with HLN small-sample correction + HAC lag h-1). For a given
rung it loads the per-observation test dumps from two retrain runs that differ only in SEQ and pairs
them by (ticker_id, target_date). Windows are built WITHIN each split frame (leakage-safe), so the
earliest test target sits at index SEQ+h-1 inside the test frame: a shorter lookback yields EXTRA
early test observations the longer lookback lacks (e.g. seq5 test_obs 9701 vs seq22 9272). The
comparison is therefore made on the INTERSECTION of the two runs' test keys (common target-dates),
with targets asserted equal on that intersection; predictions differ, targets align -> paired DM valid.

Sign convention: A = the SHORTER lookback (e.g. seq5), B = the reference (seq22). Negative dm favors A
(shorter lookback lower loss); positive favors B (reference lower loss). * marks p < 0.05.

Run: python scripts/seq_lookback/dm_seq_compare.py <TS_A> <TS_B> [rung] [seeds_csv] [horizons_csv]
  e.g. dm_seq_compare.py 2026-08-18_1200_seq5 2026-08-18_1200_seq22 FULL 42 1,5,10,22
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "baselines" / "2026-08-15_volatility" / "code",
           _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dm_report  # noqa: E402
from diebold_mariano import diebold_mariano  # noqa: E402

RESULTS = _ROOT / "results"


def _load_run(ts: str, horizon: int, rung: str, seeds) -> dict:
    """Seed-ensemble the retrain dumps for one run -> {key: (target, mean_prediction)}."""
    per_seed = [dm_report._load_rows(RESULTS / f"volatility_retrain_h{horizon}_seed{s}_{ts}" /
                                     dm_report.DUMP_DIR[rung] / "predictions_test.json") for s in seeds]
    keys = set(per_seed[0])
    for rows in per_seed[1:]:
        if set(rows) != keys:
            raise ValueError(f"{rung} {ts}: seed dumps cover different observations")
    return {k: (per_seed[0][k][0], float(np.mean([r[k][1] for r in per_seed]))) for k in keys}


def dm_seq(ts_a: str, ts_b: str, horizon: int, rung: str, seeds, loss: str) -> dict[str, Any]:
    run_a = _load_run(ts_a, horizon, rung, seeds)
    run_b = _load_run(ts_b, horizon, rung, seeds)
    common = sorted(set(run_a) & set(run_b))  # shorter lookback has extra early test obs -> intersect
    if not common:
        raise ValueError(f"DM {rung} {ts_a} vs {ts_b} h{horizon}: no common test observations")
    target_a = np.asarray([run_a[k][0] for k in common], dtype=float)
    target_b = np.asarray([run_b[k][0] for k in common], dtype=float)
    pred_a = np.asarray([run_a[k][1] for k in common], dtype=float)
    pred_b = np.asarray([run_b[k][1] for k in common], dtype=float)
    if not np.allclose(target_a, target_b):
        raise ValueError(f"DM {rung} {ts_a} vs {ts_b} h{horizon}: targets differ on common observations")
    loss_fn = dm_report.LOSSES[loss]
    result = diebold_mariano(loss_fn(target_a, pred_a), loss_fn(target_b, pred_b), h=horizon)
    favors = "tie" if result.mean_diff == 0 else ("A(short)" if result.mean_diff < 0 else "B(ref)")
    # report each run's own mean loss for context
    return {"dm_hln": result.dm_hln, "p_value": result.p_value, "mean_diff": result.mean_diff,
            "favors": favors, "n": len(common),
            "mean_loss_a": float(np.mean(loss_fn(target_a, pred_a))),
            "mean_loss_b": float(np.mean(loss_fn(target_b, pred_b)))}


def main(ts_a: str, ts_b: str, rung: str, seeds, horizons) -> None:
    if len(seeds) == 1:
        print(f"# WARNING: single-seed DM (seeds={list(seeds)}); pass a seeds CSV "
              f"(e.g. 42,123,2026) for the 3-seed ensemble.", file=sys.stderr)
    print(f"# Cross-seq DM (HLN): rung={rung}  A={ts_a} (short)  B={ts_b} (ref)  seeds={list(seeds)}")
    print("# negative dm favors A (shorter lookback lower loss); *=p<.05")
    print("# mean-loss / RMSE columns are computed on the COMMON (intersected) test observations, not each run's full test set\n")
    print("| h | QLIKE dm(p) | QLIKE meanA/B(common) | SE dm(p) | RMSE_A/B(common) | AE dm(p) | n |")
    print("|---|---|---|---|---|---|---|")
    for h in horizons:
        cells: dict[str, Any] = {}
        try:
            for loss in ("qlike", "se", "ae"):
                cells[loss] = dm_seq(ts_a, ts_b, h, rung, seeds, loss)
            q, se, ae = cells["qlike"], cells["se"], cells["ae"]
            star = lambda r: "*" if r["p_value"] < 0.05 else ""  # noqa: E731
            rmse_a = se["mean_loss_a"] ** 0.5
            rmse_b = se["mean_loss_b"] ** 0.5
            print(f"| {h} "
                  f"| {q['dm_hln']:+.2f}({q['p_value']:.3f}){star(q)} "
                  f"| {q['mean_loss_a']:.4f}/{q['mean_loss_b']:.4f} "
                  f"| {se['dm_hln']:+.2f}({se['p_value']:.3f}){star(se)} "
                  f"| {rmse_a:.4f}/{rmse_b:.4f} "
                  f"| {ae['dm_hln']:+.2f}({ae['p_value']:.3f}){star(ae)} "
                  f"| {q['n']} |")
        except (FileNotFoundError, ValueError) as exc:
            print(f"| {h} | err:{exc} | | | | | |")


if __name__ == "__main__":
    _ts_a = sys.argv[1]
    _ts_b = sys.argv[2]
    _rung = sys.argv[3] if len(sys.argv) > 3 else "FULL"
    _seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [42]
    _horizons = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [1, 5, 10, 22]
    main(_ts_a, _ts_b, _rung, _seeds, _horizons)
