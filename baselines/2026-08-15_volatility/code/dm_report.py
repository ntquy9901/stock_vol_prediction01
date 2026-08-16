"""Diebold-Mariano (HLN) verdicts for the volatility leave-one-out study.

Reads per-observation held-out test prediction dumps from
``results/volatility_ablation_h{h}_seed{s}_<TS>/{dump_dir}/predictions_test.json``, optionally
seed-ensembles the predictions (mean over seeds on the shared observation keys), computes the
per-observation QLIKE loss, and runs a Diebold-Mariano test (HLN small-sample correction, HAC lag
h-1) of FULL against each comparator. Rung -> dump dir:
HAR->P0, FULL->FULL, minus_graph->minus_graph, minus_gate->minus_gate, minus_news->minus_news,
LSTM_only->lstm_only. QLIKE uses one shared floor (1e-8) for every rung so the compared models are on
an identical positivity basis.

Run: python <.../code/dm_report.py> <TS> <horizon> [seeds_csv]
  e.g. dm_report.py 2026-08-15_085544_loo 5 42        (single seed)
       dm_report.py 2026-08-15_085544_loo 5 42,123,2026   (seed ensemble)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (_ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from diebold_mariano import diebold_mariano  # noqa: E402

RESULTS = _ROOT / "results"
DUMP_DIR = {"HAR": "P0", "FULL": "FULL", "minus_graph": "minus_graph", "minus_gate": "minus_gate",
            "minus_news": "minus_news", "LSTM_only": "lstm_only"}
_EPSILON = 1e-8


def _load_rows(path: Path) -> dict[tuple[int, str], tuple[float, float]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(int(r["ticker_id"]), str(r["target_date"])): (float(r["target_raw"]),
            float(r["prediction_raw"])) for r in rows}


def _ensemble(ts: str, horizon: int, rung: str, seeds) -> tuple[list, np.ndarray, np.ndarray]:
    per_seed = [_load_rows(RESULTS / f"volatility_ablation_h{horizon}_seed{s}_{ts}" /
                           DUMP_DIR[rung] / "predictions_test.json") for s in seeds]
    keys = sorted(set(per_seed[0]))
    for rows in per_seed[1:]:
        if set(rows) != set(keys):
            raise ValueError(f"{rung}: seed dumps cover different observations")
    targets = np.asarray([per_seed[0][k][0] for k in keys], dtype=float)
    predictions = np.asarray([np.mean([r[k][1] for r in per_seed]) for k in keys], dtype=float)
    return keys, targets, predictions


def _qlike(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.maximum(target, _EPSILON)
    prediction = np.maximum(prediction, _EPSILON)
    ratio = target / prediction
    return ratio - np.log(ratio) - 1.0


# per-observation loss families: QLIKE (proportional), squared error (MSE/RMSE/R2 family),
# absolute error (MAE). DM on each answers "does model A beat B under that loss".
LOSSES = {
    "qlike": _qlike,
    "se": lambda t, p: (t - p) ** 2,
    "ae": lambda t, p: np.abs(t - p),
}


def dm_pair(ts: str, horizon: int, rung_a: str, rung_b: str, seeds, loss: str = "qlike") -> dict[str, Any]:
    keys_a, target_a, pred_a = _ensemble(ts, horizon, rung_a, seeds)
    keys_b, target_b, pred_b = _ensemble(ts, horizon, rung_b, seeds)
    if keys_a != keys_b or not np.allclose(target_a, target_b):
        raise ValueError(f"DM {rung_a} vs {rung_b}: misaligned observations or targets")
    loss_fn = LOSSES[loss]
    result = diebold_mariano(loss_fn(target_a, pred_a), loss_fn(target_b, pred_b), h=horizon)
    favors = "tie" if result.mean_diff == 0 else ("A" if result.mean_diff < 0 else "B")
    return {"dm_hln": result.dm_hln, "p_value": result.p_value, "mean_diff": result.mean_diff,
            "favors": favors, "n": len(keys_a)}


def main(ts: str, horizon: int, seeds) -> None:
    print(f"# DM (HLN) h{horizon} seeds={list(seeds)} -- dm_hln(p); negative dm favors A=FULL; *=p<.05\n")
    print("| A vs B | QLIKE | SE (MSE/RMSE/R2) | AE (MAE) | n |")
    print("|---|---|---|---|---|")
    for rung_b in ("HAR", "minus_graph", "minus_gate", "minus_news", "LSTM_only"):
        cells = []
        n = "n/a"
        for loss in ("qlike", "se", "ae"):
            try:
                r = dm_pair(ts, horizon, "FULL", rung_b, seeds, loss=loss)
                n = r["n"]
                cells.append(f"{r['dm_hln']:+.2f}({r['p_value']:.2f})" + ("*" if r["p_value"] < 0.05 else ""))
            except (FileNotFoundError, ValueError) as exc:
                cells.append(f"err:{exc}")
        print(f"| FULL vs {rung_b} | {cells[0]} | {cells[1]} | {cells[2]} | {n} |")


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _h = int(sys.argv[2])
    _seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42]
    main(_ts, _h, _seeds)
