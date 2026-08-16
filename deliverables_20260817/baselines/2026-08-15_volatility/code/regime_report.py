"""P3: regime-split (calm / turbulent) metrics + Diebold-Mariano for the volatility study.

Market-regime analysis from Zhang et al. (arXiv:2308.01419): a model's value often concentrates on
high-volatility days, which a pooled average hides. This is a POST-HOC pass over the same held-out
prediction dumps the leave-one-out study already writes (``predictions_test.json`` per rung) — no
retraining. Observations are split by realized target volatility (top ``turbulent_frac`` = turbulent);
the 5 raw-scale metrics + QLIKE and DM(HLN) are recomputed per regime on one shared positivity floor.

Run: python <.../code/regime_report.py> <TS> <horizon> [seeds_csv] [turbulent_frac]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parent
_ROOT = CODE.resolve().parents[2]
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import dm_report  # noqa: E402  (reuse _ensemble/_qlike/LOSSES/DUMP_DIR + RESULTS)
from diebold_mariano import diebold_mariano  # noqa: E402


def split_regime(targets: np.ndarray, turbulent_frac: float = 0.10) -> np.ndarray:
    """Boolean mask (True = turbulent) for the top ``turbulent_frac`` of observations by target
    volatility. ``0 < turbulent_frac < 1``."""
    if not 0.0 < turbulent_frac < 1.0:
        raise ValueError(f"turbulent_frac must be in (0, 1), got {turbulent_frac}")
    targets = np.asarray(targets, dtype=float)
    threshold = np.quantile(targets, 1.0 - turbulent_frac)
    return targets >= threshold


def regime_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """5 raw-scale metrics + QLIKE on one subset. R2 uses the pooled global mean (project basis);
    NaN when it is undefined (n<2 or zero target variance). DirAcc is intentionally omitted: it is
    a sign-of-change metric and is not meaningful on a non-contiguous regime subset."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    n = int(target.size)
    if n == 0:
        return {"n": 0, "mse": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                "r2": float("nan"), "qlike": float("nan")}
    err = prediction - target
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if (n < 2 or ss_tot == 0.0) else 1.0 - float(np.sum(err ** 2)) / ss_tot
    qlike = float(np.mean(dm_report._qlike(target, prediction)))
    return {"n": n, "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(err))),
            "r2": r2, "qlike": qlike}


def regime_dm(target: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, horizon: int,
              loss: str = "qlike") -> dict[str, Any]:
    """DM(HLN, HAC lag h-1) of model A vs B on one regime subset. Negative dm/mean_diff favors A."""
    loss_fn = dm_report.LOSSES[loss]
    result = diebold_mariano(loss_fn(target, pred_a), loss_fn(target, pred_b), h=horizon)
    favors = "tie" if result.mean_diff == 0 else ("A" if result.mean_diff < 0 else "B")
    return {"dm_hln": result.dm_hln, "p_value": result.p_value, "mean_diff": result.mean_diff,
            "favors": favors, "n": int(target.size)}


def run_regime(ts: str, horizon: int, seeds, comparators=("HAR", "minus_graph", "minus_gate",
               "minus_news", "LSTM_only"), turbulent_frac: float = 0.10) -> dict[str, Any]:
    """Load the aligned FULL + comparator dumps, split by FULL's target volatility, and compute
    per-regime metrics (every rung) + DM (FULL vs each comparator). Returns a structured dict."""
    ref_keys, target, full_pred = dm_report._ensemble(ts, horizon, "FULL", seeds)
    mask = split_regime(target, turbulent_frac)
    preds = {"FULL": full_pred}
    for rung in comparators:
        keys, tgt, pred = dm_report._ensemble(ts, horizon, rung, seeds)
        if keys != ref_keys or not np.allclose(tgt, target):
            raise ValueError(f"{rung}: observations/targets misaligned with FULL")
        preds[rung] = pred
    out: dict[str, Any] = {"horizon": horizon, "seeds": list(seeds), "turbulent_frac": turbulent_frac,
                           "n_total": int(target.size), "n_turbulent": int(mask.sum()), "regimes": {}}
    for name, sel in (("calm", ~mask), ("turbulent", mask)):
        metrics = {rung: regime_metrics(target[sel], preds[rung][sel]) for rung in preds}
        dm = {rung: regime_dm(target[sel], full_pred[sel], preds[rung][sel], horizon)
              for rung in comparators}
        out["regimes"][name] = {"metrics": metrics, "dm_full_vs": dm}
    return out


def main(ts: str, horizon: int, seeds, turbulent_frac: float = 0.10) -> None:  # pragma: no cover
    report = run_regime(ts, horizon, seeds, turbulent_frac=turbulent_frac)
    print(f"# Regime split h{horizon} seeds={report['seeds']} turbulent_frac={turbulent_frac} "
          f"(n_total={report['n_total']}, n_turbulent={report['n_turbulent']})\n")
    for regime, block in report["regimes"].items():
        print(f"## {regime}")
        print("| rung | n | MSE | RMSE | MAE | R2 | QLIKE | DM(FULL vs) qlike |")
        print("|---|---|---|---|---|---|---|---|")
        for rung, m in block["metrics"].items():
            dm = block["dm_full_vs"].get(rung)
            dm_cell = "-" if dm is None else (f"{dm['dm_hln']:+.2f}({dm['p_value']:.2f})"
                                              + ("*" if dm["p_value"] < 0.05 else ""))
            print(f"| {rung} | {m['n']} | {m['mse']:.5f} | {m['rmse']:.5f} | {m['mae']:.5f} | "
                  f"{m['r2']:.4f} | {m['qlike']:.5f} | {dm_cell} |")
        print()


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _h = int(sys.argv[2])
    _seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42]
    _frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
    main(_ts, _h, _seeds, _frac)
