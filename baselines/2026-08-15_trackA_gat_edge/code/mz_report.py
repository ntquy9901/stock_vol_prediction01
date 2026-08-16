"""Mincer-Zarnowitz (MZ) forecast-efficiency evaluation for the trackA leave-one-out study.

For each rung and horizon we take the realized target y and the seed-ensembled forecast x
(reused from ``dm_report._ensemble``) and run the OLS regression y = a + b*x + e. The MZ joint
test H0: a=0 AND b=1 (an unbiased/efficient forecast) is a Wald statistic
W = (theta - theta0)' V^{-1} (theta - theta0), theta=(a,b), theta0=(0,1),
V = sigma2 * (X'X)^{-1}, sigma2 = RSS/(n-2); W ~ chi-square(2). A near-zero intercept, a slope
near one, and a large p-value indicate a well-calibrated forecast. MZ measures calibration/bias,
complementary to the QLIKE/MSE accuracy comparisons in ``dm_report.py``.

Run (from the worktree so relative baseline paths resolve):
  python <.../code/mz_report.py> <TS> [seeds_csv]
  e.g. mz_report.py 2026-08-15_085544_loo 42,123,2026
"""
from __future__ import annotations

import sys

import numpy as np

from dm_report import DUMP_DIR, _ensemble

try:
    from scipy.stats import chi2

    def _chi2_sf(w: float) -> float:
        return float(chi2.sf(w, 2))
except ImportError:  # pragma: no cover - scipy present in this env
    import math

    def _chi2_sf(w: float) -> float:
        return math.exp(-w / 2.0)


def mz(ts: str, horizon: int, rung: str, seeds) -> dict[str, float]:
    """OLS of realized y on forecast x plus the MZ joint Wald test (H0: a=0, b=1)."""
    _keys, y, x = _ensemble(ts, horizon, rung, seeds)
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    theta = xtx_inv @ design.T @ y            # theta = (a, b)
    resid = y - design @ theta
    rss = float(resid @ resid)
    sigma2 = rss / (n - 2)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    cov = sigma2 * xtx_inv
    diff = theta - np.array([0.0, 1.0])
    wald = float(diff @ np.linalg.inv(cov) @ diff)
    return {"a": float(theta[0]), "b": float(theta[1]), "r2": r2,
            "wald": wald, "p": _chi2_sf(wald), "n": n}


def main(ts: str, seeds, horizons) -> None:
    rungs = ["HAR", "FULL", "minus_graph", "minus_gate", "minus_news", "LSTM_only"]
    for horizon in horizons:
        print(f"# Mincer-Zarnowitz h{horizon} seeds={list(seeds)} "
              f"-- y = a + b*x; H0: a=0, b=1\n")
        print("| Rung | a | b | R^2 | Wald | p | n |")
        print("|---|---|---|---|---|---|---|")
        for rung in rungs:
            try:
                r = mz(ts, horizon, rung, seeds)
                print(f"| {rung} | {r['a']:+.6f} | {r['b']:.4f} | {r['r2']:.4f} "
                      f"| {r['wald']:.3f} | {r['p']:.4g} | {r['n']} |")
            except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - I/O guard
                print(f"| {rung} | n/a | n/a | n/a | n/a | {exc} | n/a |")
        print()


if __name__ == "__main__":  # pragma: no cover
    _ts = sys.argv[1]
    _seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42]
    _horizons = [1, 5, 10, 22]
    _ = DUMP_DIR  # rung->dump-dir map lives in dm_report; imported for reuse
    main(_ts, _seeds, _horizons)
