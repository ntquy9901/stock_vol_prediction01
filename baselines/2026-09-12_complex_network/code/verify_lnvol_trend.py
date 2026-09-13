"""Verification: is the Experiment-A ``idx_lnvol`` OOS R^2 a secular-volume-trend artifact, not topology signal?

Market volume has a strong multi-year upward trend, so a model can score a positive OOS R^2 on the future
mean log-volume target just by tracking time, with no genuine topology->volume mechanism. This check compares,
under the EXACT same causal expanding walk-forward as :mod:`run_index`, three feature sets per target:

* ``topo``       -- the topology metrics (RandomForest) = the headline Experiment-A model.
* ``time``       -- a single causal feature (the window's ordinal position) fit with LinearRegression, which
                    can extrapolate a linear trend = the pure time-trend forecast.
* ``topo+time``  -- topology metrics plus the time ordinal (RandomForest).

Read on the two targets together: if ``time`` alone matches ``topo`` on ``idx_lnvol`` (the trending target)
while BOTH are ~0 on ``idx_vol`` (the trend-free volatility target the paper actually claims), the high
``idx_lnvol`` R^2 is a trend artifact, not evidence that topology predicts volume.

Run: ``python verify_lnvol_trend.py [hose|sp500]``.  Output: results/gamma_gbm/complex_network_<market>_lnvol_trend.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import config  # noqa: E402
import run_index  # noqa: E402  (reuse build_sample / walk_forward / _oos + its import chain)
import market_index  # noqa: E402
import full_matrix as FM  # noqa: E402

TARGETS = ["idx_lnvol", "idx_vol"]


def add_time(S):
    """Add the causal ``time`` feature = the window's ordinal position (``S`` is already sorted by ``d0`` in
    :func:`run_index.build_sample`, so the row index IS the causal order; no future information)."""
    S = S.copy()
    S["time"] = np.arange(len(S), dtype=float)
    return S


def _r2(S, feat_cols, target, ctor):
    """OOS R^2 of ``ctor`` on ``target`` over ``feat_cols`` under run_index's causal walk-forward."""
    ys, yhats, _, _, _ = run_index.walk_forward(S, feat_cols, target, ctor)
    return run_index._oos(ys, yhats)[0]


def verify_sample(S):
    """Per-target OOS R^2 for the topo / time / topo+time feature sets on one assembled sample ``S``."""
    rf = lambda: RandomForestRegressor(**config.RF_KW)  # noqa: E731
    out = {}
    for tgt in TARGETS:
        out[tgt] = {"topo_rf": _r2(S, config.TOPO, tgt, rf),
                    "time_lr": _r2(S, ["time"], tgt, LinearRegression),
                    "topo_time_rf": _r2(S, config.TOPO + ["time"], tgt, rf)}
    return out


def run_verify(market, load_fn=None, index_fn=None):
    """Assemble the Experiment-A sample for ``market`` and return the trend-verification R^2 table."""
    load_fn = load_fn or FM.load
    index_fn = index_fn or market_index.load_index
    frames, _, _ = load_fn(market)
    idx = index_fn(market)
    S, _ = run_index.build_sample(frames, market, idx, config.ALPHA, config.WIN)
    S = add_time(S)
    return {"market": market, "n_windows": int(len(S)), "targets": verify_sample(S)}


def _print(res):  # pragma: no cover - console formatting only
    print(f"\n=== lnvol-trend verification: {res['market']} (n_windows={res['n_windows']}) ===", flush=True)
    print(f"{'target':10s} {'topo_rf':>9s} {'time_lr':>9s} {'topo+time':>10s}", flush=True)
    for tgt, r in res["targets"].items():
        print(f"{tgt:10s} {r['topo_rf']:9.4f} {r['time_lr']:9.4f} {r['topo_time_rf']:10.4f}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "sp500"
    res = run_verify(market)
    _print(res)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}_lnvol_trend.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
