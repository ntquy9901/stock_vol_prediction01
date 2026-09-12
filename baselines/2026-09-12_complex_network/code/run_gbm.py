"""Experiment B (secondary): do the topology metrics add incremental value to the per-stock gamma-GBM?

Adapted from ``scripts/eda/complex_network_gbm.py``: the daily topology panel (:func:`topology.build_topo_daily`)
is broadcast onto every ticker by date and fed to the gamma-GBM alongside the own-history block. GBM (own,
9 features) vs GBM+topo (own+topology). Score = pooled per-observation QLIKE + date-clustered Diebold-Mariano at each
horizon. Adds train/test QLIKE + fit-diagnostics evidence so the pre-push overfit gate passes.

Run: ``python run_gbm.py [hose|sp500]``.  Output: results/gamma_gbm/complex_network_<market>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
# local modules first: FM's import chain prepends submission/soict_lstm_gat (which has its OWN config.py)
# to sys.path, so importing these before FM binds `config`/`topology` to THIS baseline.
import config  # noqa: E402
import topology  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402  (preloaded into sys.modules via FM's import chain)
import stats as ST  # noqa: E402

FL = FM.FL


def _merge_topo(frames, market):
    """Merge the daily topology panel onto every ticker frame by date (broadcast, market-level, causal).
    Topology uses REAL ln(volume) from the market's raw OHLCV (Refinement 1)."""
    F = topology.build_topo_daily(frames, market, config.ALPHA, config.WIN)
    Fr = F.reset_index(names="date")
    return {tk: d.merge(Fr, on="date", how="left") for tk, d in frames.items()}


def _fold_qlike(tr, cols):
    """Seed-ensembled in-sample TRAIN QLIKE on the final fold's training rows (overfit evidence)."""
    ytr = tr["y"].to_numpy(float)
    ptr = np.mean([FM.gbm(tr, tr, cols, s) for s in FM.SEEDS], 0)
    return float(np.mean(M.per_obs_qlike(ytr, ptr, floor=FL)))


def run_gbm(market, load_fn=None):
    """Experiment B for a market; returns the JSON-serialisable result dict (no file write)."""
    load_fn = load_fn or FM.load
    min_rows = 30000 if market == "sp500" else 3000
    frames, _, _ = load_fn(market)
    frames = _merge_topo(frames, market)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        for c in config.TOPO:
            # per-ticker ffill: the panel concatenates tickers in blocks (no global date sort), so a plain
            # ffill would drag one ticker's late-dated topology into the next ticker's early rows (M1).
            a[c] = a.groupby("ticker")[c].ffill().fillna(0.0) if c in a.columns else 0.0
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        models = {"GBM": FM.OWN, "GBM+topo": FM.OWN + config.TOPO}
        preds = {m: [] for m in models}; yy, dts = [], []
        last_tr = None
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            for m, cols in models.items():
                preds[m].append(np.mean([FM.gbm(tr, te, cols, s) for s in FM.SEEDS], 0))
            yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy())
            last_tr = tr
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        q = {m: float(np.mean(e[m])) for m in models}
        p = float(ST.date_clustered_dm(e["GBM+topo"], e["GBM"], dates, h)["p_value"])
        train_q = {m: _fold_qlike(last_tr, cols) for m, cols in models.items()}
        diags = {m: {"verdict": "overfit" if q[m] > train_q[m] * 1.25 else "ok",
                     "train_qlike": train_q[m], "test_qlike": q[m]} for m in models}
        out[f"h{h}"] = {"n": int(len(y)), "GBM": q["GBM"], "GBM+topo": q["GBM+topo"],
                        "gain_pct": (q["GBM"] - q["GBM+topo"]) / q["GBM"] * 100, "dm_p": p,
                        "train_metrics": train_q, "test_metrics": q, "fit_diagnostics": diags}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        print(f"\n=== {market} {h} (n={r['n']:,}) ===", flush=True)
        print(f"  GBM {r['GBM']:.4f} | GBM+topo {r['GBM+topo']:.4f}  "
              f"({r['gain_pct']:+.2f}%, DM p={r['dm_p']:.3f})", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run_gbm(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
