"""DM-verify the audit's proposed feature drops: does a REDUCED own-history set match the full own-history GBM?

The feature audit (screen + VIF) recommends dropping ``rq`` (collinear with ``har_weekly``) everywhere, and the
mean-reversion block (``mr_*``) on SP500. Screen stats are a filter, not a verdict, so this check runs the
reduced sets head-to-head against the full own-history GBM under the same walk-forward, with pooled QLIKE and a
date-clustered Diebold-Mariano test. A reduced set is a SAFE drop iff it does not lose significantly to ``own``.

Feature sets (all subsets of ``FM.OWN``):
- ``own``          -- the 9 own-history features (baseline reference).
- ``own_minus_rq`` -- drop the collinear realized-quarticity term (8 features).
- ``har3``         -- the HAR trio only (drops rq AND the mr_* block).

Run: ``python verify_reduced_features.py [hose|sp500]``.  Output: results/gamma_gbm/complex_network_<market>_reduced.json
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
import config  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
BASELINE = "own"
FEATURE_SETS = {"own": FM.OWN,
                "own_minus_rq": [f for f in FM.OWN if f != "rq"],
                "har3": FM.HAR}


def _fold_qlike(tr, cols):
    """In-sample (train) pooled QLIKE of the seed-averaged GBM on ``cols`` (for the fit-diagnostics verdict)."""
    ptr = np.mean([FM.gbm(tr, tr, cols, s) for s in FM.SEEDS], 0)
    return float(np.mean(M.per_obs_qlike(tr["y"].to_numpy(float), ptr, floor=FL)))


def run_reduced(market, load_fn=None):
    """Walk-forward pooled QLIKE for each feature set + DM of each reduced set vs the full own-history set."""
    load_fn = load_fn or FM.load
    min_rows = 30000 if market == "sp500" else 3000
    frames, _, _ = load_fn(market)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        preds = {m: [] for m in FEATURE_SETS}
        yy, dts, last_tr = [], [], None
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            for m, cols in FEATURE_SETS.items():
                preds[m].append(np.mean([FM.gbm(tr, te, cols, s) for s in FM.SEEDS], 0))
            yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy())
            last_tr = tr
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        err = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in FEATURE_SETS}
        q = {m: float(np.mean(err[m])) for m in FEATURE_SETS}
        comps = {}
        for m in FEATURE_SETS:
            if m == BASELINE:
                continue
            p = float(ST.date_clustered_dm(err[m], err[BASELINE], dates, h)["p_value"])
            comps[m] = {"gain_vs_own_pct": (q[BASELINE] - q[m]) / q[BASELINE] * 100.0, "dm_p": p}
        train_q = {m: _fold_qlike(last_tr, cols) for m, cols in FEATURE_SETS.items()}
        diags = {m: {"verdict": "overfit" if q[m] > train_q[m] * 1.25 else "ok",
                     "train_qlike": train_q[m], "test_qlike": q[m]} for m in FEATURE_SETS}
        out[f"h{h}"] = {"n": int(len(y)), "qlike": q, "vs_own": comps,
                        "train_metrics": train_q, "fit_diagnostics": diags}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        base = r["qlike"][BASELINE]
        print(f"\n{market} {h} (n={r['n']:,}): own QLIKE={base:.4f}", flush=True)
        for m, c in r["vs_own"].items():
            safe = "SAFE drop" if (c["gain_vs_own_pct"] >= 0 or c["dm_p"] > 0.05) else "HURTS (keep)"
            print(f"    {m:14s} QLIKE={r['qlike'][m]:.4f}  gain vs own={c['gain_vs_own_pct']:+.2f}%  "
                  f"DM p={c['dm_p']:.3g}  -> {safe}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run_reduced(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}_reduced.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
