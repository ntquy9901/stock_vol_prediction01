"""Does adding the leverage feature (downside realized semivariance) to own-history help? (QLIKE + DM)

Follow-up to the principled-HAR finding: `semi_neg` is the one published feature that carried OOS signal.
Here we AUGMENT the strong own-history set instead of replacing it:
  own            = FM.OWN
  own+semi_neg   = FM.OWN + [semi_neg]                 (add leverage / bad-volatility, Patton-Sheppard 2015)
  own+semi       = FM.OWN + [semi_neg, semi_pos]       (add both signed semivariances; semi_pos expected noise)
Each variant is compared to own under the same walk-forward / pooled-QLIKE / date-clustered-DM protocol.

Run: ``python augment_leverage.py [hose|sp500]``.  Output: results/gamma_gbm/augment_leverage_<market>.json
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
import build_panel as BP  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
BASELINE = "own"
OWN = config.own_set(FM.OWN)   # single source: baselines... config.OWN_DROP/OWN_ADD


def _sets():
    """Feature sets: baseline own (no rq), own+leverage, own+both signed semivariances."""
    return {"own": OWN, "own+semi_neg": OWN + ["semi_neg"],
            "own+semi": OWN + ["semi_neg", "semi_pos"]}


def _pooled(a, cols, h, min_rows):
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    preds, yy, dts, last_tr = [], [], [], None
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < min_rows:
            continue
        preds.append(np.mean([FM.gbm(tr, te, cols, s) for s in FM.SEEDS], 0))
        yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy()); last_tr = tr
    if not yy:
        return None
    return np.concatenate(yy), np.concatenate(preds), np.concatenate(dts), last_tr


def run(market, load_fn=None):
    """own vs own+semi_neg vs own+semi; returns per-horizon QLIKE + DM(variant vs own) + fit diagnostics."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    frames = BP.feature_frames(frames)
    sets = _sets()
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        res = {m: _pooled(a, cols, h, min_rows) for m, cols in sets.items()}
        if any(res[m] is None for m in sets):
            continue
        y, _, dates, last_tr = res[BASELINE]
        err = {m: M.per_obs_qlike(y, res[m][1], floor=FL) for m in sets}
        q = {m: float(np.mean(err[m])) for m in sets}
        comps = {}
        for m in sets:
            if m == BASELINE:
                continue
            p = float(ST.date_clustered_dm(err[m], err[BASELINE], dates, h)["p_value"])
            comps[m] = {"gain_vs_own_pct": (q[BASELINE] - q[m]) / q[BASELINE] * 100.0, "dm_p": p}
        tr_q = {m: float(np.mean(M.per_obs_qlike(last_tr["y"].to_numpy(float),
                np.mean([FM.gbm(last_tr, last_tr, cols, s) for s in FM.SEEDS], 0), floor=FL)))
                for m, cols in sets.items()}
        out[f"h{h}"] = {"n": int(len(y)), "qlike": q, "vs_own": comps,
                        "fit_diagnostics": {m: {"verdict": "overfit" if q[m] > tr_q[m] * 1.25 else "ok",
                                                "train_qlike": tr_q[m], "test_qlike": q[m]} for m in sets}}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        print(f"\n{market} {h} (n={r['n']:,}): own QLIKE={r['qlike']['own']:.4f}", flush=True)
        for m, c in r["vs_own"].items():
            v = "HELPS" if (c["gain_vs_own_pct"] > 0 and c["dm_p"] < 0.05) else \
                ("helps(ns)" if c["gain_vs_own_pct"] > 0 else "hurts")
            print(f"    {m:14s} QLIKE={r['qlike'][m]:.4f} gain={c['gain_vs_own_pct']:+.3f}% "
                  f"DM p={c['dm_p']:.3g} -> {v}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"augment_leverage_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
