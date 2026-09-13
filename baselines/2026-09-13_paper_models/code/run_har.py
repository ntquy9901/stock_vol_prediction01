"""Principled HAR-family feature set vs own-history GBM: pooled QLIKE + date-clustered DM + leave-one-out.

Fixed Corsi windows (no data-driven selection). GBM(principled) uses the 8 cited features
(:data:`build_panel.FEATURES`); GBM(own) uses :data:`FM.OWN`. Same walk-forward / pooled-QLIKE / DM protocol
as the sibling baselines. Success = principled is NON-INFERIOR to own (not significantly worse).

Run: ``python run_har.py [hose|sp500]``.  Output: results/gamma_gbm/principled_har_<market>.json
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
FEATURES = BP.FEATURES
OWN = config.own_set(FM.OWN)   # single source: baselines... config.OWN_DROP/OWN_ADD


def _fold_qlike(tr, cols):
    """In-sample (train) pooled QLIKE of the seed-averaged GBM on ``cols`` (fit-diagnostics verdict)."""
    ptr = np.mean([FM.gbm(tr, tr, cols, s) for s in FM.SEEDS], 0)
    return float(np.mean(M.per_obs_qlike(tr["y"].to_numpy(float), ptr, floor=FL)))


def _pooled(a, cols, h, min_rows):
    """Walk-forward seed-averaged predictions for ``cols``; returns (y, preds, dates, last_train) or None."""
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
    """Principled-vs-own DM + leave-one-out for a market; returns the JSON-serialisable result dict."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    framed = BP.feature_frames(frames)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(framed, {}, h)
        models = {"principled": FEATURES, "own": OWN}
        res = {m: _pooled(a, cols, h, min_rows) for m, cols in models.items()}
        if res["principled"] is None or res["own"] is None:
            continue
        y, p_pr, dates, last_tr = res["principled"]
        _, p_own, _, _ = res["own"]
        err = {"principled": M.per_obs_qlike(y, p_pr, floor=FL), "own": M.per_obs_qlike(y, p_own, floor=FL)}
        q = {m: float(np.mean(err[m])) for m in err}
        dm = float(ST.date_clustered_dm(err["principled"], err["own"], dates, h)["p_value"])
        loo = {}
        for f in FEATURES:
            r = _pooled(a, [c for c in FEATURES if c != f], h, min_rows)
            if r is None:  # pragma: no cover - unreachable when principled scored (fold gate is feature-independent)
                continue
            el = M.per_obs_qlike(r[0], r[1], floor=FL)
            loo[f] = {"gain_vs_full_pct": (q["principled"] - float(np.mean(el))) / q["principled"] * 100.0,
                      "dm_p": float(ST.date_clustered_dm(el, err["principled"], r[2], h)["p_value"])}
        tr_q = {m: _fold_qlike(last_tr, cols) for m, cols in models.items()}
        out[f"h{h}"] = {"n": int(len(y)), "qlike": q,
                        "vs_own": {"gain_vs_own_pct": (q["own"] - q["principled"]) / q["own"] * 100.0, "dm_p": dm},
                        "leave_one_out": loo, "train_metrics": tr_q,
                        "fit_diagnostics": {m: {"verdict": "overfit" if q[m] > tr_q[m] * 1.25 else "ok",
                                                "train_qlike": tr_q[m], "test_qlike": q[m]} for m in models}}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        c = r["vs_own"]
        verdict = "non-inferior" if (c["gain_vs_own_pct"] >= 0 or c["dm_p"] > 0.05) else "WORSE"
        print(f"{market} {h} (n={r['n']:,}): principled QLIKE={r['qlike']['principled']:.4f} "
              f"own={r['qlike']['own']:.4f} gain={c['gain_vs_own_pct']:+.2f}% DM p={c['dm_p']:.3g} -> {verdict}",
              flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"principled_har_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
