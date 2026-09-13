"""Does the CAUSAL Diebold-Yilmaz volatility-spillover feature help the per-stock gamma-GBM? (QLIKE + DM).

GBM(own) vs GBM(own + [net_spillover, total_spillover]). Same protocol as
``baselines/2026-09-12_complex_network/code/verify_index_vol_feature.py``: expanding walk-forward over
``S1.FOLDS`` with an ``int(h*1.6)+5``-day embargo, seed-averaged predictions, pooled per-observation QLIKE,
date-clustered Diebold-Mariano, and train/test QLIKE for a fit-diagnostics verdict.

Run: ``python run_dy.py [hose|sp500]``.  Output: results/gamma_gbm/dy_spillover_<market>.json
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
import dy_spillover as DS  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
FEATS = [config.FEAT_NET, config.FEAT_TOTAL]


def _fold_qlike(tr, cols):
    """In-sample (train) pooled QLIKE of the seed-averaged GBM on ``cols`` (for the fit-diagnostics verdict)."""
    ptr = np.mean([FM.gbm(tr, tr, cols, s) for s in FM.SEEDS], 0)
    return float(np.mean(M.per_obs_qlike(tr["y"].to_numpy(float), ptr, floor=FL)))


def _success(out):
    """Economic go/no-go on top of DM: spillover wins only if every horizon has a positive, sign-consistent
    QLIKE gain >= config.SUCCESS_MIN_GAIN_PCT, DM p < config.SUCCESS_DM_P, and no overfit verdict."""
    hs = [k for k in out if k.startswith("h")]
    if not hs:
        return False
    gains_ok = all(out[k]["gain_pct"] >= config.SUCCESS_MIN_GAIN_PCT for k in hs)
    dm_ok = all(out[k]["dm_p"] < config.SUCCESS_DM_P for k in hs)
    fit_ok = all(out[k]["fit_diagnostics"]["GBM+spillover"]["verdict"] == "ok" for k in hs)
    return bool(gains_ok and dm_ok and fit_ok)


def run_dy(market, load_fn=None):
    """GBM(own) vs GBM(own+spillover) for a market; returns the JSON-serialisable result dict."""
    load_fn = load_fn or FM.load
    min_rows = config.GBM_MIN_TRAIN_ROWS.get(market, config.GBM_MIN_TRAIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    frames, diag = DS.build_spillover(frames)
    if not any(f[config.FEAT_TOTAL].notna().any() for f in frames.values()):
        raise ValueError("total_spillover is entirely NaN across the panel -- spillover feature is degenerate "
                         "(check DY_SECTOR_MIN_STOCKS / DY_WINDOW vs the market's sector count and history)")
    models = {"GBM": FM.OWN, "GBM+spillover": FM.OWN + FEATS}
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        preds = {m: [] for m in models}
        yy, dts, last_tr = [], [], None
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            for m, cols in models.items():
                preds[m].append(np.mean([FM.gbm(tr, te, cols, s) for s in FM.SEEDS], 0))
            yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy()); last_tr = tr
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        q = {m: float(np.mean(e[m])) for m in models}
        p = float(ST.date_clustered_dm(e["GBM+spillover"], e["GBM"], dates, h)["p_value"])
        train_q = {m: _fold_qlike(last_tr, cols) for m, cols in models.items()}
        diags = {m: {"verdict": "overfit" if q[m] > train_q[m] * 1.25 else "ok",
                     "train_qlike": train_q[m], "test_qlike": q[m]} for m in models}
        out[f"h{h}"] = {"n": int(len(y)), "GBM": q["GBM"], "GBM+spillover": q["GBM+spillover"],
                        "gain_pct": (q["GBM"] - q["GBM+spillover"]) / q["GBM"] * 100, "dm_p": p,
                        "train_metrics": train_q, "test_metrics": q, "fit_diagnostics": diags}
    out["diag"] = diag
    out["success"] = _success(out)
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    d = out["diag"]
    print(f"{market}: {d['n_sectors']} sector series, {d['n_anchor_windows']} VAR windows "
          f"({d['n_failed_windows']} failed)", flush=True)
    for h, r in out.items():
        if not h.startswith("h"):
            continue
        verdict = "HELPS" if r["gain_pct"] > 0 else "hurts"
        print(f"{market} {h} (n={r['n']:,}): GBM {r['GBM']:.4f} | +spillover {r['GBM+spillover']:.4f}  "
              f"({r['gain_pct']:+.2f}%, DM p={r['dm_p']:.3g}) -> spillover {verdict}", flush=True)
    print(f"{market}: success={out['success']}  "
          f"(criterion: every-h gain>={config.SUCCESS_MIN_GAIN_PCT}% & DM p<{config.SUCCESS_DM_P} & no overfit; "
          f"warn: at n~400k a ~0.04% 'DM-significant' gain is NOT a real win)", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run_dy(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"dy_spillover_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
