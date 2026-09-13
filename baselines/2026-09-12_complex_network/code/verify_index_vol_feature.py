"""Does the CAUSAL market-index volatility help the per-stock gamma-GBM? (QLIKE + Diebold-Mariano)

Experiment A's target was the FUTURE index volatility (forward-looking -> NOT usable as a feature). Here we
build a CAUSAL index-volatility feature -- the trailing ``IDX_RV_WINDOW``-day realized volatility of the market
index (std of its daily log-returns up to and including day t, known at t) -- broadcast onto every stock by
date, and test whether adding it to the own-history GBM improves the per-stock forecast. The index RETURN is
deliberately NOT used (a directional market feature, out of scope per request).

Same protocol as run_gbm: expanding walk-forward over S1.FOLDS with an L-day embargo, pooled per-observation
QLIKE, date-clustered Diebold-Mariano, and train/test QLIKE for a fit-diagnostics verdict.

Run: ``python verify_index_vol_feature.py [hose|sp500]``.
Output: results/gamma_gbm/complex_network_<market>_idxvol_feature.json
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
import market_index  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
FEAT = "idx_rv"


def idx_rv_series(idx):
    """Causal trailing realized vol of the index: rolling std (ddof=1) of daily log-returns over the past
    ``config.IDX_RV_WINDOW`` days, indexed by date. Uses only past/current returns (no look-ahead)."""
    s = (idx.set_index("date")["lr"]
         .rolling(config.IDX_RV_WINDOW, min_periods=config.IDX_RV_MIN_PERIODS).std())
    return s.rename(FEAT)


def merge_idx_rv(frames, idx):
    """Broadcast the causal index realized-vol onto every ticker frame by date (market-level feature)."""
    rv = idx_rv_series(idx)
    out = {}
    for tk, d in frames.items():
        e = d.copy()
        e[FEAT] = e["date"].map(rv)              # value at row-date t = trailing index vol known at t (causal)
        out[tk] = e
    return out


def _fold_qlike(tr, cols):
    """In-sample (train) pooled QLIKE of the seed-averaged GBM on ``cols`` (for the fit-diagnostics verdict)."""
    ptr = np.mean([FM.gbm(tr, tr, cols, s) for s in FM.SEEDS], 0)
    return float(np.mean(M.per_obs_qlike(tr["y"].to_numpy(float), ptr, floor=FL)))


def run_idxvol(market, load_fn=None, index_fn=None):
    """GBM(own) vs GBM(own+index-vol) for a market; returns the JSON-serialisable result dict."""
    load_fn = load_fn or FM.load
    index_fn = index_fn or market_index.load_index
    min_rows = 30000 if market == "sp500" else 3000
    frames, _, _ = load_fn(market)
    frames = merge_idx_rv(frames, index_fn(market))
    models = {"GBM": FM.OWN, "GBM+idxvol": FM.OWN + [FEAT]}
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
            yy.append(te["y"].to_numpy(float)); dts.append(te["date"].to_numpy())
            last_tr = tr
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        e = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        q = {m: float(np.mean(e[m])) for m in models}
        p = float(ST.date_clustered_dm(e["GBM+idxvol"], e["GBM"], dates, h)["p_value"])
        train_q = {m: _fold_qlike(last_tr, cols) for m, cols in models.items()}
        diags = {m: {"verdict": "overfit" if q[m] > train_q[m] * 1.25 else "ok",
                     "train_qlike": train_q[m], "test_qlike": q[m]} for m in models}
        out[f"h{h}"] = {"n": int(len(y)), "GBM": q["GBM"], "GBM+idxvol": q["GBM+idxvol"],
                        "gain_pct": (q["GBM"] - q["GBM+idxvol"]) / q["GBM"] * 100, "dm_p": p,
                        "train_metrics": train_q, "test_metrics": q, "fit_diagnostics": diags}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        verdict = "HELPS" if r["gain_pct"] > 0 else "hurts"
        print(f"{market} {h} (n={r['n']:,}): GBM {r['GBM']:.4f} | +idxvol {r['GBM+idxvol']:.4f}  "
              f"({r['gain_pct']:+.2f}%, DM p={r['dm_p']:.3g}) -> idx-vol {verdict}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run_idxvol(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}_idxvol_feature.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
