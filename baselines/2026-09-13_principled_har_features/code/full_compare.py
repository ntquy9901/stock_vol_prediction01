"""Full model comparison: ALL models (incl the principled feature set) x 5 metrics + FULL pairwise DM matrix.

Extends the paper-metrics harness to add the principled HAR-family GBM as a model and to compute the complete
Diebold-Mariano matrix (every model pair) from the per-observation QLIKE errors, on one shared walk-forward
panel. Reuses the exact per-fold graph-feature construction + helpers from ``paper_metrics_sp500`` (PM) so the
non-principled models reproduce the paper table. Output: results/gamma_gbm/full_compare_<market>.json.

Run: ``python full_compare.py [hose|sp500]``.
"""
import itertools
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
import paper_metrics_sp500 as PM  # noqa: E402  (reuse nb_col / adj_sector / all_metrics verbatim)

FL = FM.FL


def dm_matrix(err, dates, h):
    """Full pairwise date-clustered DM p-value matrix over the per-obs error dict ``err`` (all model pairs)."""
    out = {}
    for a, b in itertools.combinations(err, 2):
        out[f"{a}_vs_{b}"] = float(ST.date_clustered_dm(err[a], err[b], dates, h)["p_value"])
    return out


def run(market, load_fn=None):
    """Full comparison for a market; returns {h: {metrics, dm_qlike (full matrix), n}}."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, sect, edates = load_fn(market)
    if market != "sp500":                                             # inject REAL crawled VN announcement dates
        ep = REPO / "results" / "gamma_gbm" / "hose_earnings_combined.parquet"
        if ep.exists():
            e = pd.read_parquet(ep)
            edates = {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    has_earn = bool(edates)
    frames = BP.feature_frames(frames)                                # add semi_neg/semi_pos per ticker (causal)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        n = len(tickers)
        Wu, Ws = FM._uniform(n), PM.adj_sector(tickers, sect)
        models = ["HAR", "HARQ", "GBM", "principled", "GBM+market", "GBM+corr", "GBM+sector", "GBM+plac",
                  "GBM+oracle"]
        if has_earn:
            models = models[:8] + ["GBM+earn", "GBM+earn+corr"] + models[8:]
        preds = {m: [] for m in models}
        yy, dts = [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            rng = np.random.default_rng(11 + k)
            Wp = np.zeros((n, n))
            for i in range(n):
                j = rng.choice(np.delete(np.arange(n), i), size=min(S1.TOPK, n - 1), replace=False)
                Wp[i, j] = 1.0 / len(j)
            fold = a[(a.date >= S1.TRAIN_START) & (a.date < tend)].copy()
            fold["g_market"] = FM.nb(fold, tickers, Wu); fold["g_corr"] = FM.nb(fold, tickers, Wc)
            fold["g_sector"] = FM.nb(fold, tickers, Ws); fold["g_plac"] = FM.nb(fold, tickers, Wp)
            fold["g_oracle"] = PM.nb_col(fold, tickers, Wc, "y")
            for c in ("g_market", "g_corr", "g_sector", "g_plac", "g_oracle"):
                fold[c] = fold[c].fillna(0.0)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
            tef = fold[(fold.date >= ts) & (fold.date < tend)]
            cols = {"GBM": FM.OWN, "principled": BP.FEATURES, "GBM+market": FM.OWN + ["g_market"],
                    "GBM+corr": FM.OWN + ["g_corr"], "GBM+sector": FM.OWN + ["g_sector"],
                    "GBM+plac": FM.OWN + ["g_plac"], "GBM+oracle": FM.OWN + ["g_oracle"]}
            if has_earn:
                cols["GBM+earn"] = FM.OWN + FM.EARN; cols["GBM+earn+corr"] = FM.OWN + FM.EARN + ["g_corr"]
            preds["HAR"].append(FM._har_ols(trf, tef)); preds["HARQ"].append(FM._harq_ols(trf, tef))
            for m, cc in cols.items():
                preds[m].append(np.mean([FM.gbm(trf, tef, cc, s) for s in FM.SEEDS], 0))
            yy.append(tef["y"].to_numpy(float)); dts.append(tef["date"].to_numpy())
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        err = {m: M.per_obs_qlike(y, np.concatenate(preds[m]), floor=FL) for m in models}
        met = {m: PM.all_metrics(y, np.concatenate(preds[m])) for m in models}
        out[f"h{h}"] = {"n": int(len(y)), "metrics": met, "dm_qlike_matrix": dm_matrix(err, dates, h)}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        print(f"\n=== {market} {h} (n={r['n']:,}) ===", flush=True)
        for m, mm in sorted(r["metrics"].items(), key=lambda kv: kv[1]["qlike"]):
            print(f"  {m:16s} QLIKE {mm['qlike']:.4f} R2 {mm['r2']:+.3f}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"full_compare_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
