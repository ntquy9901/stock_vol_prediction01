"""Per-stock GARCH / GJR-GARCH conditional-variance benchmark on the shared walk-forward folds.

GARCH uses ONLY each stock's own daily log-return history (no exogenous features). Scored against the
HAR baseline (``full_matrix._har_ols``) on the IDENTICAL pooled rows: pooled per-observation QLIKE
(floor ``FM.FL``) + date-clustered Diebold-Mariano, all horizons. Mirrors the walk-forward / fold-gate /
fit-diagnostics / JSON layout of the sibling drivers ``run_gbm.py`` and ``run_har.py``.

Run: ``python run_garch.py [hose|sp500]``.  Output: results/gamma_gbm/garch_<market>.json
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
import garch_model as GM  # noqa: E402

FL = FM.FL
VARIANTS = ("garch", "gjr")
LABEL = {"garch": "GARCH", "gjr": "GJR-GARCH"}


def _int_dates(values):
    """datetime64 array -> int64 ns-since-epoch (stable key for date<->index mapping)."""
    return np.asarray(values, dtype="datetime64[ns]").astype("int64")


def _ticker_series(frames):
    """Per-ticker (sorted int-date array, return array, {date_int: index}) from the enriched frames.

    Drops rows with a missing ``daily_return`` (the first row of each series) so the conditional-variance
    recursion sees a contiguous, NaN-free return history."""
    series = {}
    for tk, d in frames.items():
        s = d[["date", "daily_return"]].dropna(subset=["daily_return"]).sort_values("date")
        di = _int_dates(s["date"].to_numpy())
        series[tk] = (di, s["daily_return"].to_numpy(float), {int(v): i for i, v in enumerate(di)})
    return series


def _ticker_task(payload):
    """Forecast every requested (horizon, fold) window for one ticker, both variants (thread-safe:
    reads only its own arrays + read-only module state).

    ``payload = (returns, jobs)`` where each job is
    ``(h, k, n_train, test_series_idx, test_rows, train_series_idx | None, train_rows | None)``.
    Returns a list of ``(h, k, test_rows, g_test, j_test, g_ok, j_ok, train_rows, g_train, j_train)``;
    the fit is done ONCE per (variant) per job and reused for the test and (optional) train windows."""
    returns, jobs = payload
    out = []
    for (h, k, n_train, ti, trows, tri, trrows) in jobs:
        preds = {}
        oks = {}
        for v in VARIANTS:
            p = GM.fit_params(returns[:n_train], v)
            oks[v] = p.ok
            preds[(v, "test")] = GM.forecast(returns, p, v, ti, h)
            preds[(v, "train")] = (GM.forecast(returns, p, v, tri, h) if tri is not None else None)
        out.append((h, k, trows, preds[("garch", "test")], preds[("gjr", "test")],
                    oks["garch"], oks["gjr"], trrows,
                    preds[("garch", "train")], preds[("gjr", "train")]))
    return out


def _active_folds(a, embargo, min_rows):
    """Folds whose pooled train/test rows pass the sibling gate; returns [(k, ts, tend, tr, te)]."""
    folds = []
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)].reset_index(drop=True)
        if len(te) == 0 or len(tr) < min_rows:
            continue
        folds.append((k, ts, tend, tr.reset_index(drop=True), te))
    return folds


def _idx(pos, dates):
    """Map a date64 array to the ticker's series indices via its {date_int: index} table."""
    return np.array([pos[int(x)] for x in _int_dates(dates)], dtype=int)


def _estimable(series, tickers, cutoff, min_obs):
    """Tickers with >= ``min_obs`` own-history returns strictly before ``cutoff`` (GARCH is estimable)."""
    return {tk for tk in tickers
            if int(np.searchsorted(series[tk][0], cutoff, side="left")) >= min_obs}


def _build_jobs(series, folds, embargo, h, last_k, min_obs):
    """Per-ticker job lists over the ESTIMABLE ticker-folds only (test windows already filtered upstream);
    on the last fold also the train-window rows of estimable train tickers (for the in-sample diagnostic),
    even those absent from that fold's test. Returns the per-fold test row counts and the estimable
    last-fold train subframe (row order == the g_train scatter order)."""
    jobs = {tk: [] for tk in series}
    te_sizes = {k: len(te) for (k, _, _, _, te) in folds}
    ts_last, tr_last = next((ts, tr) for (k, ts, tend, tr, te) in folds if k == last_k)   # folds non-empty
    est_tr = _estimable(series, tr_last["ticker"].unique(), int((ts_last - embargo).value), min_obs)
    tr_scored = tr_last[tr_last["ticker"].isin(est_tr)].reset_index(drop=True)
    tr_g = {tk: sub for tk, sub in tr_scored.groupby("ticker")}
    for (k, ts, tend, tr, te) in folds:
        cutoff = int((ts - embargo).value)
        te_g = {tk: sub for tk, sub in te.groupby("ticker")}
        tickers = set(te_g) | (set(tr_g) if k == last_k else set())
        for tk in tickers:
            di, _, pos = series[tk]
            n_train = int(np.searchsorted(di, cutoff, side="left"))
            sub = te_g.get(tk)
            ti = _idx(pos, sub["date"].to_numpy()) if sub is not None else np.empty(0, dtype=int)
            trows = sub.index.to_numpy() if sub is not None else np.empty(0, dtype=int)
            tri = trrows = None
            if k == last_k:
                subtr = tr_g.get(tk)
                tri = _idx(pos, subtr["date"].to_numpy()) if subtr is not None else np.empty(0, dtype=int)
                trrows = subtr.index.to_numpy() if subtr is not None else np.empty(0, dtype=int)
            jobs[tk].append((h, k, n_train, ti, trows, tri, trrows))
    return jobs, te_sizes, tr_scored


def _dispatch(series, jobs, n_jobs):
    """Run ``_ticker_task`` over tickers (serial when n_jobs==1, else a thread pool); flatten results."""
    payloads = [(series[tk][1], jobs[tk]) for tk in series if jobs[tk]]
    if n_jobs == 1:
        results = [_ticker_task(pl) for pl in payloads]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_ticker_task, payloads))
    return [row for chunk in results for row in chunk]


def _score(y, dates, preds, h):
    """Pooled per-obs QLIKE for every model + date-clustered DM of each GARCH variant vs HAR."""
    e = {m: M.per_obs_qlike(y, preds[m], floor=FL) for m in preds}
    q = {m: float(np.mean(e[m])) for m in e}
    dm = {}
    for v in VARIANTS:
        r = ST.date_clustered_dm(e[LABEL[v]], e["HAR"], dates, h)
        dm[LABEL[v]] = {"p_value": float(r["p_value"]), "mean_diff": float(r["mean_diff"])}
    return q, dm


def run_garch(market, load_fn=None, n_jobs=1):
    """GARCH/GJR-vs-HAR walk-forward for a market; returns the JSON-serialisable result dict (no write)."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    series = _ticker_series(frames)
    out = {}
    min_obs = config.MIN_TRAIN_OBS
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        folds = _active_folds(a, embargo, min_rows)
        # A per-stock GARCH is undefined without own history: score ONLY ticker-folds with >= min_obs
        # returns before the fold, and drop those rows from EVERY model so HAR/GARCH/GJR stay on the same
        # rows. (Without this, a newly-listed ticker's fallback variance is ~0 -> floored forecast ->
        # astronomical QLIKE, i.e. degenerate garbage rather than a meaningful GARCH-vs-HAR signal.)
        n_excluded = 0
        kept = []
        for (k, ts, tend, tr, te) in folds:
            est = _estimable(series, te["ticker"].unique(), int((ts - embargo).value), min_obs)
            keep = te["ticker"].isin(est)
            n_excluded += int((~keep).sum())
            te_s = te[keep].reset_index(drop=True)
            if len(te_s):
                kept.append((k, ts, tend, tr, te_s))
        folds = kept
        if not folds:
            continue
        last_k = folds[-1][0]
        jobs, te_sizes, tr_scored = _build_jobs(series, folds, embargo, h, last_k, min_obs)
        rows = _dispatch(series, jobs, n_jobs)

        g_test = {k: np.full(n, np.nan) for k, n in te_sizes.items()}
        j_test = {k: np.full(n, np.nan) for k, n in te_sizes.items()}
        g_train = np.full(len(tr_scored), np.nan)
        j_train = np.full(len(tr_scored), np.nan)
        n_fb = {"GARCH": 0, "GJR-GARCH": 0}
        for (rh, k, trows, gt, jt, g_ok, j_ok, trrows, gtr, jtr) in rows:
            g_test[k][trows] = gt; j_test[k][trows] = jt
            n_fb["GARCH"] += int(not g_ok); n_fb["GJR-GARCH"] += int(not j_ok)
            if trrows is not None:
                g_train[trrows] = gtr; j_train[trrows] = jtr

        y_p, dt_p, har_p, g_p, j_p = [], [], [], [], []
        for (k, ts, tend, tr, te) in folds:
            har_p.append(FM._har_ols(tr, te))
            g_p.append(g_test[k]); j_p.append(j_test[k])
            y_p.append(te["y"].to_numpy(float)); dt_p.append(te["date"].to_numpy())
        y = np.concatenate(y_p); dates = np.concatenate(dt_p)
        preds = {"HAR": np.concatenate(har_p), "GARCH": np.concatenate(g_p),
                 "GJR-GARCH": np.concatenate(j_p)}
        if any(np.isnan(v).any() for v in preds.values()):
            raise ValueError(f"unfilled forecast rows at h={h} (ticker/date misalignment)")
        q, dm = _score(y, dates, preds, h)

        tr_last = folds[-1][3]                      # full train rows (HAR is fit on all of them)
        y_tr = tr_scored["y"].to_numpy(float)       # scored on the estimable subset (== g_train order)
        tr_q = {"HAR": float(np.mean(M.per_obs_qlike(y_tr, FM._har_ols(tr_last, tr_scored), floor=FL))),
                "GARCH": float(np.mean(M.per_obs_qlike(y_tr, g_train, floor=FL))),
                "GJR-GARCH": float(np.mean(M.per_obs_qlike(y_tr, j_train, floor=FL)))}
        out[f"h{h}"] = {
            "n": int(len(y)),
            "n_excluded": int(n_excluded),
            "qlike": q,
            "gain_vs_HAR_pct": {LABEL[v]: (q["HAR"] - q[LABEL[v]]) / q["HAR"] * 100.0 for v in VARIANTS},
            "dm_vs_HAR": dm,
            "n_fallback": n_fb,
            "train_metrics": tr_q,
            "fit_diagnostics": {m: {"verdict": "overfit" if q[m] > tr_q[m] * 1.25 else "ok",
                                    "train_qlike": tr_q[m], "test_qlike": q[m]} for m in preds},
        }
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        print(f"\n=== {market} {h} (n={r['n']:,}, excluded={r['n_excluded']:,}) ===", flush=True)
        for m in ("GARCH", "GJR-GARCH", "HAR"):
            print(f"  {m:10s} QLIKE {r['qlike'][m]:.4f}", flush=True)
        for v in ("GARCH", "GJR-GARCH"):
            d = r["dm_vs_HAR"][v]
            print(f"  DM {v} vs HAR: {r['gain_vs_HAR_pct'][v]:+.2f}% (p={d['p_value']:.3f}, "
                  f"mean_diff={d['mean_diff']:+.2e}) | fallbacks={r['n_fallback'][v]}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    out = run_garch(market, n_jobs=n_jobs)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"garch_{market}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
