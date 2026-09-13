"""Hybrid feature-concat gamma-GBM: own-history+earnings block ++ richer graph-spillover block.

Does a 7-feature cross-stock spillover block add out-of-sample value to the per-stock ``GBM+earn`` model,
and does it beat the single-feature ``GBM+earn+corr``? Score = pooled per-observation QLIKE + date-clustered
Diebold-Mariano at each horizon, plus the residual error-correlation between ``GBM+earn`` and a graph-only
forecast (the ~0.98 orthogonality check). Honest prior: NO-GO (see ``../requirements/requirements.md``).

Same protocol as ``scripts/eda/paper_metrics_sp500.py`` / ``baselines/2026-09-13_principled_har_features``:
expanding walk-forward over ``S1.FOLDS`` with an ``int(h*1.6)+5``-day embargo, per-fold TRAIN-only graph,
seed-averaged predictions, real VN earnings injected for HOSE. Reuses ``FM``/``S1`` verbatim (read-only).

Run: ``python run_hybrid.py [hose|sp500]``.  Output: results/gamma_gbm/gbm_earn_graph_<market>.json
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
import spillover_features as SF  # noqa: E402
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
OWN8 = [c for c in FM.OWN if c != "rq"]                       # own-history minus rq
EARN_PARQUET = REPO / "results" / "gamma_gbm" / "hose_earnings_combined.parquet"

_EARN = "GBM+earn"
_CORR = "GBM+earn+corr"
_GRAPH = "GBM+earn+graph"
_GO = "graph_only"


def _earn_dates(market, edates):
    """Resolve the earnings-date map: SP500 keeps the loader's dates; VN markets inject the REAL crawled
    announcement dates from ``EARN_PARQUET`` when present (as ``paper_metrics_sp500`` does)."""
    if market == "sp500":
        return edates
    if EARN_PARQUET.exists():
        e = pd.read_parquet(EARN_PARQUET)
        return {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return edates


def _fold_qlike(tr, cols):
    """In-sample (train) pooled QLIKE of the seed-averaged GBM on ``cols`` (for the fit-diagnostics verdict)."""
    ptr = np.mean([FM.gbm(tr, tr, cols, s) for s in FM.SEEDS], 0)
    return float(np.mean(M.per_obs_qlike(tr["y"].to_numpy(float), ptr, floor=FL)))


def _success(out):
    """Economic go/no-go: the spillover block is retained only if EVERY horizon shows a positive,
    sign-consistent QLIKE gain over ``GBM+earn`` >= SUCCESS_MIN_GAIN_PCT with DM p < SUCCESS_DM_P, the
    richer block also beats ``GBM+earn+corr``, and no overfit verdict."""
    hs = [k for k in out if k.startswith("h")]
    if not hs:
        return False
    gains_ok = all(out[k]["gain_vs_earn_pct"] >= config.SUCCESS_MIN_GAIN_PCT for k in hs)
    dm_ok = all(out[k]["dm"][f"{_GRAPH}_vs_{_EARN}"] < config.SUCCESS_DM_P for k in hs)
    beats_corr = all(out[k]["qlike"][_GRAPH] < out[k]["qlike"][_CORR] for k in hs)
    fit_ok = all(out[k]["fit_diagnostics"][_GRAPH]["verdict"] == "ok" for k in hs)
    return bool(gains_ok and dm_ok and beats_corr and fit_ok)


def _checkpoint(out, out_path):  # pragma: no cover - I/O side effect, exercised only in real runs
    """Atomically write accumulated results so a Colab disconnect keeps completed horizons."""
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"[checkpoint] wrote {out_path.name} ({len([k for k in out if k.startswith('h')])} horizons)", flush=True)


def run_hybrid(market, load_fn=None, out_path=None):
    """Walk-forward hybrid comparison for a market; returns the JSON-serialisable result dict.

    Raises ``ValueError`` if no earnings dates are available (the hybrid requires the earnings block) or if
    no fold is ever scored (fail loud rather than write an empty result). When ``out_path`` is given, flush
    accumulated results after EACH horizon (Colab disconnect resilience)."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, edates = load_fn(market)
    edates = _earn_dates(market, edates)
    if not edates:
        raise ValueError("no earnings dates available -- this hybrid requires the earnings block "
                         f"(expected {EARN_PARQUET.name} for hose)")
    cols = {_EARN: OWN8 + FM.EARN, _CORR: OWN8 + FM.EARN + [SF.CORR_COL],
            _GRAPH: OWN8 + FM.EARN + SF.SPILL_COLS}
    allcols = {**cols, _GO: SF.SPILL_COLS}
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        preds = {m: [] for m in allcols}
        yy, dts, last_trf = [], [], None
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < min_rows:
                continue
            Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
            fold = a[(a.date >= S1.TRAIN_START) & (a.date < tend)]
            fold = SF.add_spillover_features(fold, tickers, Wc)
            trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
            tef = fold[(fold.date >= ts) & (fold.date < tend)]
            for m, cc in allcols.items():
                preds[m].append(np.mean([FM.gbm(trf, tef, cc, s) for s in FM.SEEDS], 0))
            yy.append(tef["y"].to_numpy(float)); dts.append(tef["date"].to_numpy()); last_trf = trf
        if not yy:
            continue
        y = np.concatenate(yy); dates = np.concatenate(dts)
        pooled = {m: np.concatenate(preds[m]) for m in allcols}
        e = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in allcols}
        q = {m: float(np.mean(e[m])) for m in allcols}
        dm = {f"{_GRAPH}_vs_{_EARN}": float(ST.date_clustered_dm(e[_GRAPH], e[_EARN], dates, h)["p_value"]),
              f"{_GRAPH}_vs_{_CORR}": float(ST.date_clustered_dm(e[_GRAPH], e[_CORR], dates, h)["p_value"])}
        err_corr = float(np.corrcoef(pooled[_EARN] - y, pooled[_GO] - y)[0, 1])
        train_q = {m: _fold_qlike(last_trf, cc) for m, cc in allcols.items()}
        diags = {m: {"verdict": "overfit" if q[m] > train_q[m] * config.OVERFIT_RATIO else "ok",
                     "train_qlike": train_q[m], "test_qlike": q[m]} for m in allcols}
        out[f"h{h}"] = {"n": int(len(y)), "qlike": q, "dm": dm, "err_corr": err_corr,
                        "gain_vs_earn_pct": (q[_EARN] - q[_GRAPH]) / q[_EARN] * 100,
                        "gain_vs_corr_pct": (q[_CORR] - q[_GRAPH]) / q[_CORR] * 100,
                        "train_metrics": train_q, "test_metrics": q, "fit_diagnostics": diags}
        _checkpoint(out, out_path)                  # flush after each horizon (disconnect-resilient)
    if not [k for k in out if k.startswith("h")]:
        raise ValueError(f"no fold scored for any horizon (min_rows={min_rows}) -- refusing to write an "
                         "empty result")
    out["success"] = _success(out)
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        if not h.startswith("h"):
            continue
        print(f"\n=== {market} {h} (n={r['n']:,}) ===", flush=True)
        for m in (_EARN, _CORR, _GRAPH, _GO):
            print(f"  {m:16s} QLIKE {r['qlike'][m]:.4f}", flush=True)
        print(f"  +graph vs +earn : {r['gain_vs_earn_pct']:+.2f}% (DM p={r['dm'][f'{_GRAPH}_vs_{_EARN}']:.3g})",
              flush=True)
        print(f"  +graph vs +corr : {r['gain_vs_corr_pct']:+.2f}% (DM p={r['dm'][f'{_GRAPH}_vs_{_CORR}']:.3g})",
              flush=True)
        print(f"  err_corr(earn, graph_only residuals): {r['err_corr']:.3f}", flush=True)
    print(f"\n{market}: success={out['success']}  "
          f"(criterion: every-h gain>={config.SUCCESS_MIN_GAIN_PCT}% & DM p<{config.SUCCESS_DM_P} & beats "
          "+corr & no overfit; warn: a tiny 'DM-significant' gain at large n is NOT a real win)", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    outp = REPO / "results" / "gamma_gbm" / f"gbm_earn_graph_{market}.json"
    out = run_hybrid(market, out_path=outp)
    _print(market, out)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
