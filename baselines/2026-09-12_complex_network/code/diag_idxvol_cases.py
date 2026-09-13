"""Case-level diagnostic: WHY does the causal index-vol feature hurt the per-stock GBM? (train / val / test)

For each horizon, take the last valid walk-forward fold, split its causal training window (by date) into a
train-core and a held-out validation tail (``IDXVOL_VAL_FRAC``), and keep the fold's test window. Fit
GBM(own) and GBM(own+idxvol) on the train-core (seed-averaged), then score per-(ticker, date) QLIKE on all
three splits. Output, per split: pooled QLIKE for both models (idxvol should LOWER train QLIKE but RAISE
val/test QLIKE = the overfit signature) and the individual ticker-day cases where adding idxvol changed the
QLIKE most -- concrete proof of where the market-vol feature overfits in train and misfires in val/test.

Run: ``python diag_idxvol_cases.py [hose|sp500]``.
Output: results/gamma_gbm/complex_network_<market>_idxvol_cases.json + docs/reports/<date>_complex_network_<market>_idxvol_cases.html
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
import verify_index_vol_feature as IV  # noqa: E402  (reuse merge_idx_rv + FEAT)

FL = FM.FL
MODELS = {"GBM": FM.OWN, "GBM+idxvol": FM.OWN + [IV.FEAT]}


def _val_cut(train_dates):
    """Date boundary so the last ``IDXVOL_VAL_FRAC`` of the (unique) training dates become validation."""
    u = np.sort(np.unique(train_dates))
    return u[int(len(u) * (1.0 - config.IDXVOL_VAL_FRAC))]


def _preds(tr_core, frame):
    """Seed-averaged GBM prediction for each model on ``frame`` (fit on ``tr_core``)."""
    return {m: np.mean([FM.gbm(tr_core, frame, cols, s) for s in FM.SEEDS], 0) for m, cols in MODELS.items()}


def _cases(frame, pr, best_for_idxvol):
    """Top-K per-(ticker,date) cases ranked by the QLIKE change from adding idxvol.

    ``best_for_idxvol=True`` ranks where idxvol HELPED most (q_own-q_idx largest, the train overfit);
    ``False`` ranks where it HURT most (q_idx-q_own largest, the val/test misfire)."""
    y = frame["y"].to_numpy(float)
    q = {m: M.per_obs_qlike(y, pr[m], floor=FL) for m in MODELS}
    dq = q["GBM+idxvol"] - q["GBM"]
    order = np.argsort(dq if not best_for_idxvol else -dq)[::-1][:config.IDXVOL_CASE_K]
    tk = frame["ticker"].to_numpy(); dt = frame["date"].to_numpy()
    rv = frame[IV.FEAT].to_numpy(float)
    rows = []
    for i in order:
        rows.append({"ticker": str(tk[i]), "date": str(pd.Timestamp(dt[i]).date()),
                     "y": float(y[i]), "pred_own": float(pr["GBM"][i]), "pred_idxvol": float(pr["GBM+idxvol"][i]),
                     "idx_rv": float(rv[i]), "qlike_own": float(q["GBM"][i]),
                     "qlike_idxvol": float(q["GBM+idxvol"][i]), "dqlike": float(dq[i])})
    return rows


def _split_result(frame, pr, best_for_idxvol):
    """Pooled QLIKE (both models) + top cases for one split."""
    y = frame["y"].to_numpy(float)
    q = {m: float(np.mean(M.per_obs_qlike(y, pr[m], floor=FL))) for m in MODELS}
    return {"n": int(len(frame)), "qlike_own": q["GBM"], "qlike_idxvol": q["GBM+idxvol"],
            "gain_pct": (q["GBM"] - q["GBM+idxvol"]) / q["GBM"] * 100.0,
            "cases": _cases(frame, pr, best_for_idxvol)}


def diag_horizon(a, h, min_rows):
    """Train-core/val/test case diagnostic for one horizon on panel ``a``; None if no valid fold."""
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    chosen = None
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) and len(tr) >= min_rows:
            chosen = (tr, te, ts)                                 # keep the LAST valid fold
    if chosen is None:
        return None
    tr, te, ts = chosen
    cut = _val_cut(tr["date"].to_numpy())
    tr_core = tr[tr.date < cut]
    val = tr[tr.date >= cut]
    pr = {name: _preds(tr_core, fr) for name, fr in (("train", tr_core), ("val", val), ("test", te))}
    return {"fold_start": str(ts.date()), "val_cut": str(pd.Timestamp(cut).date()),
            "train": _split_result(tr_core, pr["train"], best_for_idxvol=True),   # where idxvol overfits
            "val": _split_result(val, pr["val"], best_for_idxvol=False),          # where it misfires
            "test": _split_result(te, pr["test"], best_for_idxvol=False)}


def run_cases(market, load_fn=None, index_fn=None):
    """Case diagnostic across all horizons for a market."""
    load_fn = load_fn or FM.load
    index_fn = index_fn or market_index.load_index
    min_rows = 30000 if market == "sp500" else 3000
    frames, _, _ = load_fn(market)
    frames = IV.merge_idx_rv(frames, index_fn(market))
    out = {"market": market, "horizons": {}}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        res = diag_horizon(a, h, min_rows)
        if res is not None:
            out["horizons"][f"h{h}"] = res
    return out


_CSS = """body{font:13px/1.5 system-ui,Segoe UI,Arial;margin:24px;color:#1a1a1a;max-width:1100px}
h1{font-size:21px}h2{font-size:16px;margin-top:26px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:14px;margin:14px 0 4px}table{border-collapse:collapse;margin:6px 0;font-size:12px}
th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#f2f2f2}td.l,th.l{text-align:left}
.good{color:#0a7d2c}.bad{color:#c0271a}.muted{color:#666}"""


def _split_table(market, hblocks):  # pragma: no cover - HTML formatting
    rows = ["<tr><th class=l>horizon</th><th>split</th><th>n</th><th>QLIKE own</th><th>QLIKE +idxvol</th>"
            "<th>gain %</th></tr>"]
    for h, r in hblocks.items():
        for sp in ("train", "val", "test"):
            s = r[sp]; cls = "good" if s["gain_pct"] > 0 else "bad"
            rows.append(f"<tr><td class=l>{h}</td><td>{sp}</td><td>{s['n']:,}</td>"
                        f"<td>{s['qlike_own']:.4f}</td><td>{s['qlike_idxvol']:.4f}</td>"
                        f"<td class={cls}>{s['gain_pct']:+.2f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _cases_table(cases):  # pragma: no cover - HTML formatting
    rows = ["<tr><th class=l>ticker</th><th class=l>date</th><th>actual y</th><th>pred own</th>"
            "<th>pred +idxvol</th><th>idx_rv</th><th>QLIKE own</th><th>QLIKE +idxvol</th><th>ΔQLIKE</th></tr>"]
    for c in cases:
        cls = "bad" if c["dqlike"] > 0 else "good"
        rows.append(f"<tr><td class=l>{c['ticker']}</td><td class=l>{c['date']}</td><td>{c['y']:.2e}</td>"
                    f"<td>{c['pred_own']:.2e}</td><td>{c['pred_idxvol']:.2e}</td><td>{c['idx_rv']:.4f}</td>"
                    f"<td>{c['qlike_own']:.3f}</td><td>{c['qlike_idxvol']:.3f}</td>"
                    f"<td class={cls}>{c['dqlike']:+.3f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def build_html(res, date):  # pragma: no cover - orchestration + HTML assembly
    m = res["market"]
    parts = [f"<h1>Why the index-vol feature hurts the per-stock GBM &mdash; case-level ({m.upper()})</h1>",
             "<p class=muted>Per horizon: last walk-forward fold, train-core / held-out validation / test. "
             "idxvol = own-history + trailing index realized vol. Gain % &gt; 0 means idxvol helps that split.</p>",
             "<h2>1. Pooled QLIKE per split (the overfit signature)</h2>",
             _split_table(m, res["horizons"]),
             "<p class=muted>idxvol typically LOWERS train QLIKE (positive train gain) while raising val/test "
             "QLIKE (negative gain): extra capacity fits training ticker-days that do not generalise.</p>"]
    for h, r in res["horizons"].items():
        parts.append(f"<h2>2. {h} worst/most-overfit ticker-day cases (fold {r['fold_start']}, "
                     f"val_cut {r['val_cut']})</h2>")
        parts.append("<h3>train &mdash; where idxvol helped most in-sample (overfit)</h3>" + _cases_table(r["train"]["cases"]))
        parts.append("<h3>validation &mdash; where idxvol hurt most</h3>" + _cases_table(r["val"]["cases"]))
        parts.append("<h3>test &mdash; where idxvol hurt most</h3>" + _cases_table(r["test"]["cases"]))
    html = f"<!doctype html><html><head><meta charset=utf-8><style>{_CSS}</style></head><body>" \
           + "".join(parts) + "</body></html>"
    outp = REPO / "docs" / "reports" / f"{date}_complex_network_{m}_idxvol_cases.html"
    outp.write_text(html, encoding="utf-8")
    return outp


def _print(res):  # pragma: no cover - console formatting only
    for h, r in res["horizons"].items():
        print(f"{res['market']} {h}: train gain {r['train']['gain_pct']:+.2f}% | "
              f"val {r['val']['gain_pct']:+.2f}% | test {r['test']['gain_pct']:+.2f}%", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON + HTML
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    date = sys.argv[2] if len(sys.argv) > 2 else "2026-09-13"
    res = run_cases(market)
    _print(res)
    outp = REPO / "results" / "gamma_gbm" / f"complex_network_{market}_idxvol_cases.json"
    outp.write_text(json.dumps(res, indent=2))
    html = build_html(res, date)
    print(f"saved {outp.relative_to(REPO)} + {html.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
