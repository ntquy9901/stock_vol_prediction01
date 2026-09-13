"""GBM(own-8) residual-graph refiner (falsification test): does a causal graph-spillover refiner trained on
the GBM residual add OOS value on the QLIKE + DM arbiter?

Base = seed-averaged gamma-GBM on own-8 (OOS per walk-forward fold, identical to full_compare). The refiner is
a ridge trained on EARLIER folds' OOS residuals (expanding, causal) from causal neighbour-spillover features,
reconstructed additively in log-variance so the final forecast stays positive. See ``../design/design.md``.

Run: ``python run_residual_graph.py [hose|sp500]``.  Output: results/gamma_gbm/residual_graph_<market>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
OWN = [c for c in FM.OWN if c != "rq"]     # own-8 (drop rq) -- matches the paper_models GBM base
NB_FEATS = list(S1.GRAPH)                  # causal neighbour-spillover features (structural set, like FM.OWN)


def fold_stream(a, tickers, h, min_rows):
    """Per scored fold (time order): OOS seed-averaged GBM(own-8) + causal neighbour features + realised y.

    Every field is causal at prediction time: the GBM, the top-10 correlation graph, and the neighbour
    features are all built from train/day-t data only; ``y`` (the h-ahead target) is returned for scoring and
    for the residual LABEL, never used as an input feature."""
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    stream = []
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < min_rows:
            continue
        Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
        fold = a[(a.date >= S1.TRAIN_START) & (a.date < tend)].copy()
        gf = S1.graph_feats(fold, tickers, Wc, "")
        for c in NB_FEATS:
            fold[c] = gf[c].fillna(0.0)
        trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
        tef = fold[(fold.date >= ts) & (fold.date < tend)]
        gbm = np.mean([FM.gbm(trf, tef, OWN, s) for s in FM.SEEDS], 0)
        stream.append({"y": tef["y"].to_numpy(float), "gbm": gbm,
                       "feats": tef[NB_FEATS].to_numpy(float), "dates": tef["date"].to_numpy()})
    return stream


def refine(stream):
    """Expanding causal residual stack: a ridge trained on EARLIER folds' OOS residuals refines each fold.

    Two variants per fold (both positive by construction, ``gbm * exp(.)``):
      * ``raw``  = unclipped ridge on the raw log-variance residual (diagnostic; blows up on HOSE floor tails);
      * ``clip`` = robust ridge on the winsorized residual (+-RESID_CLIP), prediction clipped to +-RESID_CLIP.
    Folds before ``MIN_STACK_ROWS`` earlier rows are identity (resid_hat = 0). ``train_mse``/``test_mse`` are the
    robust ridge's winsorized-residual MSEs on the active folds (overfit diagnostics).
    Returns (y, gbm, raw, clip, dates, train_mse, test_mse)."""
    c = config.RESID_CLIP
    y_all, gbm_all, raw_all, clip_all, dt_all = [], [], [], [], []
    seen_x, seen_r, train_mse, test_mse = [], [], [], []
    for fd in stream:
        gbm = np.maximum(fd["gbm"], FL)
        resid = np.log(np.maximum(fd["y"], FL)) - np.log(gbm)
        if sum(len(x) for x in seen_x) >= config.MIN_STACK_ROWS:
            xtr = np.concatenate(seen_x); rtr = np.concatenate(seen_r); rwin = np.clip(rtr, -c, c)
            sc = StandardScaler().fit(xtr)
            xt = sc.transform(fd["feats"])
            rh_raw = Ridge(alpha=config.RIDGE_ALPHA).fit(sc.transform(xtr), rtr).predict(xt)
            rob = Ridge(alpha=config.RIDGE_ALPHA).fit(sc.transform(xtr), rwin)
            rh_clip = np.clip(rob.predict(xt), -c, c)
            train_mse.append(float(np.mean((np.clip(rob.predict(sc.transform(xtr)), -c, c) - rwin) ** 2)))
            test_mse.append(float(np.mean((rh_clip - np.clip(resid, -c, c)) ** 2)))
        else:
            rh_raw = np.zeros_like(resid); rh_clip = np.zeros_like(resid)
        raw_all.append(gbm * np.exp(rh_raw)); clip_all.append(gbm * np.exp(rh_clip))
        y_all.append(fd["y"]); gbm_all.append(fd["gbm"]); dt_all.append(fd["dates"])
        seen_x.append(fd["feats"]); seen_r.append(resid)
    return (np.concatenate(y_all), np.concatenate(gbm_all), np.concatenate(raw_all),
            np.concatenate(clip_all), np.concatenate(dt_all), train_mse, test_mse)


def _verdict(gain_pct, dm_p):
    """GO only if the refiner both lowers QLIKE and is DM-significant (a big-n p-value alone is not a win)."""
    return bool(gain_pct > 0 and dm_p < config.SUCCESS_DM_P)


def _checkpoint(out, out_path):  # pragma: no cover - I/O side effect, exercised only in real runs
    """Atomically write accumulated results so a Colab disconnect keeps completed horizons."""
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"[checkpoint] wrote {out_path.name} ({len([k for k in out if k.startswith('h')])} horizons)", flush=True)


def run(market, load_fn=None, out_path=None):
    """Residual-graph refiner vs GBM(own-8) for a market; per-horizon QLIKE + DM + verdict + kill criterion."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, _ = load_fn(market)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, {}, h)
        tickers = sorted(a["ticker"].unique())
        stream = fold_stream(a, tickers, h, min_rows)
        if not stream:
            continue
        y, gbm, raw, clip, dates, train_mse, test_mse = refine(stream)
        e_gbm = M.per_obs_qlike(y, gbm, floor=FL)
        q_gbm = float(np.mean(e_gbm))
        variant = {}
        for name, pred in (("raw", raw), ("clip", clip)):
            e = M.per_obs_qlike(y, pred, floor=FL)
            q = float(np.mean(e))
            dm = ST.date_clustered_dm(e, e_gbm, dates, h)
            variant[name] = {"qlike": q, "gain_vs_gbm_pct": (q_gbm - q) / q_gbm * 100.0,
                             "dm_vs_gbm": {"p_value": float(dm["p_value"]), "mean_diff": float(dm["mean_diff"])}}
        trm = float(np.mean(train_mse)) if train_mse else 0.0
        tem = float(np.mean(test_mse)) if test_mse else 0.0
        # the bounded (clip) refiner is the fair best case; GO requires IT to beat GBM
        gain_c, dmp_c = variant["clip"]["gain_vs_gbm_pct"], variant["clip"]["dm_vs_gbm"]["p_value"]
        out[f"h{h}"] = {
            "n": int(len(y)),
            "qlike_gbm": q_gbm,
            "raw": variant["raw"],
            "clip": variant["clip"],
            "refiner_active_folds": len(test_mse),
            "fit_diagnostics": {"resid_train_mse": trm, "resid_test_mse": tem,
                                "verdict": "overfit" if (trm > 0 and tem > trm * config.OVERFIT_RATIO) else "ok"},
            "verdict": "GO" if _verdict(gain_c, dmp_c) else "NO-GO",
        }
        _checkpoint(out, out_path)
    out["success"] = bool([k for k in out if k.startswith("h")]) and all(
        out.get(f"h{x}", {}).get("verdict") == "GO" for x in config.SUCCESS_HORIZONS)
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        if not str(h).startswith("h"):
            continue
        print(f"\n=== {market} {h} (n={r['n']:,}, refiner folds={r['refiner_active_folds']}) ===", flush=True)
        print(f"  QLIKE gbm={r['qlike_gbm']:.4f}", flush=True)
        for name in ("raw", "clip"):
            v = r[name]
            print(f"    {name:4s} QLIKE={v['qlike']:.4f} gain={v['gain_vs_gbm_pct']:+.3f}% "
                  f"DM p={v['dm_vs_gbm']['p_value']:.3g} -> ({'beats' if v['gain_vs_gbm_pct'] > 0 else 'worse'})",
                  flush=True)
        print(f"  verdict={r['verdict']}", flush=True)
    print(f"\n{market}: success={out['success']} (GO requires gain>0 & DM p<{config.SUCCESS_DM_P} at "
          f"h{'/'.join(map(str, config.SUCCESS_HORIZONS))})", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, writes JSON
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    outp = REPO / "results" / "gamma_gbm" / f"residual_graph_{market}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = run(market, out_path=outp)
    _print(market, out)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
