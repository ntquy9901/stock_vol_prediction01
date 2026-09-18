"""Walk-forward driver (Hướng B falsification): does a zero-shot foundation-model forecast, added as ONE causal
feature to the champion own-history gamma-GBM (GBME), beat GBME out-of-sample on QLIKE?

Two stages, pre-gated:
  Stage 0  -- the frozen Chronos-Bolt zero-shot forecast, used DIRECTLY as the predictor, vs HAR (date-clustered
              DM per horizon). Records whether the foundation model even matches HAR (prior = NO-GO).
  Stage 1  -- GBME vs GBME+FND (the zero-shot forecast + optional spread as extra causal columns) vs GBME+PLAC
              (a wrong-ticker placebo forecast that MUST NOT reproduce any gain). Date-clustered DM, spike-robust,
              per-fold. Pre-registered kill: GBME+FND beats GBME DM-sig at >= KILL_MIN_HORIZONS horizons AND the
              placebo does not beat AND the win survives the regime-spike exclusion -> else NO-GO.

Output: results/gamma_gbm/foundation_<market>_h<h>.json, one per horizon, atomic checkpoint after every fold.

Run: python run_foundation.py [hose|sp500] [--smoke] [--rebuild-cache]
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(REPO / "scripts" / "quality_gate"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
import overfit_check as OF  # noqa: E402
import foundation_config as C  # noqa: E402
import foundation_forecaster as FF  # noqa: E402

FL = FM.FL
GBME, FND, PLAC = "GBME", "GBME+FND", "GBME+PLAC"
HAR, ZS = "HAR", "Chronos_ZS"
ORDER = [GBME, FND, PLAC]                 # the GBM family that carries full train/val/test + fit evidence
VALID_LEN = 22                           # trailing train dates held out as a true val slice (mirrors siblings)


def _own8():
    """OWN-8 own-history feature list, single-sourced from the paper_models config, loaded by path so it does NOT
    register a second bare ``config`` module (avoids the sys.modules collision)."""
    cfg_path = REPO / "baselines" / "2026-09-13_paper_models" / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("paper_models_config", cfg_path)
    pmc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmc)
    return pmc.own_set(FM.OWN)


OWN = _own8()


def _load_earn(market, edates):
    """Replace edates with the REAL crawled VN announcement dates if present (mirrors the sibling drivers);
    SP500 keeps its own earnings from FM.load."""
    if market == "sp500":
        return edates
    ep = REPO / "results" / "gamma_gbm" / "hose_earnings_combined.parquet"
    if ep.exists():
        e = pd.read_parquet(ep)
        return {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return edates


def _fnd_cols(prefix):
    """Extra causal feature columns for a foundation model: the horizon-matched forecast (+ spread when on)."""
    return [prefix] + ([f"{prefix}_spread"] if C.USE_SPREAD else [])


def _placebo(cache, h):
    """Wrong-ticker placebo: relabel each source ticker to the target it serves under a cyclic map, so merging on
    (ticker, date) hands each ticker ANOTHER ticker's foundation forecast at the same date (destroys firm
    identity, keeps the marginal + calendar). Returns a (ticker, date, fnd_plac[, fnd_plac_spread]) frame."""
    tickers = sorted(cache["ticker"].unique())
    nxt = {t: tickers[(i + 1) % len(tickers)] for i, t in enumerate(tickers)}
    inv = {src: tgt for tgt, src in nxt.items()}             # source ticker -> the target row it fills
    cols = {"date": cache["date"], "ticker": cache["ticker"].map(inv), "fnd_plac": cache[f"fnd_h{h}"]}
    if C.USE_SPREAD:
        cols["fnd_plac_spread"] = cache[f"fspread_h{h}"]
    return pd.DataFrame(cols)


def _attach(a0, cache, h, panel_cols):
    """Trim the panel to the columns the models actually use (the full enriched panel carries ~30 columns,
    most unused — trimming keeps the merge + GBM working set small), then inner-join the horizon-``h``
    foundation forecast and its wrong-ticker placebo so GBME / GBME+FND / GBME+PLAC are scored on the SAME
    paired rows."""
    a0 = a0[[c for c in panel_cols if c in a0.columns]]
    keep = ["ticker", "date", f"fnd_h{h}"] + ([f"fspread_h{h}"] if C.USE_SPREAD else [])
    fh = cache[keep].rename(columns={f"fnd_h{h}": "fnd", f"fspread_h{h}": "fnd_spread"})
    a = a0.merge(fh, on=["ticker", "date"], how="inner")
    return a.merge(_placebo(cache, h), on=["ticker", "date"], how="inner")


def _metrics5(y, p):
    """The 5 reported metrics (DirAcc dropped per paper style)."""
    return {"mse": float(mean_squared_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
            "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}


def _predict_gbm(trf, combo, cols, seeds):
    """Seed-ensembled champion gamma-GBM prediction over the ``combo`` rows."""
    return np.mean([FM.gbm(trf, combo, cols, s) for s in seeds], 0)


def verdict(gain_pct, dm_p):
    """A horizon 'beats' iff the QLIKE gain is strictly positive AND the DM p-value clears DM_ALPHA."""
    return bool(gain_pct > C.GAIN_MIN * 100.0 and dm_p < C.DM_ALPHA)


def success(docs):
    """Pre-registered: GBME+FND beats GBME at >= KILL_MIN_HORIZONS horizons, no placebo win, and every beating
    horizon survives the regime-spike exclusion (HOSE). Missing evidence => not a success."""
    beats = [h for h, d in docs.items() if d.get("verdict", {}).get("beats")]
    if len(beats) < C.KILL_MIN_HORIZONS:
        return False
    if any(d.get("placebo", {}).get("beats") for d in docs.values()):
        return False
    for h in beats:
        sr = docs[h].get("spike_robustness")
        if sr is not None and not sr.get("beats_ex_spike"):
            return False
    return True


def _safe_dm(err_a, err_b, dates, h):
    """Date-clustered DM, returning p=1.0 for the degenerate cases the reused DM cannot handle (identical loss
    series, or too few surviving dates for the HLN factor)."""
    degenerate = {"p_value": 1.0, "mean_diff": 0.0, "dm_hln": 0.0, "n_dates": int(np.unique(dates).size)}
    if np.allclose(err_a, err_b):
        return degenerate
    try:
        return ST.date_clustered_dm(err_a, err_b, dates, h)
    except ValueError:
        return degenerate


def _spike_mask(dates):
    """Boolean mask of test dates inside any configured regime-spike window (HOSE robustness)."""
    d = pd.to_datetime(dates)
    m = np.zeros(len(d), bool)
    for lo, hi in C.SPIKE_WINDOWS:
        m |= (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    return m


def _pool_doc(h, market, seeds, preds, yy, dts, per_fold, spike):
    """Per-horizon result doc: pooled 5-metric train/val/test for the GBM family, fit_diagnostics, Stage-0
    (Chronos zero-shot vs HAR), Stage-1 DM (GBME+FND vs GBME primary, vs placebo secondary), placebo gain,
    verdict, plus HOSE per-fold / spike robustness."""
    y = np.concatenate(yy["te"])
    dates = np.concatenate(dts)
    pooled = {m: np.concatenate(preds["te"][m]) for m in ORDER + [HAR, ZS]}
    err = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in pooled}
    metrics = {m: _metrics5(y, pooled[m]) for m in pooled}
    y_tr, y_va = np.concatenate(yy["tr"]), np.concatenate(yy["va"])
    train_metrics = {m: _metrics5(y_tr, np.concatenate(preds["tr"][m])) for m in ORDER}
    val_metrics = {m: _metrics5(y_va, np.concatenate(preds["va"][m])) for m in ORDER}
    fit = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in ORDER}
    dm_fnd = _safe_dm(err[FND], err[GBME], dates, h)         # primary: foundation feature beats the champion?
    dm_plac = _safe_dm(err[PLAC], err[GBME], dates, h)       # placebo: wrong-ticker feature vs champion
    dm_fnd_plac = _safe_dm(err[FND], err[PLAC], dates, h)    # isolate: real foundation vs placebo
    dm_zs_har = _safe_dm(err[ZS], err[HAR], dates, h)        # Stage-0: zero-shot vs HAR
    gain = (metrics[GBME]["qlike"] - metrics[FND]["qlike"]) / metrics[GBME]["qlike"] * 100.0
    gain_plac = (metrics[GBME]["qlike"] - metrics[PLAC]["qlike"]) / metrics[GBME]["qlike"] * 100.0
    gain_zs = (metrics[HAR]["qlike"] - metrics[ZS]["qlike"]) / metrics[HAR]["qlike"] * 100.0
    doc = {"market": market, "h": h, "n": int(len(y)), "n_folds": len(yy["te"]), "seeds": list(seeds),
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit,
           "stage0_zeroshot_vs_har": {"qlike_zeroshot": metrics[ZS]["qlike"], "qlike_har": metrics[HAR]["qlike"],
                                      "gain_pct": gain_zs, "dm_p": dm_zs_har["p_value"],
                                      "zeroshot_beats_har": bool(gain_zs > C.GAIN_MIN * 100.0
                                                                 and dm_zs_har["p_value"] < C.DM_ALPHA)},
           "dm": {f"{FND}_vs_{GBME}": dm_fnd["p_value"], f"{PLAC}_vs_{GBME}": dm_plac["p_value"],
                  f"{FND}_vs_{PLAC}": dm_fnd_plac["p_value"]},
           "gain_pct": {f"{FND}_vs_{GBME}": gain, f"{PLAC}_vs_{GBME}": gain_plac},
           "verdict": {"beats": verdict(gain, dm_fnd["p_value"]), "gain_pct": gain, "dm_p": dm_fnd["p_value"],
                       "criterion": "GBME+FND beats GBME: gain>GAIN_MIN AND DM p<DM_ALPHA"},
           "placebo": {"beats": verdict(gain_plac, dm_plac["p_value"]), "gain_pct": gain_plac,
                       "dm_p": dm_plac["p_value"]}}
    if per_fold:
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy["te"][i], preds["te"][m][i], floor=FL)))
                                     for i in range(len(yy["te"]))] for m in ORDER}
    if spike:
        sm = _spike_mask(dates)
        keep = ~sm
        if keep.any():
            dm_ex = _safe_dm(err[FND][keep], err[GBME][keep], dates[keep], h)
            qb, qz = float(np.mean(err[GBME][keep])), float(np.mean(err[FND][keep]))
            doc["spike_robustness"] = {
                "n_spike_obs": int(sm.sum()), "n_ex_spike_obs": int(keep.sum()),
                "qlike_ex_spike": {GBME: qb, FND: qz}, "gain_pct_ex_spike": (qb - qz) / qb * 100.0,
                "dm_p_ex_spike": dm_ex["p_value"], "beats_ex_spike": verdict((qb - qz) / qb * 100.0,
                                                                             dm_ex["p_value"])}
    return doc


def _checkpoint(doc, out_path):
    """Atomically write the per-horizon doc (tmp + replace) so a disconnect keeps the completed horizon."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out_path)


def run(market, load_fn=None, cache_fn=None, out_dir=None, smoke=False, rebuild_cache=False):
    """Walk-forward GBME vs GBME+FND vs GBME+PLAC (+ Stage-0 Chronos zero-shot vs HAR) for a market. Writes one
    JSON per horizon and returns {h: doc}. ``cache_fn(frames)`` supplies the foundation forecast cache (injected
    as a fast fake in tests); by default the real Chronos cache parquet is built/loaded once."""
    load_fn = load_fn or FM.load
    seeds = (FM.SEEDS[0],) if smoke else FM.SEEDS
    horizons = C.HORIZONS_SMOKE if smoke else C.HORIZONS
    fold_cap = 1 if smoke else None
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "gamma_gbm")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""

    frames, _sect, edates = load_fn(market)
    edates = _load_earn(market, edates)
    if cache_fn is not None:
        cache = cache_fn(frames)
    else:  # pragma: no cover - real Chronos cache build/load (covered by the opt-in real-model test, not here)
        cache_path = out_dir / f"foundation_cache_{market}.parquet"
        cache = FF.load_or_build_cache(frames, cache_path, cfg=C, rebuild=rebuild_cache)
    base_cols = OWN + (FM.EARN if bool(edates) else [])
    model_cols = {GBME: base_cols, FND: base_cols + _fnd_cols("fnd"), PLAC: base_cols + _fnd_cols("fnd_plac")}
    panel_cols = ["ticker", "date", "y"] + base_cols          # trim the enriched panel to only what the models use
    per_fold = market != "sp500"
    spike = market != "sp500"
    docs = {}
    for h in horizons:
        t0 = time.time()
        out_path = out_dir / f"foundation_{market}{tag}_h{h}.json"
        a = _attach(FM.panel(frames, edates, h), cache, h, panel_cols)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        preds = {sp: {m: [] for m in ORDER} for sp in ("te", "tr", "va")}
        preds["te"][HAR], preds["te"][ZS] = [], []
        yy = {"te": [], "tr": [], "va": []}
        dts = []
        n_done = 0
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            if fold_cap is not None and n_done >= fold_cap:
                break
            n_done += 1
            val_dates = np.sort(trf["date"].unique())[-VALID_LEN:]
            is_val = trf["date"].isin(val_dates)
            trf_e, vaf = trf[~is_val], trf[is_val]
            combo = pd.concat([tef, trf_e, vaf])
            n_te, n_tr = len(tef), len(trf_e)
            for m in ORDER:
                p = _predict_gbm(trf_e, combo, model_cols[m], seeds).astype(np.float32)  # float32: half the RAM
                preds["te"][m].append(p[:n_te])
                preds["tr"][m].append(p[n_te:n_te + n_tr])
                preds["va"][m].append(p[n_te + n_tr:])
            preds["te"][HAR].append(FM._har_ols(trf_e, tef).astype(np.float32))
            preds["te"][ZS].append(np.clip(tef["fnd"].to_numpy(np.float32), FL, C.PRED_CAP))
            yy["te"].append(tef["y"].to_numpy(float))
            yy["tr"].append(trf_e["y"].to_numpy(float))
            yy["va"].append(vaf["y"].to_numpy(float))
            dts.append(tef["date"].to_numpy())
            del combo, trf, tef, trf_e, vaf
            gc.collect()                                        # release the wide fold copies before the next fold
            doc = _pool_doc(h, market, seeds, preds, yy, dts, per_fold, spike)
            _checkpoint(doc, out_path)
            print(f"  h{h} fold {k} ({ts.date()}) done, {time.time()-t0:.0f}s (checkpointed)", flush=True)
        if not yy["te"]:  # pragma: no cover - defensive: a horizon with no eligible walk-forward fold
            continue
        doc = _pool_doc(h, market, seeds, preds, yy, dts, per_fold, spike)
        _checkpoint(doc, out_path)
        v, s0 = doc["verdict"], doc["stage0_zeroshot_vs_har"]
        print(f"\n== {market} h{h} (n={doc['n']:,}, {doc['n_folds']} folds) ==", flush=True)
        for m in ORDER + [HAR, ZS]:
            print(f"  {m:11s} QLIKE {doc['metrics'][m]['qlike']:.4f}", flush=True)
        print(f"  Stage0 {ZS} vs {HAR}: {s0['gain_pct']:+.2f}% (p={s0['dm_p']:.4f}) beats={s0['zeroshot_beats_har']}",
              flush=True)
        print(f"  Stage1 {FND} vs {GBME}: {v['gain_pct']:+.2f}% (p={v['dm_p']:.4f}) beats={v['beats']} | "
              f"placebo {doc['placebo']['gain_pct']:+.2f}% (p={doc['placebo']['dm_p']:.4f}) "
              f"beats={doc['placebo']['beats']} | saved {out_path.name}", flush=True)
        docs[h] = doc
    return docs


def main():  # pragma: no cover - entry driver: loads real data + full walk-forward over folds/seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("sp500", "hose"), default="hose")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed")
    ap.add_argument("--rebuild-cache", action="store_true", help="force rebuild the foundation forecast parquet")
    args = ap.parse_args()
    docs = run(args.market, smoke=args.smoke, rebuild_cache=args.rebuild_cache)
    print(f"\nPRE-REGISTERED SUCCESS (>= {C.KILL_MIN_HORIZONS} horizons beat, placebo negative, spike-robust): "
          f"{success(docs)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
