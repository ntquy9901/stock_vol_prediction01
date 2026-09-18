"""Walk-forward falsification (LITERAL Option A): does smoothing the champion gamma-GBME's OWN predictions over
a graph built from the GBME's OWN trees (leaf-index cooccurrence) beat the un-smoothed GBME on out-of-sample
QLIKE? No XGBoost proxy — the leaf-vectors are extracted from the fitted HGBR itself (``hgbr_leaf``).

Per horizon h and outer fold k: fit GBME (canonical HGBR gamma) on the causal train window (minus a val slice),
seed-ensemble its predictions, extract its seed[0] leaf-matrix (with a fail-loud reconstruction guard), fit the
per-day leaf-graph smoothing weight ``alpha`` on the val slice, and apply the frozen ``alpha`` to smooth
test/train/val. Score train/val/test, pool over folds, date-clustered DM (GBME+leafgraph vs GBME), verdict +
HOSE per-fold QLIKE + regime-spike robustness, and the fitted alpha per fold.

Pre-registered kill criterion: GBME+leafgraph beats GBME at BOTH h1 and h5 (QLIKE gain>0 AND date-clustered
DM p<0.05). Prior (XGB proxy): small positive on HOSE, ~0 on SP500 (alpha -> 0).

Output: results/gamma_gbm/gbme_lg_<market>_h<h>.json, one per horizon, atomic checkpoint after every fold.

Run: python run_gbme_leafgraph.py [hose|sp500] [--smoke]
"""
from __future__ import annotations

import argparse
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
import gbme_lg_config as C  # noqa: E402
import leaf_graph as LG  # noqa: E402
import hgbr_leaf as HL  # noqa: E402

FL = FM.FL
GBME, GBMELG = "GBME", "GBME+leafgraph"
ORDER = [GBME, GBMELG]


def _own8():
    """OWN-8 own-history feature list, single-sourced from the paper_models config, loaded by path so it does
    NOT register a second bare ``config`` module (avoids the sys.modules collision)."""
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


def _metrics5(y, p):
    """The 5 reported metrics (DirAcc dropped per paper style)."""
    return {"mse": float(mean_squared_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
            "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}


def verdict(gain_pct, dm_p):
    """A horizon 'beats' iff the QLIKE gain over GBME is strictly positive AND the DM p-value clears DM_ALPHA."""
    return bool(gain_pct > C.GAIN_MIN * 100.0 and dm_p < C.DM_ALPHA)


def success(docs):
    """Pre-registered: GBME+leafgraph beats GBME at BOTH KILL_HORIZONS. Missing horizon = fail."""
    return bool(all(docs.get(h, {}).get("verdict", {}).get("beats") for h in C.KILL_HORIZONS))


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


def _fold_predictions(trf_e, vaf, tef, cols, seeds):
    """GBME and GBME+leafgraph predictions for one fold, plus the fold's fitted alpha.

    GBME = seed-ensembled base over combo=[test, train, val]. The leaf-matrix is extracted from seed[0]'s FITTED
    HGBR (real trees, reconstruction-guarded). alpha is fit on the val slice's leaf-graph and applied (frozen)
    to smooth test/train/val. Returns dicts keyed by split ('te'|'tr'|'va') -> {model: array}, and alpha."""
    combo = pd.concat([tef, trf_e, vaf])
    n_te, n_tr = len(tef), len(trf_e)
    dates = combo["date"].to_numpy()
    X = combo[cols].to_numpy(float)
    models = [HL.fit_gbme(trf_e, cols, s) for s in seeds]
    p_gbme = np.mean([HL.predict_gbme(m, X) for m in models], 0)
    leaves = HL.leaf_matrix(models[0], X)                          # seed[0] trees + reconstruction guard

    def sl(arr, s):
        return {"te": arr[:n_te], "tr": arr[n_te:n_te + n_tr], "va": arr[n_te + n_tr:]}[s]

    alpha, _ = LG.fit_alpha(vaf["y"].to_numpy(float), sl(p_gbme, "va"), sl(leaves, "va"), sl(dates, "va"),
                            C.K_NEIGHBOURS, C.ALPHA_GRID, FL)
    out = {s: {} for s in ("te", "tr", "va")}
    for s in ("te", "tr", "va"):
        base = sl(p_gbme, s)
        out[s][GBME] = base
        smoothed = LG.smooth_all(base, sl(leaves, s), sl(dates, s), C.K_NEIGHBOURS, alpha)
        # Floor identically to the base (which is np.maximum(pred, FL)) so at alpha==0 the smoothed arm is
        # EXACTLY the GBME arm (no asymmetric upper cap): the alpha->0 NO-GO shows a strictly-zero gain
        # attributable only to the graph, not to a clipping difference. The smoothed value is a convex combo
        # of >=FL gamma predictions, so no upper guard is needed (HGBR predict is bounded, unlike an XGB exp-link).
        out[s][GBMELG] = np.maximum(smoothed, FL)
    return out, float(alpha)


def _pool_doc(h, market, seeds, preds, yy, dts, alphas, per_fold, spike):
    """Per-horizon result doc: pooled 5-metric train/val/test for both models, fit_diagnostics, date-clustered
    DM (GBME+leafgraph vs GBME), gain, verdict, fitted alpha, plus HOSE per-fold/spike robustness."""
    y = np.concatenate(yy["te"])
    dates = np.concatenate(dts)
    pooled = {m: np.concatenate(preds["te"][m]) for m in ORDER}
    err = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in ORDER}
    metrics = {m: _metrics5(y, pooled[m]) for m in ORDER}
    y_tr, y_va = np.concatenate(yy["tr"]), np.concatenate(yy["va"])
    train_metrics = {m: _metrics5(y_tr, np.concatenate(preds["tr"][m])) for m in ORDER}
    val_metrics = {m: _metrics5(y_va, np.concatenate(preds["va"][m])) for m in ORDER}
    fit = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in ORDER}
    dm_gbme = _safe_dm(err[GBMELG], err[GBME], dates, h)
    gain = (metrics[GBME]["qlike"] - metrics[GBMELG]["qlike"]) / metrics[GBME]["qlike"] * 100.0
    doc = {"market": market, "h": h, "n": int(len(y)), "n_folds": len(yy["te"]), "seeds": list(seeds),
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit,
           "alpha": {"per_fold": [float(a) for a in alphas], "mean": float(np.mean(alphas))},
           "dm": {f"{GBMELG}_vs_{GBME}": dm_gbme["p_value"]},
           "gain_pct": {f"{GBMELG}_vs_{GBME}": gain},
           "verdict": {"beats": verdict(gain, dm_gbme["p_value"]), "gain_pct": gain,
                       "dm_p": dm_gbme["p_value"],
                       "criterion": "GBME+leafgraph beats GBME: gain>GAIN_MIN AND DM p<DM_ALPHA"}}
    if per_fold:
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy["te"][i], preds["te"][m][i], floor=FL)))
                                     for i in range(len(yy["te"]))] for m in ORDER}
    if spike:
        sm = _spike_mask(dates)
        keep = ~sm
        if keep.any():
            dm_ex = _safe_dm(err[GBMELG][keep], err[GBME][keep], dates[keep], h)
            qb, qz = float(np.mean(err[GBME][keep])), float(np.mean(err[GBMELG][keep]))
            doc["spike_robustness"] = {
                "n_spike_obs": int(sm.sum()), "n_ex_spike_obs": int(keep.sum()),
                "qlike_ex_spike": {GBME: qb, GBMELG: qz}, "gain_pct_ex_spike": (qb - qz) / qb * 100.0,
                "dm_p_ex_spike": dm_ex["p_value"], "beats_ex_spike": verdict((qb - qz) / qb * 100.0,
                                                                             dm_ex["p_value"])}
    return doc


def _checkpoint(doc, out_path):
    """Atomically write the per-horizon doc (tmp + replace) so a disconnect keeps the completed horizon."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out_path)


def run(market, load_fn=None, out_dir=None, smoke=False):
    """Walk-forward GBME vs GBME+leafgraph for a market. Writes one JSON per horizon and returns {h: doc}."""
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
    has_earn = bool(edates)
    cols = OWN + (FM.EARN if has_earn else [])
    per_fold = market != "sp500"
    spike = market != "sp500"
    docs = {}
    for h in horizons:
        t0 = time.time()
        out_path = out_dir / f"gbme_lg_{market}{tag}_h{h}.json"
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        preds = {sp: {m: [] for m in ORDER} for sp in ("te", "tr", "va")}
        yy = {"te": [], "tr": [], "va": []}
        dts, alphas = [], []
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
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            is_val = trf["date"].isin(val_dates)
            trf_e, vaf = trf[~is_val], trf[is_val]
            fp, alpha = _fold_predictions(trf_e, vaf, tef, cols, seeds)
            for s in ("te", "tr", "va"):
                for m in ORDER:
                    preds[s][m].append(fp[s][m])
            yy["te"].append(tef["y"].to_numpy(float))
            yy["tr"].append(trf_e["y"].to_numpy(float))
            yy["va"].append(vaf["y"].to_numpy(float))
            dts.append(tef["date"].to_numpy())
            alphas.append(alpha)
            del fp
            doc = _pool_doc(h, market, seeds, preds, yy, dts, alphas, per_fold, spike)
            _checkpoint(doc, out_path)
            print(f"  h{h} fold {k} ({ts.date()}) alpha={alpha:.2f} done, {time.time()-t0:.0f}s (checkpointed)",
                  flush=True)
        if not yy["te"]:  # pragma: no cover - defensive: a horizon with no eligible walk-forward fold
            continue
        doc = _pool_doc(h, market, seeds, preds, yy, dts, alphas, per_fold, spike)
        _checkpoint(doc, out_path)
        v = doc["verdict"]
        print(f"\n== {market} h{h} (n={doc['n']:,}, {doc['n_folds']} folds, alpha_mean="
              f"{doc['alpha']['mean']:.2f}) ==", flush=True)
        for m in ORDER:
            print(f"  {m:16s} QLIKE {doc['metrics'][m]['qlike']:.4f}", flush=True)
        print(f"  DM {GBMELG} vs {GBME}: {v['gain_pct']:+.2f}% (p={v['dm_p']:.4f}) beats={v['beats']} | "
              f"saved {out_path.name}", flush=True)
        docs[h] = doc
    return docs


def main():  # pragma: no cover - entry driver: loads real data + full walk-forward over folds/seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("sp500", "hose"), default="hose")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed")
    args = ap.parse_args()
    docs = run(args.market, smoke=args.smoke)
    print(f"\nPRE-REGISTERED SUCCESS (h1 & h5 both beat GBME): {success(docs)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
