"""Walk-forward leaf-graph v2: do RF-GAP / KeRF proximity weightings beat v1's hard leaf-Hamming kNN (and the
un-smoothed XGB base, and the GBME champion) on out-of-sample HOSE QLIKE?

Per horizon h and outer fold k: fit GBME (canonical HGBR gamma), XGB (plain XGBoost gamma, capacity-matched),
and the three smoothed variants that share the XGB base — XGB+knn (v1), XGB+rfgap, XGB+rfgap+kerf — each with its
own alpha fit on a val slice and frozen for test. Score train/val/test, pool over folds, date-clustered DM
(each variant vs XGB = isolate the graph; vs GBME = vs the deployed champion; each v2 variant vs XGB+knn = did v2
beat v1), verdict + HOSE per-fold QLIKE + regime-spike robustness, and the fitted alphas per fold.

Memory: v1 accumulated every fold's smoothed TRAIN predictions and OOM'd at h22. Here train metrics are streamed
via sufficient statistics (``_Stream``) and the per-fold train arrays are freed immediately; only the smaller
val/test arrays are pooled (needed for DM). Run one horizon per process (``--horizon``) for extra isolation.

Pre-registered success (v2 vs v1): an RF-GAP(+KeRF) variant beats XGB+knn on QLIKE at >=1 horizon (gain>0 AND
date-clustered DM p<DM_ALPHA), OR is at least as good while strictly MORE spike-robust. Otherwise v1 stays the
champion version and v2 is reported as "no improvement over the simple graph."

Output: results/gamma_gbm/rfgap_<market>_h<h>.json, one per horizon, atomic checkpoint after every fold.

Run: python run_rfgap.py [hose|sp500] [--horizon H] [--smoke]
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
import rfgap_config as C  # noqa: E402
import rfgap as RG  # noqa: E402

FL = FM.FL
GBME, XGB = "GBME", "XGB"
KNN, RFGAP, KERF = "XGB+knn", "XGB+rfgap", "XGB+rfgap+kerf"
SCHEME_OF = {KNN: "knn", RFGAP: "rfgap", KERF: "rfgap_kerf"}
SMOOTHED = [KNN, RFGAP, KERF]
ORDER = [GBME, XGB] + SMOOTHED


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


class _Stream:
    """Streaming pooled-metric accumulator so the large TRAIN arrays never all live in memory at once.

    Accumulates the sufficient statistics for the 5 reported metrics (mse/rmse/mae/r2/qlike) over folds; each
    fold's arrays are folded in then dropped. ``finalize`` reproduces ``mean_squared_error`` / ``r2_score`` /
    per-obs QLIKE exactly (pooled ybar, pooled SS_tot)."""

    def __init__(self):
        self.n = 0
        self.sse = 0.0      # sum (y-p)^2
        self.sae = 0.0      # sum |y-p|
        self.sy = 0.0       # sum y
        self.syy = 0.0      # sum y^2
        self.sq = 0.0       # sum per-obs qlike

    def update(self, y, p):
        y = np.asarray(y, float)
        p = np.asarray(p, float)
        r = y - p
        self.n += len(y)
        self.sse += float(np.dot(r, r))
        self.sae += float(np.abs(r).sum())
        self.sy += float(y.sum())
        self.syy += float(np.dot(y, y))
        self.sq += float(M.per_obs_qlike(y, p, floor=FL).sum())

    def finalize(self):
        n = self.n
        mse = self.sse / n
        ss_tot = self.syy - self.sy * self.sy / n
        r2 = 1.0 - self.sse / ss_tot if ss_tot > 0 else 0.0
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(self.sae / n),
                "r2": float(r2), "qlike": float(self.sq / n)}


def _metrics5(y, p):
    """The 5 reported metrics (DirAcc dropped per paper style) computed directly from pooled arrays."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    r = y - p
    mse = float(np.dot(r, r) / len(y))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - float(np.dot(r, r)) / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.abs(r).mean()),
            "r2": float(r2), "qlike": float(np.mean(M.per_obs_qlike(y, p, floor=FL)))}


def verdict(gain_pct, dm_p):
    """A comparison 'beats' iff the QLIKE gain is strictly positive AND the DM p-value clears DM_ALPHA."""
    return bool(gain_pct > C.GAIN_MIN * 100.0 and dm_p < C.DM_ALPHA)


def success(docs):
    """Pre-registered: some v2 variant beats v1 (XGB+knn) at >=1 horizon on QLIKE+DM, OR is at least as good
    while strictly more spike-robust. Empty/missing docs => fail."""
    if not docs:
        return False
    for doc in docs.values():
        for v in C.V2_VARIANTS:
            b = doc.get("v2_vs_v1", {}).get(v, {})
            if b.get("beats_knn") or b.get("as_good_more_robust"):
                return True
    return False


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
    """All models' predictions for one fold, plus each smoothed variant's fitted alpha.

    GBME + XGB are seed-ensembled base predictions over combo=[test, train, val]. The smoothed variants reuse the
    XGB base (isolating the graph), fit their own alpha on the val slice's leaf-graph, and apply the frozen alpha
    to smooth test/train/val. Returns dicts keyed by split ('te'|'tr'|'va') -> {model: array}, and {model: alpha}."""
    combo = pd.concat([tef, trf_e, vaf])
    n_te, n_tr = len(tef), len(trf_e)
    dates = combo["date"].to_numpy()
    p_gbme = np.mean([FM.gbm(trf_e, combo, cols, s) for s in seeds], 0)
    p_xgb = RG.predict_xgb(trf_e, combo, cols, seeds)
    leaves = RG.leaf_matrix(RG.fit_booster(trf_e, cols, seeds[0]), combo[cols].to_numpy(float))

    def sl(arr, s):
        return {"te": arr[:n_te], "tr": arr[n_te:n_te + n_tr], "va": arr[n_te + n_tr:]}[s]

    out = {s: {GBME: sl(p_gbme, s), XGB: sl(p_xgb, s)} for s in ("te", "tr", "va")}
    alphas = {}
    for mdl in SMOOTHED:
        scheme = SCHEME_OF[mdl]
        alpha, _ = RG.fit_alpha(vaf["y"].to_numpy(float), sl(p_xgb, "va"), sl(leaves, "va"), sl(dates, "va"),
                                scheme, C.K_NEIGHBOURS, C.ALPHA_GRID, C.KERF_FUNC, FL)
        alphas[mdl] = float(alpha)
        for s in ("te", "tr", "va"):
            sm = RG.smooth_all(sl(p_xgb, s), sl(leaves, s), sl(dates, s), scheme, C.K_NEIGHBOURS, alpha, C.KERF_FUNC)
            out[s][mdl] = np.clip(sm, FL, C.PRED_CAP)
    return out, alphas


def _dm_block(err, dates, h, ref):
    """DM p-values + QLIKE gains of every smoothed variant vs a reference model ``ref`` (XGB / GBME / XGB+knn)."""
    dm, gain = {}, {}
    q_ref = float(np.mean(err[ref]))
    for mdl in SMOOTHED:
        if mdl == ref:
            continue
        dm[mdl] = _safe_dm(err[mdl], err[ref], dates, h)["p_value"]
        gain[mdl] = (q_ref - float(np.mean(err[mdl]))) / q_ref * 100.0
    return dm, gain


def _spike_for(err, dates, h, ref, variants):
    """Ex-spike QLIKE + DM of each ``variant`` vs ``ref`` after excluding the configured spike windows."""
    sm = _spike_mask(dates)
    keep = ~sm
    block = {"n_spike_obs": int(sm.sum()), "n_ex_spike_obs": int(keep.sum())}
    if not keep.any():
        return block, {}
    q = {m: float(np.mean(err[m][keep])) for m in [ref] + variants}
    block["qlike_ex_spike"] = q
    per = {}
    for m in variants:
        dm_ex = _safe_dm(err[m][keep], err[ref][keep], dates[keep], h)["p_value"]
        g = (q[ref] - q[m]) / q[ref] * 100.0
        per[m] = {"gain_pct_ex_spike": g, "dm_p_ex_spike": dm_ex, "beats_ex_spike": verdict(g, dm_ex)}
    return block, per


def _pool_doc(h, market, seeds, te_pred, va_pred, yy_te, yy_va, dts, train_metrics, alphas, per_fold, spike):
    """Per-horizon result doc: pooled 5-metric train(streamed)/val/test for all models, fit_diagnostics, DM of
    every smoothed variant vs XGB (isolate the graph), vs GBME (champion) and vs XGB+knn (v2-vs-v1), gains,
    verdicts, fitted alphas, plus HOSE per-fold/spike robustness and the v2-vs-v1 decision block."""
    y = np.concatenate(yy_te)
    dates = np.concatenate(dts)
    pooled = {m: np.concatenate(te_pred[m]) for m in ORDER}
    err = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in ORDER}
    metrics = {m: _metrics5(y, pooled[m]) for m in ORDER}
    y_va = np.concatenate(yy_va)
    val_metrics = {m: _metrics5(y_va, np.concatenate(va_pred[m])) for m in ORDER}
    fit = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in ORDER}
    dm_xgb, gain_xgb = _dm_block(err, dates, h, XGB)
    dm_gbme, gain_gbme = _dm_block(err, dates, h, GBME)
    dm_knn, gain_knn = _dm_block(err, dates, h, KNN)
    doc = {"market": market, "h": h, "n": int(len(y)), "n_folds": len(yy_te), "seeds": list(seeds),
           "schemes": SCHEME_OF, "kerf_func": C.KERF_FUNC,
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit,
           "alpha": {m: {"per_fold": [float(a) for a in alphas[m]], "mean": float(np.mean(alphas[m]))}
                     for m in SMOOTHED},
           "dm": {"vs_XGB": dm_xgb, "vs_GBME": dm_gbme, "vs_knn": dm_knn},
           "gain_pct": {"vs_XGB": gain_xgb, "vs_GBME": gain_gbme, "vs_knn": gain_knn},
           "verdict": {m: {"beats_xgb": verdict(gain_xgb[m], dm_xgb[m]),
                           "gain_pct_vs_xgb": gain_xgb[m], "dm_p_vs_xgb": dm_xgb[m]} for m in SMOOTHED}}
    spike_per = {}
    if spike:
        blk, spike_per = _spike_for(err, dates, h, XGB, SMOOTHED)
        doc["spike_robustness"] = {**blk, "per_variant_vs_xgb": spike_per}
    # v2-vs-v1 decision: each v2 variant vs the v1 champion XGB+knn (QLIKE+DM), and the "as good + more robust" path
    v2 = {}
    for mdl in C.V2_VARIANTS:
        g, p = gain_knn[mdl], dm_knn[mdl]
        entry = {"gain_pct_vs_knn": g, "dm_p_vs_knn": p, "beats_knn": verdict(g, p)}
        if spike and spike_per:
            qx = doc["spike_robustness"]["qlike_ex_spike"]
            entry["more_spike_robust"] = bool(qx[mdl] < qx[KNN])       # lower ex-spike QLIKE than v1
            entry["as_good_more_robust"] = bool(g >= 0.0 and qx[mdl] < qx[KNN])
        v2[mdl] = entry
    doc["v2_vs_v1"] = v2
    if per_fold:
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy_te[i], te_pred[m][i], floor=FL)))
                                     for i in range(len(yy_te))] for m in ORDER}
    return doc


def _checkpoint(doc, out_path):
    """Atomically write the per-horizon doc (tmp + replace) so a disconnect keeps the completed horizon."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out_path)


def run(market, load_fn=None, out_dir=None, smoke=False, horizons=None):
    """Walk-forward GBME vs XGB vs {knn, rfgap, rfgap+kerf} for a market. Writes one JSON per horizon; returns
    {h: doc}. Train metrics are streamed (no cross-fold train-array accumulation)."""
    load_fn = load_fn or FM.load
    seeds = (FM.SEEDS[0],) if smoke else FM.SEEDS
    horizons = horizons or (C.HORIZONS_SMOKE if smoke else C.HORIZONS)
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
        out_path = out_dir / f"rfgap_{market}{tag}_h{h}.json"
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        te_pred = {m: [] for m in ORDER}
        va_pred = {m: [] for m in ORDER}
        stream = {m: _Stream() for m in ORDER}
        yy_te, yy_va, dts = [], [], []
        alphas = {m: [] for m in SMOOTHED}
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
            fp, fold_alpha = _fold_predictions(trf_e, vaf, tef, cols, seeds)
            y_tr = trf_e["y"].to_numpy(float)
            for m in ORDER:
                te_pred[m].append(fp["te"][m])
                va_pred[m].append(fp["va"][m])
                stream[m].update(y_tr, fp["tr"][m])                 # fold train arrays streamed then dropped
            for m in SMOOTHED:
                alphas[m].append(fold_alpha[m])
            yy_te.append(tef["y"].to_numpy(float))
            yy_va.append(vaf["y"].to_numpy(float))
            dts.append(tef["date"].to_numpy())
            del fp
            train_metrics = {m: stream[m].finalize() for m in ORDER}
            doc = _pool_doc(h, market, seeds, te_pred, va_pred, yy_te, yy_va, dts, train_metrics,
                            alphas, per_fold, spike)
            _checkpoint(doc, out_path)
            amsg = " ".join(f"{SCHEME_OF[m]}={fold_alpha[m]:.2f}" for m in SMOOTHED)
            print(f"  h{h} fold {k} ({ts.date()}) {amsg} done, {time.time()-t0:.0f}s (checkpointed)", flush=True)
        if not yy_te:  # pragma: no cover - defensive: a horizon with no eligible walk-forward fold
            continue
        train_metrics = {m: stream[m].finalize() for m in ORDER}
        doc = _pool_doc(h, market, seeds, te_pred, va_pred, yy_te, yy_va, dts, train_metrics, alphas,
                        per_fold, spike)
        _checkpoint(doc, out_path)
        print(f"\n== {market} h{h} (n={doc['n']:,}, {doc['n_folds']} folds) ==", flush=True)
        for m in ORDER:
            print(f"  {m:16s} QLIKE {doc['metrics'][m]['qlike']:.4f}", flush=True)
        for m in SMOOTHED:
            print(f"  {m:16s} vs XGB {doc['gain_pct']['vs_XGB'][m]:+.3f}% (p={doc['dm']['vs_XGB'][m]:.4f}) "
                  f"beats_xgb={doc['verdict'][m]['beats_xgb']}", flush=True)
        for m in C.V2_VARIANTS:
            b = doc["v2_vs_v1"][m]
            print(f"  {m:16s} vs knn {b['gain_pct_vs_knn']:+.3f}% (p={b['dm_p_vs_knn']:.4f}) "
                  f"beats_knn={b['beats_knn']}", flush=True)
        print(f"  saved {out_path.name}", flush=True)
        docs[h] = doc
    return docs


def main():  # pragma: no cover - entry driver: loads real data + full walk-forward over folds/seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("sp500", "hose"), default="hose")
    ap.add_argument("--horizon", type=int, default=None, help="run a single horizon in a fresh process")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed")
    args = ap.parse_args()
    hz = (args.horizon,) if args.horizon else None
    docs = run(args.market, smoke=args.smoke, horizons=hz)
    print(f"\nPRE-REGISTERED SUCCESS (a v2 variant beats/robustifies v1 XGB+knn): {success(docs)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
