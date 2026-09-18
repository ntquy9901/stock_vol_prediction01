"""LSTM-feature / GBME-blend walk-forward driver: does a small per-fold sequential LSTM add anything the
champion own-history gamma-GBM (GBME) does not already have, on out-of-sample variance QLIKE?

This is the transferable germ of arXiv:2505.23084 (deep-sequential (+) tree-ensemble) adapted to the
variance-QLIKE thesis, run to CLOSE the direction concretely. Prior repo evidence: GNN/deep (+) GBM blends
fail with error-correlation ~0.98, and constrained stacking with diverse bases was falsified.

Per outer walk-forward fold k and horizon h:
  * lstmfeat_train = OUT-OF-FOLD LSTM predictions (inner temporal K-fold; no stacking leakage); lstmfeat_test
    = predictions from ONE LSTM trained on the full causal train window (lstm_feat.oof_train_feat / test_feat).
  * Fit GBME = GBM(gamma) on [OWN-8 | EARN] and GBME+lstmfeat on [OWN-8 | EARN | lstmfeat], seed-ensembled.
  * Also form a convex val-fit blend  w*GBME + (1-w)*LSTM  (w chosen per fold on that fold's validation QLIKE).
  * Score train / val / test (fit evidence), pool over folds, date-clustered DM (GBME+lstmfeat vs GBME primary,
    blend vs GBME secondary), verdict, plus the standalone-LSTM<->GBME error-correlation that explains a null.

Output: results/gamma_gbm/lstm_blend_<market>_h<h>.json, ONE PER HORIZON, atomic tmp+replace after every fold
(disconnect resilient). Each carries metrics/train_metrics/val_metrics (all 5) + fit_diagnostics +
learning_curves for the LEARNED model (name contains 'lstm'), so the pre-push overfit-evidence gate validates
it. HOSE additionally carries per-fold QLIKE + regime-spike robustness.

Pre-registered kill criterion (config): success iff GBME+lstmfeat beats GBME (gain>GAIN_MIN AND DM p<DM_ALPHA)
at BOTH KILL_HORIZONS; else NO-GO (expected).

Run: python run_lstm_blend.py [hose|sp500] [--smoke]
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
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(REPO / "scripts" / "quality_gate"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402  (import first so metrics/stats resolve via its path setup)
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
import overfit_check as OF  # noqa: E402
import lstm_blend_config as C  # noqa: E402
import lstm_feat as L  # noqa: E402

FL = FM.FL
BASE = "GBME"
FEAT = "GBME+lstmfeat"          # contains 'lstm' -> the overfit gate treats it as the LEARNED model
BLEND = "blend"                 # convex val-fit combination (not name-detected as learned; reported)
ORDER = [BASE, FEAT, BLEND]
LSTMFEAT = "lstmfeat"


def _own8():
    """OWN-8 own-history feature list, single-sourced from the paper_models config (FM.OWN minus rq), loaded
    by path so it does NOT register a second bare `config` module (avoids the sys.modules collision)."""
    cfg_path = REPO / "baselines" / "2026-09-13_paper_models" / "code" / "config.py"
    spec = importlib.util.spec_from_file_location("paper_models_config", cfg_path)
    pmc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmc)
    return pmc.own_set(FM.OWN)


OWN = _own8()


def _load_earn(market, edates):
    """For non-SP500 markets, replace edates with the REAL crawled VN announcement dates if present (mirrors
    the sibling drivers); SP500 keeps its own earnings from FM.load."""
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


def _gbm_ensemble(trf, combo, cols, seeds):
    """Seed-ensembled gamma-GBM: mean prediction over `seeds` (fit on trf, predict combo). Reuses FM.gbm."""
    return np.mean([FM.gbm(trf, combo, cols, s) for s in seeds], 0)


def _best_blend_weight(y_va, gbme_va, lstm_va):
    """Convex weight w in [0,1] (GBME weight) minimising validation QLIKE of w*GBME + (1-w)*LSTM. Grid search
    (BLEND_GRID points); ties resolved toward the GBME end (argmin returns the first minimiser)."""
    ws = np.linspace(0.0, 1.0, C.BLEND_GRID)
    q = [np.mean(M.per_obs_qlike(y_va, np.clip(w * gbme_va + (1 - w) * lstm_va, FL, None), floor=FL)) for w in ws]
    return float(ws[int(np.argmin(q))])


def verdict(gain_pct, dm_p):
    """A single horizon 'beats' iff the QLIKE gain is strictly positive AND the DM p-value clears DM_ALPHA."""
    return bool(gain_pct > C.GAIN_MIN * 100.0 and dm_p < C.DM_ALPHA)


def success(docs):
    """Pre-registered success: GBME+lstmfeat beats GBME at BOTH KILL_HORIZONS. Missing horizon = fail."""
    return bool(all(docs.get(h, {}).get("verdict", {}).get("beats") for h in C.KILL_HORIZONS))


def _safe_dm(err_a, err_b, dates, h):
    """Date-clustered DM, returning p=1.0 for the degenerate cases the reused DM cannot handle: numerically
    identical loss series (the expected-NO-GO case can leave the tree ignoring lstmfeat, making the two loss
    series equal), or too few surviving dates for the HLN factor."""
    degenerate = {"p_value": 1.0, "mean_diff": 0.0, "dm_hln": 0.0, "n_dates": int(np.unique(dates).size)}
    if np.allclose(err_a, err_b):
        return degenerate
    try:
        return ST.date_clustered_dm(err_a, err_b, dates, h)
    except ValueError:
        return degenerate


def _spike_mask(dates):
    """Boolean mask of test dates that fall inside any configured regime-spike window (HOSE robustness)."""
    d = pd.to_datetime(dates)
    m = np.zeros(len(d), bool)
    for lo, hi in C.SPIKE_WINDOWS:
        m |= (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    return m


def _err_corr(a, b):
    """Pearson correlation of two aligned vectors; 0.0 if either is (near-)constant (undefined correlation)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < FL or b.std() < FL:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _pool_doc(h, market, seeds, preds, yy, dts, lstm_te, weights, curves, per_fold, spike):
    """Per-horizon result doc: pooled 5-metric train/val/test for GBME / GBME+lstmfeat / blend, fit_diagnostics
    for the learned model, date-clustered DM + gain + verdict, the standalone-LSTM<->GBME error-correlation,
    plus HOSE per-fold/spike robustness."""
    y = np.concatenate(yy["te"])
    dates = np.concatenate(dts)
    pooled = {m: np.concatenate(preds["te"][m]) for m in ORDER}
    lstm_pooled = np.concatenate(lstm_te)
    err = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in ORDER}
    err_lstm = M.per_obs_qlike(y, lstm_pooled, floor=FL)
    metrics = {m: _metrics5(y, pooled[m]) for m in ORDER}
    y_tr, y_va = np.concatenate(yy["tr"]), np.concatenate(yy["va"])
    train_metrics = {m: _metrics5(y_tr, np.concatenate(preds["tr"][m])) for m in ORDER}
    val_metrics = {m: _metrics5(y_va, np.concatenate(preds["va"][m])) for m in ORDER}
    fit = {FEAT: OF.classify_fit(train_metrics[FEAT], val_metrics[FEAT], metrics[FEAT])}
    dm_feat = _safe_dm(err[FEAT], err[BASE], dates, h)
    dm_blend = _safe_dm(err[BLEND], err[BASE], dates, h)
    gain_feat = (metrics[BASE]["qlike"] - metrics[FEAT]["qlike"]) / metrics[BASE]["qlike"] * 100.0
    gain_blend = (metrics[BASE]["qlike"] - metrics[BLEND]["qlike"]) / metrics[BASE]["qlike"] * 100.0
    doc = {"market": market, "h": h, "n": int(len(y)), "n_folds": len(yy["te"]),
           "seeds": list(seeds), "lstm_seeds": list(C.LSTM_SEEDS), "inner_k": C.INNER_K, "seq_len": C.SEQ_LEN,
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit, "learning_curves": curves,
           "blend_weights": [float(w) for w in weights],
           "lstm_standalone": {"qlike": float(np.mean(err_lstm)),
                               "err_correlation_vs_gbme": _err_corr(err_lstm, err[BASE]),
                               "pred_correlation_vs_gbme": _err_corr(lstm_pooled, pooled[BASE])},
           "dm": {f"{FEAT}_vs_{BASE}": dm_feat["p_value"], f"{BLEND}_vs_{BASE}": dm_blend["p_value"]},
           "gain_pct": {f"{FEAT}_vs_{BASE}": gain_feat, f"{BLEND}_vs_{BASE}": gain_blend},
           "verdict": {"beats": verdict(gain_feat, dm_feat["p_value"]), "gain_pct": gain_feat,
                       "dm_p": dm_feat["p_value"],
                       "blend_beats": verdict(gain_blend, dm_blend["p_value"]), "blend_gain_pct": gain_blend,
                       "blend_dm_p": dm_blend["p_value"],
                       "criterion": "GBME+lstmfeat beats GBME: gain>GAIN_MIN AND DM p<DM_ALPHA"}}
    if per_fold:
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy["te"][i], preds["te"][m][i], floor=FL)))
                                     for i in range(len(yy["te"]))] for m in ORDER}
    if spike:
        sm = _spike_mask(dates)
        keep = ~sm
        if keep.any():
            dm_ex = _safe_dm(err[FEAT][keep], err[BASE][keep], dates[keep], h)
            qb, qf = float(np.mean(err[BASE][keep])), float(np.mean(err[FEAT][keep]))
            gain_ex = (qb - qf) / qb * 100.0
            doc["spike_robustness"] = {
                "n_spike_obs": int(sm.sum()), "n_ex_spike_obs": int(keep.sum()),
                "qlike_ex_spike": {BASE: qb, FEAT: qf}, "gain_pct_ex_spike": gain_ex,
                "dm_p_ex_spike": dm_ex["p_value"], "beats_ex_spike": verdict(gain_ex, dm_ex["p_value"])}
    return doc


def _checkpoint(doc, out_path):
    """Atomically write the per-horizon doc (tmp + replace) so a disconnect keeps the completed horizon."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out_path)


def run(market, load_fn=None, out_dir=None, smoke=False, trainer=None):
    """Walk-forward GBME vs GBME+lstmfeat vs blend for a market. Writes one JSON per horizon (atomic per-fold
    checkpoint) and returns {h: doc}. `trainer` injects an LSTM trainer (tests use a cheap fake)."""
    load_fn = load_fn or FM.load
    epochs = C.EPOCHS_SMOKE if smoke else C.EPOCHS
    lstm_seeds = C.LSTM_SEEDS
    gbm_seeds = (FM.SEEDS[0],) if smoke else FM.SEEDS
    horizons = C.HORIZONS_SMOKE if smoke else C.HORIZONS
    fold_cap = 1 if smoke else None
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "gamma_gbm")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""

    frames, _sect, edates = load_fn(market)
    edates = _load_earn(market, edates)
    has_earn = bool(edates)
    cols_base = OWN + (FM.EARN if has_earn else [])
    cols_feat = cols_base + [LSTMFEAT]
    per_fold = market != "sp500"
    spike = market != "sp500"
    docs = {}
    for h in horizons:
        t0 = time.time()
        out_path = out_dir / f"lstm_blend_{market}{tag}_h{h}.json"
        a = FM.panel(frames, edates, h)
        dates = a["date"].to_numpy()
        X_all, logvar_all = L.build_sequences(a, cols_base, C.SEQ_LEN)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        preds = {sp: {m: [] for m in ORDER} for sp in ("te", "tr", "va")}
        yy = {"te": [], "tr": [], "va": []}
        dts, lstm_te, weights, curves = [], [], [], {}
        n_done = 0
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trmask = (a.date >= S1.TRAIN_START) & (a.date < ts - embargo)
            temask = (a.date >= ts) & (a.date < tend)
            if temask.sum() == 0 or trmask.sum() < min_rows:
                continue
            if fold_cap is not None and n_done >= fold_cap:
                break
            n_done += 1
            pos_train = np.where(trmask.to_numpy())[0]
            pos_test = np.where(temask.to_numpy())[0]
            base_seed = S1.RNG_SEED + k
            lf_train = L.oof_train_feat(X_all, logvar_all, dates, pos_train, lstm_seeds, epochs, C.PATIENCE,
                                        base_seed, trainer=trainer)
            lf_test, cv = L.test_feat(X_all, logvar_all, dates, pos_train, pos_test, lstm_seeds, epochs,
                                      C.PATIENCE, base_seed, trainer=trainer)
            curves[f"fold{k}"] = cv
            trf = a.loc[pos_train].copy()
            tef = a.loc[pos_test].copy()
            trf[LSTMFEAT] = lf_train
            tef[LSTMFEAT] = lf_test
            val_dates = np.sort(trf["date"].unique())[-C.VALID_LEN:]
            is_val = trf["date"].isin(val_dates).to_numpy()
            trf_e, vaf = trf[~is_val], trf[is_val]
            lstm_tr, lstm_va = lf_train[~is_val], lf_train[is_val]
            combo = pd.concat([tef, trf_e, vaf])
            n_te, n_tr = len(tef), len(trf_e)
            gbme = {}
            for mdl, cols in ((BASE, cols_base), (FEAT, cols_feat)):
                p = _gbm_ensemble(trf_e, combo, cols, gbm_seeds)
                preds["te"][mdl].append(p[:n_te])
                preds["tr"][mdl].append(p[n_te:n_te + n_tr])
                preds["va"][mdl].append(p[n_te + n_tr:])
                if mdl == BASE:
                    gbme = {"te": p[:n_te], "tr": p[n_te:n_te + n_tr], "va": p[n_te + n_tr:]}
            y_va = vaf["y"].to_numpy(float)
            w = _best_blend_weight(y_va, gbme["va"], lstm_va)
            weights.append(w)
            preds["te"][BLEND].append(np.clip(w * gbme["te"] + (1 - w) * lf_test, FL, None))
            preds["tr"][BLEND].append(np.clip(w * gbme["tr"] + (1 - w) * lstm_tr, FL, None))
            preds["va"][BLEND].append(np.clip(w * gbme["va"] + (1 - w) * lstm_va, FL, None))
            yy["te"].append(tef["y"].to_numpy(float))
            yy["tr"].append(trf_e["y"].to_numpy(float))
            yy["va"].append(y_va)
            dts.append(tef["date"].to_numpy())
            lstm_te.append(lf_test)
            del trf, tef, combo
            gc.collect()
            if L.DEVICE.type == "cuda":  # pragma: no cover - GPU-only VRAM cleanup, not exercised on CPU CI
                torch.cuda.empty_cache()
            doc = _pool_doc(h, market, gbm_seeds, preds, yy, dts, lstm_te, weights, curves, per_fold, spike)
            _checkpoint(doc, out_path)
            print(f"  h{h} fold {k} ({ts.date()}) done w={w:.2f}, {time.time()-t0:.0f}s (checkpointed)",
                  flush=True)
        if not yy["te"]:  # pragma: no cover - defensive: a horizon with no eligible walk-forward fold
            continue
        doc = _pool_doc(h, market, gbm_seeds, preds, yy, dts, lstm_te, weights, curves, per_fold, spike)
        _checkpoint(doc, out_path)
        v = doc["verdict"]
        print(f"\n== {market} h{h} (n={doc['n']:,}, {doc['n_folds']} folds) ==", flush=True)
        for m in ORDER:
            print(f"  {m:16s} QLIKE {doc['metrics'][m]['qlike']:.4f}", flush=True)
        print(f"  DM {FEAT} vs {BASE}: {v['gain_pct']:+.2f}% (p={v['dm_p']:.3f}) beats={v['beats']} | "
              f"blend {v['blend_gain_pct']:+.2f}% (p={v['blend_dm_p']:.3f}) | "
              f"err-corr(LSTM,GBME)={doc['lstm_standalone']['err_correlation_vs_gbme']:.3f}", flush=True)
        print(f"  fit[{FEAT}]={doc['fit_diagnostics'][FEAT]['status']} | saved {out_path.name}", flush=True)
        docs[h] = doc
    return docs


def main():  # pragma: no cover - entry driver: loads real data + full LSTM training over folds/seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("sp500", "hose"), default="hose")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed, few epochs")
    args = ap.parse_args()
    docs = run(args.market, smoke=args.smoke)
    print(f"\nPRE-REGISTERED SUCCESS (h1 & h5 both beat GBME): {success(docs)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
