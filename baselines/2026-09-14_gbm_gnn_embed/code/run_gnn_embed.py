"""GBME+GNN-embed walk-forward driver: does a learned 2-layer GNN node embedding, concatenated as features
into the champion gamma-GBM, beat GBM+earn (GBME) on out-of-sample QLIKE?

Per outer walk-forward fold k and horizon h:
  * z_train = OUT-OF-FOLD embeddings (inner temporal K-fold; no stacking leakage), z_test = embeddings from a
    GNN trained on the full train window (embed.oof_train_z / embed.test_z).
  * Fit GBME = GBM(gamma) on [OWN-8 | EARN-4] and GBME+GNN-embed on [OWN-8 | EARN-4 | z], seed-ensembled.
  * Score train / val / test (fit evidence), pool over folds, date-clustered DM (GBME+z vs GBME), verdict.

Output: results/gamma_gbm/gnn_embed_<market>_h<h>.json, ONE PER HORIZON, atomic tmp+replace after every fold
(Colab/disconnect resilient). Each carries metrics/train_metrics/val_metrics (all 5) + fit_diagnostics +
learning_curves for the learned model, so the pre-push overfit-evidence gate validates it. HOSE additionally
carries per-fold QLIKE + regime-spike robustness.

Pre-registered kill criterion (config): success iff GBME+z beats GBME (gain>GAIN_MIN AND DM p<DM_ALPHA) at
BOTH h1 and h5; else NO-GO (expected).

Run: python run_gnn_embed.py [sp500|hose] [--smoke]
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

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(REPO / "scripts" / "quality_gate"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402  (import first so metrics/stats resolve)
import gnnhar_sp500 as G  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
import overfit_check as OF  # noqa: E402
import gnn_embed_config as C  # noqa: E402
import embed as E  # noqa: E402

FL = FM.FL
BASE = "GBME"
LEARNED = "GBME+GNN-embed"          # contains 'gnn' -> overfit gate treats it as a learned model


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
    full_compare); SP500 keeps its own earnings from FM.load."""
    if market == "sp500":
        return edates
    ep = REPO / "results" / "gamma_gbm" / "hose_earnings_combined.parquet"
    if ep.exists():
        e = pd.read_parquet(ep)
        return {tk: np.sort(g["earnings_date"].to_numpy()) for tk, g in e.groupby("ticker")}
    return edates


def _with_z(df, z, zcols):
    """Return a copy of `df` with the embedding columns `zcols` set to the rows of `z` (aligned by order)."""
    out = df.copy()
    for j, col in enumerate(zcols):
        out[col] = z[:, j]
    return out


def _gbm_ensemble(trf, combo, cols, seeds):
    """Seed-ensembled gamma-GBM: mean prediction over `seeds` (fit on trf, predict combo). Reuses FM.gbm."""
    return np.mean([FM.gbm(trf, combo, cols, s) for s in seeds], 0)


def verdict(gain_pct, dm_p):
    """A single horizon 'beats' iff the QLIKE gain is strictly positive AND the DM p-value clears DM_ALPHA."""
    return bool(gain_pct > C.GAIN_MIN * 100.0 and dm_p < C.DM_ALPHA)


def success(docs):
    """Pre-registered success: GBME+z beats GBME at BOTH KILL_HORIZONS (h1 and h5). Missing horizon = fail."""
    return bool(all(docs.get(h, {}).get("verdict", {}).get("beats") for h in C.KILL_HORIZONS))


def _safe_dm(err_a, err_b, dates, h):
    """Date-clustered DM, but return p=1.0 (no detectable difference) for the degenerate cases the reused DM
    cannot handle: numerically identical loss series (the expected-NO-GO case can leave the tree ignoring
    `z`, making GBME+z == GBME), a (near-)constant date-level differential (zero long-run variance), or too
    few surviving dates for the HLN factor (h >= n_dates, e.g. a spike-excluded long-horizon fold)."""
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


def _pool_doc(h, market, emb_seeds, gbm_seeds, preds, yy, dts, curves, per_fold, spike):
    """Build the per-horizon result doc: pooled 5-metric train/val/test for both models, fit_diagnostics for
    the learned model, date-clustered DM + gain + verdict, plus HOSE per-fold/spike robustness."""
    order = [BASE, LEARNED]
    y = np.concatenate(yy["te"])
    dates = np.concatenate(dts)
    pooled = {m: np.concatenate(preds["te"][m]) for m in order}
    err = {m: M.per_obs_qlike(y, pooled[m], floor=FL) for m in order}
    metrics = {m: G._metrics5(y, pooled[m]) for m in order}
    y_tr, y_va = np.concatenate(yy["tr"]), np.concatenate(yy["va"])
    train_metrics = {m: G._metrics5(y_tr, np.concatenate(preds["tr"][m])) for m in order}
    val_metrics = {m: G._metrics5(y_va, np.concatenate(preds["va"][m])) for m in order}
    fit = {LEARNED: OF.classify_fit(train_metrics[LEARNED], val_metrics[LEARNED], metrics[LEARNED])}
    dm = _safe_dm(err[LEARNED], err[BASE], dates, h)
    gain = (metrics[BASE]["qlike"] - metrics[LEARNED]["qlike"]) / metrics[BASE]["qlike"] * 100.0
    beats = verdict(gain, dm["p_value"])
    doc = {"market": market, "h": h, "n": int(len(y)), "n_folds": len(yy["te"]),
           "emb_seeds": list(emb_seeds), "gbm_seeds": list(gbm_seeds), "inner_k": C.INNER_K,
           "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
           "fit_diagnostics": fit,
           "learning_curves": curves,
           "dm": {f"{LEARNED}_vs_{BASE}": dm["p_value"]},
           "gain_pct": {f"{LEARNED}_vs_{BASE}": gain},
           "verdict": {"beats": beats, "gain_pct": gain, "dm_p": dm["p_value"],
                       "criterion": "gain>GAIN_MIN AND DM p<DM_ALPHA"}}
    if per_fold:
        doc["per_fold_qlike"] = {m: [float(np.mean(M.per_obs_qlike(yy["te"][i], preds["te"][m][i], floor=FL)))
                                     for i in range(len(yy["te"]))] for m in order}
    if spike:
        sm = _spike_mask(dates)
        keep = ~sm
        if keep.any():
            dm_ex = _safe_dm(err[LEARNED][keep], err[BASE][keep], dates[keep], h)
            qb, qz = float(np.mean(err[BASE][keep])), float(np.mean(err[LEARNED][keep]))
            gain_ex = (qb - qz) / qb * 100.0
            doc["spike_robustness"] = {
                "n_spike_obs": int(sm.sum()), "n_ex_spike_obs": int(keep.sum()),
                "qlike_ex_spike": {BASE: qb, LEARNED: qz}, "gain_pct_ex_spike": gain_ex,
                "dm_p_ex_spike": dm_ex["p_value"], "beats_ex_spike": verdict(gain_ex, dm_ex["p_value"])}
    return doc


def _checkpoint(doc, out_path):
    """Atomically write the per-horizon doc (tmp + replace) so a disconnect keeps the completed horizon."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(out_path)


def run(market, load_fn=None, out_dir=None, smoke=False, trainer=None, frozen=False):
    """Walk-forward GBME vs GBME+GNN-embed for a market. Writes one JSON per horizon (atomic per-fold
    checkpoint) and returns {h: doc}. `trainer` injects a GNN embedder (tests use a cheap fake).

    `frozen=True` selects the fixed-basis variant: instead of per-fold OOF/test embeddings (which live in
    different, ill-posed latent bases and made the long-horizon QLIKE detonate), ONE GNN is trained on the
    burn-in window (the first eligible fold's causal train window, before every test fold) and frozen; its
    single-basis embeddings are reused for every fold's train and test rows. Output tag `_frozen`."""
    load_fn = load_fn or FM.load
    epochs = C.EPOCHS_SMOKE if smoke else C.EPOCHS
    # ONE embedding GNN seed (not a seed-ensemble): neural hidden units are only defined up to a
    # permutation/rotation/sign, so averaging embeddings across differently-initialised seeds shrinks the
    # representation toward its mean in an ill-defined basis. The GBM is still seed-ensembled (gbm_seeds).
    emb_seeds = (G.SEEDS[0],)
    gbm_seeds = (FM.SEEDS[0],) if smoke else FM.SEEDS
    horizons = (C.HORIZONS[0],) if smoke else C.HORIZONS
    fold_cap = 1 if smoke else None
    min_rows = C.MIN_ROWS.get(market, C.MIN_ROWS["default"])
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "gamma_gbm")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ("_frozen" if frozen else "") + ("_smoke" if smoke else "")

    frames, sect, edates = load_fn(market)
    edates = _load_earn(market, edates)
    has_earn = bool(edates)
    cols_base = OWN + (FM.EARN if has_earn else [])
    zcols = [f"z{j}" for j in range(G.N_HID)]
    cols_z = cols_base + zcols
    per_fold = market != "sp500"
    spike = market != "sp500"
    docs = {}
    for h in horizons:
        t0 = time.time()
        out_path = out_dir / f"gnn_embed_{market}{tag}_h{h}.json"
        a = FM.panel(frames, edates, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        tickers = sorted(a["ticker"].unique())
        keep = list(dict.fromkeys(OWN + (FM.EARN if has_earn else []) +
                                  ["date", "ticker", "y", "logpk", "parkinson_variance"]))
        preds = {sp: {BASE: [], LEARNED: []} for sp in ("te", "tr", "va")}
        yy = {"te": [], "tr": [], "va": []}
        dts, curves = [], {}
        n_done = 0
        zmap = None                                       # frozen-basis: filled once at the first eligible fold
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            tef = a[(a.date >= ts) & (a.date < tend)]
            if len(tef) == 0 or len(trf) < min_rows:
                continue
            if fold_cap is not None and n_done >= fold_cap:
                break
            n_done += 1
            fold = None
            base_seed = S1.RNG_SEED + k
            if frozen:
                if zmap is None:
                    # burn-in = the first eligible fold's causal train window (before every test fold); one
                    # GNN, frozen, embeds the whole panel in a single shared basis.
                    burnin_dates = np.sort(trf["date"].unique())
                    zmap, curves["frozen"] = E.frozen_z(a[keep], tickers, OWN, burnin_dates, emb_seeds,
                                                        epochs, C.PATIENCE, S1.RNG_SEED, trainer=trainer)
                z_train = np.asarray([zmap[int(ix)] for ix in trf.index.to_numpy()], np.float32)
                z_test = np.asarray([zmap[int(ix)] for ix in tef.index.to_numpy()], np.float32)
            else:
                fold = a.loc[(a.date >= S1.TRAIN_START) & (a.date < tend), keep].copy()
                z_train = E.oof_train_z(trf[keep], tickers, OWN, emb_seeds, epochs, C.PATIENCE, base_seed,
                                        trainer=trainer)
                z_test, cv = E.test_z(fold, trf[keep], tef[keep], tickers, OWN, emb_seeds, epochs,
                                      C.PATIENCE, base_seed, trainer=trainer)
                curves[f"fold{k}"] = cv
            trf_z = _with_z(trf, z_train, zcols)
            tef_z = _with_z(tef, z_test, zcols)
            val_dates = np.sort(trf_z["date"].unique())[-E._val_len(trf_z["date"].nunique()):]
            is_val = trf_z["date"].isin(val_dates)
            trf_e, vaf = trf_z[~is_val], trf_z[is_val]
            combo = pd.concat([tef_z, trf_e, vaf])
            n_te, n_tr = len(tef_z), len(trf_e)
            # Fit on trf_e (train MINUS the val slice) so val_metrics is a TRUE hold-out, not in-sample:
            # the gate-required fit_diagnostics then compares a genuine val->test gap. GBM has no early stop,
            # so dropping the ~VALID_LEN val dates costs negligible train data on a multi-year window.
            for mdl, cols in ((BASE, cols_base), (LEARNED, cols_z)):
                p = _gbm_ensemble(trf_e, combo, cols, gbm_seeds)
                preds["te"][mdl].append(p[:n_te])
                preds["tr"][mdl].append(p[n_te:n_te + n_tr])
                preds["va"][mdl].append(p[n_te + n_tr:])
            yy["te"].append(tef_z["y"].to_numpy(float))
            yy["tr"].append(trf_e["y"].to_numpy(float))
            yy["va"].append(vaf["y"].to_numpy(float))
            dts.append(tef_z["date"].to_numpy())
            del fold, trf_z, tef_z, combo
            gc.collect()
            if G.DEVICE.type == "cuda":  # pragma: no cover - GPU-only VRAM cleanup, not exercised on CPU CI
                import torch
                torch.cuda.empty_cache()
            doc = _pool_doc(h, market, emb_seeds, gbm_seeds, preds, yy, dts, curves, per_fold, spike)
            _checkpoint(doc, out_path)
            print(f"  h{h} fold {k} ({ts.date()}) done, {time.time()-t0:.0f}s (checkpointed)", flush=True)
        if not yy["te"]:  # pragma: no cover - defensive: a horizon with no eligible walk-forward fold
            continue
        doc = _pool_doc(h, market, emb_seeds, gbm_seeds, preds, yy, dts, curves, per_fold, spike)
        _checkpoint(doc, out_path)
        v = doc["verdict"]
        print(f"\n== {market} h{h} (n={doc['n']:,}, {doc['n_folds']} folds) ==", flush=True)
        for m in (BASE, LEARNED):
            print(f"  {m:16s} QLIKE {doc['metrics'][m]['qlike']:.4f}", flush=True)
        print(f"  DM {LEARNED} vs {BASE}: {v['gain_pct']:+.2f}% (p={v['dm_p']:.3f}) beats={v['beats']}",
              flush=True)
        print(f"  fit[{LEARNED}]={doc['fit_diagnostics'][LEARNED]['status']} | saved {out_path.name}",
              flush=True)
        docs[h] = doc
    return docs


def main():  # pragma: no cover - entry driver: loads real data + full training loop over folds/seeds
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("sp500", "hose"), default="hose")
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed, few epochs")
    ap.add_argument("--frozen", action="store_true", help="fixed-basis variant: one burn-in GNN, no per-fold refit")
    args = ap.parse_args()
    docs = run(args.market, smoke=args.smoke, frozen=args.frozen)
    print(f"\nPRE-REGISTERED SUCCESS (h1 & h5 both beat): {success(docs)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
