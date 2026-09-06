"""HAR-X + XGBoost residual-ratio (and direct XGBoost) walk-forward study on enriched VN30/VN100.

Reuses the delivered panel/fold/HAR-X/metric/DM machinery READ-ONLY (wf_enriched_panel, wf_folds,
run_walkforward._har_ols_preds, run_masked_rich metric/DM helpers, edge_hmatched.directed_vol2pk_hmatched)
and adds only the causal feature table (xgb_features), the chronological OOF HAR-X residual target
(xgb_oof), the XGBoost tuning (xgb_model) and the pooled diagnostics (xgb_eval). CPU-only
(``tree_method='hist'``, bounded ``n_jobs``, one deterministic seed) -- the GPU is left for the concurrent
transformer agent.

Canonical split: lb10, folds_target=7 (matches the delivered edge_hmatched runs, so HAR-X/VolGA/LSTM
pooled numbers are directly comparable).

Smoke: .venv_gpu_encode/Scripts/python.exe .../run_xgboost_residual.py --market vn30 --horizon 1 --smoke
Full:  .venv_gpu_encode/Scripts/python.exe .../run_xgboost_residual.py --market vn30 --horizon 1
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-09-05_edge_horizon_matched" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import _har_ols_preds, training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_edge_hmatched import directed_vol2pk_hmatched  # noqa: E402

import xgb_config as C  # noqa: E402
import xgb_eval as E  # noqa: E402
from xgb_features import build_graph_features, build_sm_features, read_aligned_columns  # noqa: E402
from xgb_model import tune_direct, tune_residual, _floored  # noqa: E402
from xgb_oof import oof_harx, reconstruct, residual_target  # noqa: E402

_MODELS = ("XGB_direct", "XGB_resid_base", "XGB_resid_market", "XGB_resid_graph")
_RESID = ("XGB_resid_base", "XGB_resid_market", "XGB_resid_graph")


def feature_column_indices(groups, n_graph):
    """Column-index subsets into the concatenated ``[stock..., market..., graph...]`` feature tensor.

    ``resid_base`` = stock only; ``resid_market`` adds the cross-sectional market columns; ``resid_graph``
    adds the neighbour columns; ``direct`` (direct XGBoost) uses stock+market."""
    sm = len(groups)
    stock = [k for k, g in enumerate(groups) if g == "stock"]
    market = [k for k, g in enumerate(groups) if g == "market"]
    graph = list(range(sm, sm + n_graph))
    return {"XGB_direct": stock + market, "XGB_resid_base": stock,
            "XGB_resid_market": stock + market, "XGB_resid_graph": stock + market + graph}


def design_rows(Fall, anchors, mask, cols):
    """Flatten the valid (mask) ``(anchor, node)`` cells of ``Fall[anchors][:,:,cols]`` to ``[rows, len(cols)]``."""
    return Fall[anchors][:, :, cols][mask]


def place_rows(rows, mask, shape):
    """Scatter ``rows`` back to a dense ``shape`` array at the ``mask`` cells (0 elsewhere)."""
    out = np.zeros(shape, dtype=float)
    out[mask] = rows
    return out


def lock_keys_fold(y_te, zrf_tgt, mask, dates, mult, floor):
    """Limit-lock target cells this fold: ``(node, date)`` where target <= mult*floor OR zero_range_flag."""
    keys = set()
    a, n = y_te.shape
    for i in range(a):
        for j in range(n):
            if mask[i, j] and (y_te[i, j] <= mult * floor or zrf_tgt[i, j] >= 0.5):
                keys.add((j, dates[i]))
    return keys


def _agg(dicts):
    """Aggregate per-fold split-metric dicts -> mean mse/qlike/r2, summed n (walk-forward fit evidence)."""
    keys = ("mse", "qlike", "r2")
    agg = {k: float(np.mean([d[k] for d in dicts])) for k in keys}
    agg["n"] = int(sum(d["n"] for d in dicts))
    return agg


def _nf_rows(nfloor, mask):
    """Per-row positivity floor for the valid cells of ``mask`` (broadcast the per-node [N] floor)."""
    return np.broadcast_to(nfloor, mask.shape)[mask]


def process_fold(panel, fold, wf, Fsm, groups, ret, zrf, fl):
    """Train HAR/HAR-X + the 4 XGBoost models on one fold; return pooled test dicts, fit metrics,
    per-fold lock keys and the selection record."""
    D = pack_fold(panel, fold, wf.lookback, wf.horizon)
    n = D.N
    nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS         # [N] shared per-node positivity floor
    har, harx = _har_ols_preds(D, fl, nfloor)
    last_tr_row = int(panel.anchors[fold.train][-1]) + wf.horizon    # train-only edge cutoff
    adj = directed_vol2pk_hmatched(panel.feats[:, :, 4], np.sqrt(panel.pk), last_tr_row, wf.horizon, C.EDGE_TOP_K)
    G, _ = build_graph_features(panel, ret, adj)
    Fall = np.concatenate([Fsm, G], axis=2)
    cols = feature_column_indices(groups, G.shape[2])
    aa_tr, aa_va, aa_te = panel.anchors[fold.train], panel.anchors[fold.val], panel.anchors[fold.forecast]
    mtr, mva, mte = D.tmask_tr.astype(bool), D.tmask_va.astype(bool), D.tmask_te.astype(bool)

    oof = oof_harx(D.har5_tr, D.y_tr, mtr, fl, nfloor)               # chronological OOF HAR-X (train rows)
    z_full = residual_target(D.y_tr, oof)
    resid_mask = mtr & np.isfinite(oof) & np.isfinite(z_full)
    zva_full = residual_target(D.y_va, harx["va"])                   # val early-stop residual target

    pooled_te = {}
    tr_m, va_m, te_m = {}, {}, {}
    select = {}

    # HAR / HAR-X (deterministic references, pooled test)
    pooled_te["HAR"] = RMR._pred_dict(har["te"], D.y_te, D.tmask_te, D.d_te, n)
    pooled_te["HAR-X"] = RMR._pred_dict(harx["te"], D.y_te, D.tmask_te, D.d_te, n)

    # direct XGBoost on the log target
    dcols = cols["XGB_direct"]
    xtr, xva, xte = (design_rows(Fall, aa_tr, mtr, dcols), design_rows(Fall, aa_va, mva, dcols),
                     design_rows(Fall, aa_te, mte, dcols))
    bd = tune_direct(xtr, D.y_tr[mtr], xva, D.y_va[mva], _nf_rows(nfloor, mtr), _nf_rows(nfloor, mva), fl)
    p_tr = _floored(np.exp(bd["model"].predict(xtr)), fl, _nf_rows(nfloor, mtr))
    p_va = _floored(np.exp(bd["model"].predict(xva)), fl, _nf_rows(nfloor, mva))
    p_te = _floored(np.exp(bd["model"].predict(xte)), fl, _nf_rows(nfloor, mte))
    tr_m["XGB_direct"] = E.fit_metrics(D.y_tr[mtr], p_tr, fl)
    va_m["XGB_direct"] = E.fit_metrics(D.y_va[mva], p_va, fl)
    te_m["XGB_direct"] = E.fit_metrics(D.y_te[mte], p_te, fl)
    pooled_te["XGB_direct"] = RMR._pred_dict(place_rows(p_te, mte, D.y_te.shape), D.y_te, D.tmask_te, D.d_te, n)
    select["XGB_direct"] = {"params": bd["params"], "best_iteration": bd["best_iteration"],
                            "val_qlike": bd["val_qlike"]}

    # HAR-X + XGBoost residual-ratio ladder
    for name in _RESID:
        cc = cols[name]
        xtr = design_rows(Fall, aa_tr, resid_mask, cc)
        xva = design_rows(Fall, aa_va, mva, cc)
        xte = design_rows(Fall, aa_te, mte, cc)
        br = tune_residual(xtr, z_full[resid_mask], xva, zva_full[mva], harx["va"][mva],
                           D.y_va[mva], _nf_rows(nfloor, mva), fl)
        a_sh, cl = br["alpha"], br["clip"]
        p_tr = _floored(reconstruct(harx["tr"][resid_mask], br["model"].predict(xtr), a_sh, cl),
                        fl, _nf_rows(nfloor, resid_mask))
        p_va = _floored(reconstruct(harx["va"][mva], br["model"].predict(xva), a_sh, cl), fl, _nf_rows(nfloor, mva))
        p_te = _floored(reconstruct(harx["te"][mte], br["model"].predict(xte), a_sh, cl), fl, _nf_rows(nfloor, mte))
        tr_m[name] = E.fit_metrics(D.y_tr[resid_mask], p_tr, fl)
        va_m[name] = E.fit_metrics(D.y_va[mva], p_va, fl)
        te_m[name] = E.fit_metrics(D.y_te[mte], p_te, fl)
        pooled_te[name] = RMR._pred_dict(place_rows(p_te, mte, D.y_te.shape), D.y_te, D.tmask_te, D.d_te, n)
        select[name] = {"params": br["params"], "alpha": a_sh, "clip": cl,
                        "best_iteration": br["best_iteration"], "val_qlike": br["val_qlike"]}

    zrf_tgt = zrf[aa_te + wf.horizon]
    lk = lock_keys_fold(D.y_te, zrf_tgt, mte, D.d_te, C.LIMIT_LOCK_MULT, fl)
    edge_density = float(np.count_nonzero(adj - np.eye(n, dtype=adj.dtype))) / (n * (n - 1))
    return {"pooled": pooled_te, "train_m": tr_m, "val_m": va_m, "test_m": te_m,
            "lock_keys": lk, "select": select, "edge_density": edge_density}


def _metrics_block(pooled, lock_keys, fl):
    """Full metric block for one model: standard + non-lock + lock-only + shares + top-1% + win rates."""
    m = dict(RMR._metrics(pooled, fl))
    nonlock = E.subset(pooled, lock_keys, inside=False)
    lockonly = E.subset(pooled, lock_keys, inside=True)
    m["qlike_nonlock"] = RMR._metrics(nonlock, fl)["qlike"] if nonlock else None
    m["qlike_lockonly"] = RMR._metrics(lockonly, fl)["qlike"] if lockonly else None
    m["n_lock"] = len(lockonly)
    m["lock_qlike_share"] = E.qlike_share(pooled, lock_keys, fl)
    q_excl, n_excl, n_dates = E.exclude_top_pct_dates(pooled, fl, C.TOP_PCT_DATES)
    m["qlike_excl_top_dates"] = q_excl
    m["n_top_dates_excluded"] = n_excl
    ntd, nud, ntk = E.count_summary(pooled)
    m["n_ticker_date"] = ntd
    m["n_unique_dates"] = nud
    m["n_tickers"] = ntk
    return m


def _reference_models(market, horizon):
    """VolGA / LSTM pooled QLIKE + VolGA-vs-HAR-X DM cited from the delivered edge_hmatched result JSON
    (GPU models not retrained here). Returns ``{}`` if the artifact is absent."""
    p = REPO / "results" / "edge_hmatched" / f"edgehm_{market}_h{horizon}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    mt = d.get("metrics", {})
    return {"source": str(p.relative_to(REPO)),
            "VolGA_qlike": mt.get("VolGA", {}).get("qlike"),
            "LSTM_qlike": mt.get("LSTM", {}).get("qlike"),
            "HAR-X_qlike": mt.get("HAR-X", {}).get("qlike"),
            "dm_VolGA_vs_HARX": d.get("dm_date_clustered", {}).get("VolGA_vs_HAR-X")}


def _git_commit():  # pragma: no cover - environment probe
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return None


def run(market, horizon, smoke=False, folds_target=C.FOLDS_TARGET, lookback=C.LOOKBACK,
        out=None, date_str="unset"):  # pragma: no cover - driver glue (logic covered via process_fold/helpers)
    t0 = time.time()
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    if smoke:
        keep = keep[:12]
    panel = build_enriched_panel(files, lookback, horizon, keep)
    aligned = read_aligned_columns(files, panel, ["daily_return", "log_range", "zero_range_flag"])
    ret, logrange, zrf = aligned["daily_return"], aligned["log_range"], aligned["zero_range_flag"]
    Fsm, names, groups = build_sm_features(panel, ret, logrange, panel.feats[:, :, 4])
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=(1 if smoke else folds_target))
    n = len(panel.anchors)
    ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    fl = training_config().qlike_floor

    pooled = {m: {} for m in ("HAR", "HAR-X") + _MODELS}
    tr_acc = {m: [] for m in _MODELS}
    va_acc = {m: [] for m in _MODELS}
    te_acc = {m: [] for m in _MODELS}
    lock_keys, dens, selections = set(), [], []
    print(f"[xgb] {market} h{horizon}: {panel.N} nodes, {len(folds)} folds, {Fsm.shape[2]} sm-features", flush=True)
    for fi, fold in enumerate(folds):
        r = process_fold(panel, fold, wf, Fsm, groups, ret, zrf, fl)
        for m in pooled:
            pooled[m].update(r["pooled"][m])
        for m in _MODELS:
            tr_acc[m].append(r["train_m"][m]); va_acc[m].append(r["val_m"][m]); te_acc[m].append(r["test_m"][m])
        lock_keys |= r["lock_keys"]
        dens.append(r["edge_density"])
        selections.append(r["select"])
        print(f"[xgb] {market} h{horizon} fold {fi + 1}/{len(folds)} done "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    metrics = {m: _metrics_block(pooled[m], lock_keys, fl) for m in pooled}
    win = {m: dict(zip(("ticker_win_rate", "date_win_rate"), E.win_rate_vs(pooled[m], pooled["HAR-X"], fl)))
           for m in _MODELS}
    train_metrics = {m: _agg(tr_acc[m]) for m in _MODELS}
    val_metrics = {m: _agg(va_acc[m]) for m in _MODELS}
    fit_diagnostics = {m: RMR.OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in _MODELS}
    dm = {
        "XGB_resid_base_vs_HAR-X": RMR._dm_all(pooled["XGB_resid_base"], pooled["HAR-X"], horizon, fl),
        "XGB_direct_vs_HAR-X": RMR._dm_all(pooled["XGB_direct"], pooled["HAR-X"], horizon, fl),
        "XGB_resid_market_vs_base": RMR._dm_all(pooled["XGB_resid_market"], pooled["XGB_resid_base"], horizon, fl),
        "XGB_resid_graph_vs_market": RMR._dm_all(pooled["XGB_resid_graph"], pooled["XGB_resid_market"], horizon, fl),
    }
    shock = {m: E.shock_month_diagnostics(pooled[m], pooled["HAR-X"], lock_keys, fl, C.SHOCK_MONTH)
             for m in ("XGB_resid_graph", "HAR-X")}
    ref = _reference_models(market, horizon)
    result = {
        "experiment": "xgboost_residual", "market": market, "horizon": horizon,
        "design": "expanding-window walk-forward XGBoost residual-ratio (enriched panel, CPU)",
        "data_source": f"data/processed_enriched/{market}", "git_commit": _git_commit(),
        "date": date_str, "smoke": smoke, "num_nodes": int(panel.N), "n_folds": len(folds),
        "lookback": lookback, "folds_target": wf.folds_target, "retrain_cadence_K": K,
        "seed": C.XGB_SEED, "device": "cpu", "n_jobs": C.XGB_N_JOBS, "feature_names": names,
        "versions": _versions(), "config": _provenance(),
        "edge_density_mean": float(np.mean(dens)) if dens else None,
        "metrics": metrics, "win_rates_vs_harx": win, "dm_date_clustered": dm,
        "train_metrics": train_metrics, "val_metrics": val_metrics, "fit_diagnostics": fit_diagnostics,
        "selections_per_fold": selections, "shock_diagnostics": shock, "reference_models": ref,
        "n_lock_total": len(lock_keys), "seconds": round(time.time() - t0, 1)}
    for m in _MODELS + ("HAR-X",):
        mm = metrics[m]
        print(f"[xgb] {m}: qlike={mm['qlike']:.4f} nonlock={mm['qlike_nonlock']} "
              f"fit={fit_diagnostics.get(m, {}).get('status', 'det')}", flush=True)
    for name, dd in dm.items():
        print(f"[xgb] DM {name} qlike p={dd['qlike'].get('p_value')} ({dd['qlike'].get('favors')})", flush=True)
    if not smoke:
        out = Path(out) if out else REPO / "results" / "xgboost_residual" / f"xgb_{market}_h{horizon}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(f"[xgb] wrote {out}", flush=True)
    return result


def _versions():  # pragma: no cover - environment probe
    import sklearn
    import xgboost
    return {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "python": sys.version.split()[0]}


def _provenance():  # pragma: no cover - static config snapshot
    return {"pk_lags": list(C.PK_LAGS), "roll_means": list(C.ROLL_MEANS), "oof_splits": C.OOF_SPLITS,
            "oof_warmup_frac": C.OOF_WARMUP_FRAC, "alpha_grid": list(C.ALPHA_GRID),
            "clip_grid": list(C.CLIP_GRID), "grid_size": len(C.XGB_GRID), "edge_top_k": C.EDGE_TOP_K,
            "qlike_floor": C.QLIKE_FLOOR, "limit_lock_mult": C.LIMIT_LOCK_MULT}


def main():  # pragma: no cover - entry driver
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn30", choices=["vn30", "vn100"])
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=C.FOLDS_TARGET)
    ap.add_argument("--lookback", type=int, default=C.LOOKBACK)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--date", default="unset", help="fixed YYYY-MM-DD stamp for the result JSON (scripts can't call Date.now)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.market, a.horizon, a.smoke, a.folds_target, a.lookback, a.out, a.date)


if __name__ == "__main__":  # pragma: no cover
    main()
