"""GNNHAR (Zhang, Pu, Cucuringu & Dong, Int. J. Forecasting 2024, arXiv:2308.01419; official code
https://github.com/chaozhang-ox/GNNHAR) as a directly comparable baseline in the project's full-matrix
walk-forward protocol, on HOSE (local GPU) and SP500 (Colab).

The faithful GNNHAR model + tensor plumbing are REUSED read-only from the already-tested
``scripts/eda/gnnhar_sp500.py`` (``import gnnhar_sp500 as G``): ``G.GNNHAR`` (``relu(H1 + GCN)`` with
GraphConvLayer ``adj @ (X @ W) + b``), ``G._qlike_loss`` (masked paper QLIKE), ``G.build_fold_tensors`` /
``G.row_preds`` / ``G._row_in_te``. This module contributes the tunable-constant config, a training loop
that records learning curves, the 4-model walk-forward comparison, and the gate-aware single-file
incremental output. See design/design.md.

Models (leave-one-out on the graph edge):
  * HAR             -- OLS on the 3 HAR lags, floored (paper baseline; deterministic).
  * GBM             -- own-8 gamma-GBM (FM.gbm on OWN8), seed-ensemble (paper GBM(own-8)).
  * GNNHAR          -- full faithful GNNHAR2L on the 3 HAR lags with the train-only corr graph.
  * GNNHAR-nograph  -- Full - graph: identical network with A = I (per-node MLP; the paper's own ablation).

Output: results/gamma_gbm/gnnhar_<market>.json -- top-level flat dicts keyed ``<model>_h<h>`` carrying
test/train/val metrics + fit_diagnostics + learning_curves + date-clustered DM, so the pre-push
overfit-evidence gate validates every learned horizon. Written atomically after EACH horizon.

Run: ``python run_gnnhar.py [hose|sp500] [--smoke]``.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"), str(REPO / "scripts" / "quality_gate"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import gnnhar_sp500 as G  # noqa: E402  (faithful GNNHAR model + tensor plumbing, reused read-only)
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402
import overfit_check as OF  # noqa: E402
import gnnhar_config as cfg  # noqa: E402  (unique module name: submission/soict_lstm_gat/config.py collides)

FL = FM.FL
DEVICE = G.DEVICE
MODELS = ["HAR", "GBM", "GNNHAR", "GNNHAR-nograph"]
LEARNED = ["GNNHAR", "GNNHAR-nograph"]
GCFG = [("GNNHAR", cfg.N_GCN, "corr"), ("GNNHAR-nograph", cfg.N_GCN, "none")]   # (name, n_gcn, adj_type)
CMP = [("GNNHAR", "HAR"), ("GNNHAR", "GBM"), ("GNNHAR", "GNNHAR-nograph")]


def train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn, seed, max_epochs, patience):
    """Train one GNNHAR on the scaled target, batched over dates on GPU, early-stopping on validation QLIKE.

    Returns ``(best_val_model, best_val_loss, curve)`` where ``curve`` is a per-epoch list of
    ``{"epoch", "train", "val"}`` scaled-QLIKE(+1) losses (the learning-curve evidence the reused
    ``G._train_once`` does not expose). ``tr_idx`` / ``va_idx`` are date positions into ``X``'s first axis."""
    torch.manual_seed(seed)
    model = G.GNNHAR(in_f, cfg.N_HID, n_gcn).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    tr = torch.as_tensor(tr_idx, device=DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    best_val, best_state, bad = float("inf"), None, 0
    curve = []
    for epoch in range(max_epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen, device=DEVICE)]
        tl_sum, n_b = None, 0
        for s in range(0, len(perm), cfg.BATCH_DATES):
            b = perm[s:s + cfg.BATCH_DATES]
            opt.zero_grad()
            loss = G._qlike_loss(model(X[b], adj), Ys[b], Mt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            tl_sum = loss.detach() if tl_sum is None else tl_sum + loss.detach()   # curve train pt from the
            n_b += 1                                                               # batch losses (no 2nd pass)
        model.eval()
        with torch.no_grad():                                                     # val forward = early-stop signal
            vl = G._qlike_loss(model(X[va_idx], adj), Ys[va_idx], Mt[va_idx]).item()
        tl = float((tl_sum / n_b).item())
        curve.append({"epoch": epoch, "train": tl, "val": vl})
        if vl < best_val - 1e-6:   # config-ok: numerical improvement guard, not a tunable pipeline constant
            best_val, best_state, bad = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience and epoch >= cfg.MIN_EPOCHS - 1:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, curve


def fit(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn, seed, max_epochs, patience):
    """``train_with_curves`` with the paper's restart-on-collapse: a scaled val-QLIKE(+1) above
    ``cfg.COLLAPSE_VAL`` (or non-finite) means dead-ReLU / non-convergence -> retry with a fresh seed
    (faithful to GNNHAR.py:428-433). Returns ``(model, best_val, curve)`` of the best attempt."""
    model, best_val, curve = train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn,
                                                seed, max_epochs, patience)
    tries = 0
    while (not np.isfinite(best_val) or best_val > cfg.COLLAPSE_VAL) and tries < cfg.MAX_RESTARTS:
        tries += 1
        model, best_val, curve = train_with_curves(X, Ys, Mt, adj, tr_idx, va_idx, in_f, n_gcn,
                                                    seed + 1000 * tries, max_epochs, patience)
    return model, best_val, curve


def _metrics5(y, p):
    """The 5 point/volatility metrics for one model on one split (DirAcc N/A for a pooled cross-section)."""
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "r2": M.r2(y, p), "qlike": M.qlike(y, p, floor=FL)}


def _fold_splits(fold, ts, tend, embargo):
    """The four row windows: full-train ``trf`` (baseline fit window), ``trf_e`` (train minus the last
    VALID_LEN dates), ``vaf`` (those last VALID_LEN train dates = the GNN validation split), ``tef`` (test)."""
    trf = fold[(fold.date >= S1.TRAIN_START) & (fold.date < ts - embargo)]
    tef = fold[(fold.date >= ts) & (fold.date < tend)]
    val_dates = np.sort(trf["date"].unique())[-cfg.VALID_LEN:]
    is_val = trf["date"].isin(val_dates)
    return trf, trf[~is_val], trf[is_val], tef


def _gnn_maps(fold, tickers, tr_date_fn, ts, tend, trf_e, vaf, tef):
    """Build the shared fold tensors (HAR3 features) once and the (date_idx, row_in, cidx) gather maps for
    each split, so both adjacency variants score the SAME trained cells."""
    Xn, _Yraw, Ys, mask, sc, dpos, cpos, dts = G.build_fold_tensors(fold, tickers, cfg.HAR3, tr_date_fn)
    Xt = torch.as_tensor(Xn, device=DEVICE)
    Yst = torch.as_tensor(Ys, device=DEVICE)
    Mt = torch.as_tensor(mask, device=DEVICE)
    all_tr = np.where(np.array([tr_date_fn(d) for d in dts]))[0]
    va_idx, tr_idx = all_tr[-cfg.VALID_LEN:], all_tr[:-cfg.VALID_LEN]
    te_idx = np.where((dts >= np.datetime64(ts)) & (dts < np.datetime64(tend)))[0]
    maps = {"te": (te_idx, G._row_in_te(tef["date"], dpos, te_idx), tef["ticker"].map(cpos).to_numpy()),
            "tr": (tr_idx, G._row_in_te(trf_e["date"], dpos, tr_idx), trf_e["ticker"].map(cpos).to_numpy()),
            "va": (va_idx, G._row_in_te(vaf["date"], dpos, va_idx), vaf["ticker"].map(cpos).to_numpy())}
    return Xt, Yst, Mt, sc, tr_idx, va_idx, maps


def _horizon(a, h, market, seeds, max_epochs, patience, fold_cap, min_train):
    """Walk-forward one horizon; return the pooled per-model evidence dict, or None if no fold qualifies."""
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    tickers = sorted(a["ticker"].unique())
    n_nodes = len(tickers)
    preds = {sp: {m: [] for m in MODELS} for sp in ("te", "tr", "va")}
    yy = {"te": [], "tr": [], "va": []}
    seed_q = {g[0]: [] for g in GCFG}
    curves = {g[0]: [] for g in GCFG}
    dts_te = []
    n_done = 0
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
        te = a[(a.date >= ts) & (a.date < tend)]
        if len(te) == 0 or len(tr) < min_train:
            continue
        if fold_cap is not None and n_done >= fold_cap:
            break
        n_done += 1
        Wc, _ = S1.build_graph(tr, tickers, np.random.default_rng(S1.RNG_SEED + k))
        keep = list(dict.fromkeys(cfg.OWN8 + ["date", "ticker", "y"]))
        fold = a.loc[(a.date >= S1.TRAIN_START) & (a.date < tend), keep].copy()
        trf, trf_e, vaf, tef = _fold_splits(fold, ts, tend, embargo)
        y_te = tef["y"].to_numpy(float)
        for sp, sub in (("te", tef), ("tr", trf_e), ("va", vaf)):
            preds[sp]["HAR"].append(FM._har_ols(trf, sub))
            preds[sp]["GBM"].append(np.mean([FM.gbm(trf, sub, cfg.OWN8, s) for s in cfg.SEEDS], 0))
        tcut = ts - embargo
        tr_date_fn = (lambda d, tcut=tcut: d < np.datetime64(tcut))
        Xt, Yst, Mt, sc, tr_idx, va_idx, maps = _gnn_maps(fold, tickers, tr_date_fn, ts, tend, trf_e, vaf, tef)
        adj_cache = {"corr": torch.as_tensor(Wc, dtype=torch.float32, device=DEVICE),
                     "none": torch.eye(n_nodes, dtype=torch.float32, device=DEVICE)}
        for name, n_gcn, adj_type in GCFG:
            adj = adj_cache[adj_type]
            spreds = {"te": [], "tr": [], "va": []}
            for sd in seeds:
                model, _bv, curve = fit(Xt, Yst, Mt, adj, tr_idx, va_idx, len(cfg.HAR3), n_gcn,
                                        sd, max_epochs, patience)
                for sp in ("te", "tr", "va"):
                    spreds[sp].append(G.row_preds(model, Xt, adj, *maps[sp], sc))
                seed_q[name].append(float(np.mean(M.per_obs_qlike(y_te, spreds["te"][-1], floor=FL))))
                if n_done == 1:                        # record learning curves on the first fold only (bounded)
                    curves[name].append(curve)
            for sp in ("te", "tr", "va"):
                preds[sp][name].append(np.mean(spreds[sp], 0))
        yy["te"].append(y_te)
        yy["tr"].append(trf_e["y"].to_numpy(float))
        yy["va"].append(vaf["y"].to_numpy(float))
        dts_te.append(tef["date"].to_numpy())
        del fold, Xt, Yst, Mt, adj_cache
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    if not dts_te:
        return None
    return _score(h, market, preds, yy, dts_te, seed_q, curves)


def _score(h, market, preds, yy, dts_te, seed_q, curves):
    """Pool per-obs QLIKE over folds; 5 metrics per split per model; classify_fit per learned model;
    date-clustered DM for each comparison. Returns the per-horizon evidence block."""
    y = np.concatenate(yy["te"])
    dates = np.concatenate(dts_te)
    e = {m: M.per_obs_qlike(y, np.concatenate(preds["te"][m]), floor=FL) for m in MODELS}
    metrics = {m: _metrics5(y, np.concatenate(preds["te"][m])) for m in MODELS}
    y_tr, y_va = np.concatenate(yy["tr"]), np.concatenate(yy["va"])
    train_metrics = {m: _metrics5(y_tr, np.concatenate(preds["tr"][m])) for m in MODELS}
    val_metrics = {m: _metrics5(y_va, np.concatenate(preds["va"][m])) for m in MODELS}
    fit_diag = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in LEARNED}
    dm = {}
    for x, b in CMP:
        r = ST.date_clustered_dm(e[x], e[b], dates, h)
        dm[f"{x}_vs_{b}"] = {"p_value": float(r["p_value"]), "mean_diff": float(r["mean_diff"]),
                             "gain_pct": (metrics[b]["qlike"] - metrics[x]["qlike"]) / metrics[b]["qlike"] * 100.0}
    res = {"n": int(len(y)), "n_folds": len(dts_te), "metrics": metrics, "train_metrics": train_metrics,
           "val_metrics": val_metrics, "fit_diagnostics": fit_diag, "dm": dm,
           "learning_curves": {m: curves[m] for m in LEARNED},
           "qlike": {m: metrics[m]["qlike"] for m in MODELS},
           "per_seed_qlike": {m: [round(v, 6) for v in seed_q[m]] for m in seed_q}}
    if market == "hose":
        res["per_fold"] = {m: [float(np.mean(M.per_obs_qlike(yy["te"][i], preds["te"][m][i], floor=FL)))
                               for i in range(len(dts_te))] for m in MODELS}
    return res


def _checkpoint(out, out_path):
    """Atomically write accumulated results so a Colab disconnect keeps completed horizons."""
    if out_path is None:
        return
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)


def _merge(out, res, h, market):
    """Fold one horizon's evidence into the top-level flat dicts under ``<model>_h<h>`` keys."""
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m, v in res[blk].items():
            out[blk][f"{m}_h{h}"] = v
    for m, v in res["fit_diagnostics"].items():
        out["fit_diagnostics"][f"{m}_h{h}"] = v
    for m, v in res["learning_curves"].items():
        out["learning_curves"][f"{m}_h{h}"] = v
    for c, v in res["dm"].items():
        out["dm"][f"{c}_h{h}"] = v
    for m, v in res["qlike"].items():
        out["qlike"][f"{m}_h{h}"] = v
    out["n"][f"h{h}"], out["n_folds"][f"h{h}"] = res["n"], res["n_folds"]
    out["per_seed_qlike"].update({f"{m}_h{h}": v for m, v in res["per_seed_qlike"].items()})
    if market == "hose":
        out["per_fold_qlike"].update({f"{m}_h{h}": v for m, v in res["per_fold"].items()})


def run(market, load_fn=None, out_path=None, smoke=False, min_train=None, max_epochs=None, seeds=None):
    """GNNHAR-vs-HAR/GBM walk-forward for a market; returns the JSON-serialisable result dict and, when
    ``out_path`` is given, flushes after EACH horizon (Colab disconnect resilience)."""
    load_fn = load_fn or FM.load
    frames, _sect, _edates = load_fn(market)
    seeds = seeds if seeds is not None else ((0,) if smoke else cfg.SEEDS)
    max_epochs = max_epochs if max_epochs is not None else (cfg.SMOKE_EPOCHS if smoke else cfg.MAX_EPOCHS)
    fold_cap = 1 if smoke else None
    if min_train is None:
        min_train = cfg.MIN_TRAIN_ROWS.get(market, cfg.MIN_TRAIN_ROWS["default"])
    horizons = (1,) if smoke else cfg.HORIZONS
    out = {"market": market, "horizons": list(horizons), "n": {}, "n_folds": {}, "metrics": {},
           "train_metrics": {}, "val_metrics": {}, "fit_diagnostics": {}, "learning_curves": {},
           "dm": {}, "qlike": {}, "per_seed_qlike": {}}
    if market == "hose":
        out["per_fold_qlike"] = {}
    for h in horizons:
        t0 = time.time()
        a = FM.panel(frames, {}, h)
        res = _horizon(a, h, market, seeds, max_epochs, cfg.PATIENCE, fold_cap, min_train)
        if res is not None:
            _merge(out, res, h, market)
            _checkpoint(out, out_path)
            print(f"  h{h}: GNNHAR QLIKE {res['qlike']['GNNHAR']:.4f} vs HAR {res['qlike']['HAR']:.4f} "
                  f"vs GBM {res['qlike']['GBM']:.4f} (n={res['n']:,}, {res['n_folds']} folds, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        del a
        gc.collect()
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h in out["horizons"]:
        key = f"GNNHAR_h{h}"
        if key not in out["metrics"]:
            continue
        print(f"\n=== {market} h{h} (n={out['n'][f'h{h}']:,}) ===", flush=True)
        for m in MODELS:
            print(f"  {m:16s} QLIKE {out['qlike'][f'{m}_h{h}']:.4f}", flush=True)
        for x, b in CMP:
            d = out["dm"][f"{x}_vs_{b}_h{h}"]
            print(f"  DM {x} vs {b:16s}: {d['gain_pct']:+.2f}% (p={d['p_value']:.3f})", flush=True)
        for m in LEARNED:
            print(f"  fit[{m}] = {out['fit_diagnostics'][f'{m}_h{h}']['status']}", flush=True)


def main():  # pragma: no cover - entry driver: loads real data, trains on GPU, writes JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", default="hose", choices=("hose", "sp500"))
    ap.add_argument("--smoke", action="store_true", help="1 horizon, 1 fold, 1 seed, few epochs")
    args = ap.parse_args()
    outp = REPO / "results" / "gamma_gbm" / f"gnnhar_{args.market}{'_smoke' if args.smoke else ''}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(f"loaded market={args.market} device={DEVICE}", flush=True)
    out = run(args.market, out_path=outp, smoke=args.smoke)
    _print(args.market, out)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
