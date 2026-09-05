"""Edge-fix experiment: horizon-matched volume->Parkinson edge + significance-floor fallback.

Root cause (verified in masked_rich._directed_vol2pk): the delivered vol->PK edge always correlates
volume_i(t) with sqrt_pk_j(t+1) -- a FIXED 1-day lead-lag -- regardless of the forecast horizon h. It
matches the target at h1 but is mis-aligned at h5/h10/h22 (the model forecasts vol at t+h while the
graph encodes a t->t+1 relation), which is one reason the graph is unstable at longer horizons.

Fix tested here (`directed_vol2pk_hmatched`):
  * horizon-matched: correlate volume_i(t) with sqrt_pk_j(t+h) (shift by h, not 1) so the edge encodes
    the SAME relation the model forecasts;
  * significance floor: keep a source only if |corr| clears a Bonferroni threshold over the (n-1)
    candidate sources per target (z_bonf / sqrt(n_pairs), z_bonf = Phi^-1(1 - alpha/(2(n-1)))); a
    target with no source passing the floor keeps only its self-loop -> the graph automatically falls
    back to no-graph where the lead-lag signal is at noise level (expected at long h on VN data, per
    project EDA: h22 signal ~1.05x shuffled null). A plain per-pair z/sqrt(m) floor is too weak here
    because n_pairs is large and the Top-K selection over ~100 sources always clears it.

Compares, on the same folds/seeds: HAR, HAR-X, no-graph LSTM, VolGA (fixed lag-1, the delivered edge),
VolGA_hm (horizon-matched + floor). DM: VolGA_hm vs VolGA (edge fix effect), VolGA_hm vs LSTM (graph
marginal value under the fixed edge), VolGA vs LSTM. Reports per-fold edge density to show the fallback.

Run:  .venv_gpu_encode/Scripts/python.exe baselines/2026-09-05_edge_horizon_matched/code/run_edge_hmatched.py --market vn100 --horizon 22 --folds-target 7 --n-seeds 3
Smoke: ... --smoke   (1 fold, 8 epochs, 3 seeds)
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

import run_masked_rich as RMR  # noqa: E402
import masked_rich as MR  # noqa: E402
import pipeline_config as pc  # noqa: E402
from run_walkforward import _har_ols_preds, training_config  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402

_MODELS = ("HAR", "HAR-X", "LSTM", "VolGA", "VolGA_hm")
EDGE_SIG_ALPHA = 0.05   # Bonferroni family-wise level over the (n-1) candidate sources per target


def directed_vol2pk_hmatched(vshock, sqrt_pk, last_row, horizon, top_k,
                             alpha=EDGE_SIG_ALPHA, min_pairs=MR._MIN_PAIRS):
    """Horizon-matched directed edge A[j,i] = corr(vshock_i(t), sqrt_pk_j(t+h)) over TRAIN dates, kept
    only if |corr| clears a Bonferroni significance floor over the (n-1) candidate sources per target
    (threshold z_bonf/sqrt(n_pairs), z_bonf = Phi^-1(1 - alpha/(2(n-1)))); Top-K per target; self-loop=1.
    ``alpha=None`` disables the floor. A target with no surviving source keeps only its self-loop ->
    local no-graph fallback (expected at long h where the lead-lag signal is at noise level).

    Vectorised: the pairwise-complete Pearson correlation of every (source i, target j) pair is computed
    with BLAS matmuls over NaN-masked [T, N] matrices instead of an O(N^2) Python np.corrcoef loop, so it
    scales to large universes (S&P 500, N~500). ``R[i, j] = corr(src_i, tgt_j)`` over the rows where both
    are finite; count/sum matrices give the exact per-pair r = (n*Sab - Sa*Sb) / (sqrt(n*Saa - Sa^2) *
    sqrt(n*Sbb - Sb^2)); den==0 (zero variance) and n<min_pairs collapse to NaN and are dropped."""
    v = vshock[:last_row + 1]
    p = sqrt_pk[:last_row + 1]
    src = v[:-horizon]                 # source volume at t          [T, N]
    tgt = p[horizon:]                  # target sqrt_pk at t+h       [T, N]  (horizon-matched)
    n = v.shape[1]
    z_bonf = NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(n - 1, 1))) if alpha else 0.0

    Sf = np.isfinite(src); Ff = np.isfinite(tgt)                    # finite masks
    S = np.where(Sf, src, 0.0).astype(np.float64)                  # NaN -> 0 so masked sums drop them
    F = np.where(Ff, tgt, 0.0).astype(np.float64)
    Sm = Sf.astype(np.float64); Fm = Ff.astype(np.float64)
    npair = Sm.T @ Fm                                              # [i,j] jointly-finite row count
    sa = S.T @ Fm                                                  # sum src_i over rows where tgt_j finite
    sb = Sm.T @ F                                                  # sum tgt_j over rows where src_i finite
    saa = (S * S).T @ Fm; sbb = Sm.T @ (F * F); sab = S.T @ F
    with np.errstate(invalid="ignore", divide="ignore"):
        num = npair * sab - sa * sb
        den = np.sqrt(npair * saa - sa * sa) * np.sqrt(npair * sbb - sb * sb)
        R = num / den                                             # [source i, target j] Pearson r
    R[~np.isfinite(R)] = np.nan                                    # den==0 (zero variance) -> drop
    R[npair < min_pairs] = np.nan                                 # too few overlapping pairs -> drop
    np.fill_diagonal(R, np.nan)                                    # i==j: no self edge from correlation
    with np.errstate(invalid="ignore", divide="ignore"):
        thr = (z_bonf / np.sqrt(npair)) if alpha else 0.0         # per-pair Bonferroni floor (0 = disabled)
    sig = np.isfinite(R) & (np.abs(R) > thr)                       # [source, target]
    A = np.zeros((n, n), dtype=np.float32)
    scores = np.where(sig, np.abs(R), -np.inf)                     # rank valid sources per target column
    topk = np.argsort(-scores, axis=0)[:top_k]                     # [top_k, N_targets]
    tgts = np.arange(n)
    for rank in range(topk.shape[0]):
        srcs = topk[rank]
        keep = sig[srcs, tgts]
        A[tgts[keep], srcs[keep]] = R[srcs[keep], tgts[keep]].astype(np.float32)   # A[j,i] = corr(src_i, tgt_j)
    np.fill_diagonal(A, 1.0)
    return A


def _edge_density(A):
    """Fraction of off-diagonal entries that are non-zero (kept edges) -- shows the fallback at long h."""
    n = A.shape[0]
    off = A.copy(); np.fill_diagonal(off, 0.0)
    return float(np.count_nonzero(off)) / (n * (n - 1))


def _progress(body: str, elapsed_min: float) -> str:
    """Format an edgehm progress line: ``[edgehm] <body> (<elapsed> min)`` (testable; run() prints it)."""
    return f"[edgehm] {body} ({elapsed_min:.1f} min)"


def _agg_split_metrics(dicts):
    """Mean of mse/qlike/r2 across per-fold split-metric dicts; n = total obs. Aggregates the per-fold
    train (or val) fit metrics of a walk-forward run into one dict for the overfit verdict."""
    keys = ("mse", "qlike", "r2")
    agg = {k: float(np.mean([d[k] for d in dicts])) for k in keys}
    agg["n"] = int(sum(d["n"] for d in dicts))
    return agg


def _pool(o, D):
    return RMR._pred_dict(o, D.y_te, D.tmask_te, D.d_te, D.N)


def run(horizon, folds_target, epochs, smoke, out=None, n_seeds=3, market="vn100",
        lookback=pc.LOOKBACK, batch=None):  # pragma: no cover
    t0 = time.time()
    files = _glob.glob(enriched_glob(market))
    keep = frozen_universe(files, lookback, horizon)
    panel = build_enriched_panel(files, lookback, horizon, keep)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=(1 if smoke else folds_target))
    n = len(panel.anchors); ts = int(n * wf.test_frac)
    K = max(1, math.ceil((n - ts) / wf.folds_target))
    folds = make_folds(n, ts, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    tc_kw = {"batch": batch} if batch else {}   # None -> training_config's own default batch; larger = faster on big GPUs
    cfg = training_config(epochs=(8 if smoke else epochs),
                          seeds=((42, 123, 2026) if smoke else (42, 123, 2026, 7, 2024)[:n_seeds]), **tc_kw)
    fl = cfg.qlike_floor
    pooled = {m: {} for m in _MODELS}
    lstm = [{} for _ in cfg.seeds]; volga = [{} for _ in cfg.seeds]; volga_hm = [{} for _ in cfg.seeds]
    dens_fix, dens_hm = [], []
    _NN = ("LSTM", "VolGA", "VolGA_hm")                       # NN models get over/under-fit evidence
    tr_acc = {m: [] for m in _NN}; va_acc = {m: [] for m in _NN}; curves = {m: [] for m in _NN}
    print(f"[edgehm] {market} h{horizon}: {panel.N} nodes, {len(folds)} folds, {len(cfg.seeds)} seeds", flush=True)
    for fi, fold in enumerate(folds):
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        eye = np.eye(D.N, dtype=np.float32)
        last_tr_row = int(panel.anchors[fold.train][-1]) + wf.horizon
        adj_fix = D.adj_vol2pk                                   # delivered fixed lag-1 edge
        adj_hm = directed_vol2pk_hmatched(panel.feats[:, :, 4], np.sqrt(panel.pk),
                                          last_tr_row, wf.horizon, MR.EDGE_TOP_K)
        dens_fix.append(_edge_density(adj_fix)); dens_hm.append(_edge_density(adj_hm))
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        har, harx = _har_ols_preds(D, fl, nfloor)
        pooled["HAR"].update(_pool(har["te"], D)); pooled["HAR-X"].update(_pool(harx["te"], D))
        print(_progress(f"fold {fi + 1}/{len(folds)} start: N={D.N} train={len(D.X_tr)} val={len(D.X_va)} "
                        f"test={len(D.y_te)} anchors, edge dens fix={dens_fix[-1]:.3f} hm={dens_hm[-1]:.3f} - "
                        f"training {len(cfg.seeds)} seeds x 3 models", (time.time() - t0) / 60), flush=True)
        fold_out = {m: [] for m in _NN}                      # full per-seed returns (train/val/test + curves)
        for si, s in enumerate(cfg.seeds):
            o_l = RMR.train_masked_rich(D, cfg, s, False, eye, return_splits=True)
            o_v = RMR.train_masked_rich(D, cfg, s, True, adj_fix, return_splits=True)
            o_h = RMR.train_masked_rich(D, cfg, s, True, adj_hm, return_splits=True)
            lstm[si].update(_pool(o_l["test"], D)); volga[si].update(_pool(o_v["test"], D))
            volga_hm[si].update(_pool(o_h["test"], D))
            fold_out["LSTM"].append(o_l); fold_out["VolGA"].append(o_v); fold_out["VolGA_hm"].append(o_h)
            print(_progress(f"  fold {fi + 1}/{len(folds)} seed {si + 1}/{len(cfg.seeds)} done "
                            f"(LSTM+VolGA+VolGA_hm)", (time.time() - t0) / 60), flush=True)
        for m in _NN:                                        # per-fold seed-ensembled train/val fit metrics + curves
            etr = RMR._ens_split(fold_out[m], "train"); eva = RMR._ens_split(fold_out[m], "val")
            tr_acc[m].append(RMR._split_metrics(etr, D.y_tr, D.tmask_tr, fl))
            va_acc[m].append(RMR._split_metrics(eva, D.y_va, D.tmask_va, fl))
            curves[m].append({"fold": fi, "train": [o["train_curve"] for o in fold_out[m]],
                              "val": [o["val_curve"] for o in fold_out[m]],
                              "best_epoch": [o["best_epoch"] for o in fold_out[m]]})
        print(f"[edgehm] fold {fi + 1}/{len(folds)} done, edge density fix={dens_fix[-1]:.3f} "
              f"hm={dens_hm[-1]:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pooled["LSTM"] = RMR._ens(lstm); pooled["VolGA"] = RMR._ens(volga); pooled["VolGA_hm"] = RMR._ens(volga_hm)
    metrics = {m: RMR._metrics(pooled[m], fl) for m in _MODELS}
    train_metrics = {m: _agg_split_metrics(tr_acc[m]) for m in _NN}   # over/under-fit evidence (walk-forward mean)
    val_metrics = {m: _agg_split_metrics(va_acc[m]) for m in _NN}
    fit_diagnostics = {m: RMR.OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in _NN}
    dm = {"VolGAhm_vs_VolGA": RMR._dm_all(pooled["VolGA_hm"], pooled["VolGA"], horizon, fl),
          "VolGAhm_vs_LSTM": RMR._dm_all(pooled["VolGA_hm"], pooled["LSTM"], horizon, fl),
          "VolGA_vs_LSTM": RMR._dm_all(pooled["VolGA"], pooled["LSTM"], horizon, fl)}
    result = {"experiment": "edge_horizon_matched", "horizon": horizon, "market": market,
              "num_nodes": int(panel.N), "n_folds": len(folds), "seeds": list(cfg.seeds), "smoke": smoke,
              "edge_sig_alpha": EDGE_SIG_ALPHA, "edge_density_fix_mean": float(np.mean(dens_fix)),
              "edge_density_hm_mean": float(np.mean(dens_hm)), "seconds": time.time() - t0,
              "metrics": metrics, "train_metrics": train_metrics, "val_metrics": val_metrics,
              "fit_diagnostics": fit_diagnostics, "learning_curves": curves, "dm_date_clustered": dm}
    print(f"[edgehm] QLIKE h{horizon}: " + ", ".join(f"{m}={metrics[m]['qlike']:.4f}" for m in _MODELS), flush=True)
    print("[edgehm] fit (train->val->test): " + ", ".join(f"{m}={fit_diagnostics[m]['status']}" for m in _NN), flush=True)
    print(f"[edgehm] edge density: fix(lag1)={np.mean(dens_fix):.3f} hm(h={horizon})={np.mean(dens_hm):.3f}", flush=True)
    a = dm["VolGAhm_vs_VolGA"]["qlike"]; b = dm["VolGAhm_vs_LSTM"]["qlike"]
    print(f"[edgehm] DM VolGAhm-vs-VolGA qlike p={a['p_value']:.3f} ({a['favors']}); "
          f"VolGAhm-vs-LSTM qlike p={b['p_value']:.3f} ({b['favors']})", flush=True)
    if not smoke:
        out = Path(out) if out else REPO / "results" / "edge_hmatched" / f"edgehm_{market}_h{horizon}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[edgehm] wrote {out}", flush=True)
    return result


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=22, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--lookback", type=int, default=22)   # experiment value (matches delivered VolGA edge); library default = pc.LOOKBACK
    ap.add_argument("--batch", type=int, default=None)     # None -> training_config default; raise (e.g. 1024) to use a big GPU (A100) better
    ap.add_argument("--market", default="vn100", choices=["vn100", "vn30", "hose", "hnx", "sp500", "sp500_clean"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.horizon, a.folds_target, a.epochs, a.smoke, a.out, a.n_seeds, a.market, a.lookback, a.batch)


if __name__ == "__main__":  # pragma: no cover
    main()
