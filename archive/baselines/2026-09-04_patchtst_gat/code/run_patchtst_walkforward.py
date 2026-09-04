"""Multi-horizon expanding-window walk-forward: HAR-X vs no-graph PatchTST vs PatchTST+GAT on clean
enriched panels — the PatchTST-backbone analogue of ``run_volga_walkforward.py``.

PatchTST+GAT = PatchTST temporal branch + 2-hop weighted-GAT over a per-fold TRAIN-ONLY vol->PK Top-5
graph (the exact VolGA graph). Each fold refits all three learned/OLS models on the data before the
retrain point, freezes them (+ graph + scalers), and forecasts the next K days. Predictions are pooled
over the whole OOS region; the leave-one-out graph comparison is ``PatchTST+GAT - PatchTST`` (graph
marginal value) with both vs HAR-X, via a date-clustered Diebold-Mariano.

Reuses the delivered fold/leakage guards (wf_folds), the enriched reader + per-fold packer
(wf_enriched_panel), HAR-X OLS + GPU politeness (run_walkforward), and RMR metric/DM/evidence helpers
UNCHANGED; only the PatchTST trainer (run_patchtst) and the model names are new.

Real run (GPU, per horizon):  CUDA_VISIBLE_DEVICES=0 python run_patchtst_walkforward.py --horizon 1 --lookback 22
Smoke (CPU sanity):           CUDA_VISIBLE_DEVICES= python run_patchtst_walkforward.py --smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code",
           REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           REPO / "scripts" / "quality_gate", _HERE):
    sys.path.insert(0, str(_p))

import metrics as M            # noqa: E402  (pooled train/val aggregates)
import overfit_check as OF     # noqa: E402  (fit verdict + evidence gate, self-run with correct learned tuple)
import pipeline_config as pc   # noqa: E402  (single source of truth for shared tunables)
import run_masked_rich as RMR  # noqa: E402  (metric/DM/evidence helpers, reused unchanged)
from run_walkforward import _har_ols_preds, training_config, wait_for_gpu  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402

from patchtst_config import PatchTSTHParams  # noqa: E402
from run_patchtst import train_patchtst  # noqa: E402

MARKETS = ("vn30", "vn100", "hose", "hnx", "sp500")
# learned model keys — PatchTST replaces the LSTM backbone. NOTE (design §8): the shared pre-push
# overfit gate defaults to LEARNED=("LSTM","LSTM_wGAT_vol2pk") and will not match these names; we
# self-run the evidence check with LEARNED below so the evidence is still enforced in-code.
LEARNED = ("PatchTST", "PatchTST_wGAT_vol2pk")
_MODELS = ("HAR", "HAR-X") + LEARNED


def enriched_glob(market: str) -> str:
    return str(REPO / "data" / "processed_enriched" / market / "*.csv")


@dataclass(frozen=True)
class PatchTSTWFConfig:
    lookback: int = pc.LOOKBACK
    horizon: int = pc.WF_HORIZON
    folds_target: int = 22
    val: int = pc.WF_VAL_TAIL
    test_frac: float = pc.WF_TEST_FRAC
    test_start: int | None = None


def default_out_path(horizon: int, market: str = "vn100") -> Path:
    return REPO / "results" / "walkforward_patchtst" / f"walkforward_patchtst_{market}_h{horizon}.json"


def _agg_metrics(y, p, floor):
    return {"mse": M.mse(y, p), "rmse": M.rmse(y, p), "mae": M.mae(y, p),
            "qlike": M.qlike(y, p, floor), "r2": M.r2(y, p), "n": int(len(y))}


def run_fold(panel, fold, wf: PatchTSTWFConfig, cfg, hp: PatchTSTHParams | None = None):
    """Train HAR/HAR-X/PatchTST/PatchTST+GAT on one fold; return pooled-update dicts, flat train/val
    arrays, and this fold's fit evidence."""
    hp = hp or PatchTSTHParams()
    D = pack_fold(panel, fold, wf.lookback, wf.horizon)     # D.adj_vol2pk = TRAIN-only vol->PK edge
    fl = cfg.qlike_floor
    nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
    har, harx = _har_ols_preds(D, fl, nfloor)
    eye = np.eye(D.N, dtype=np.float32)
    pt_outs = [train_patchtst(D, cfg, s, False, eye, return_splits=True, hp=hp) for s in cfg.seeds]
    ptg_outs = [train_patchtst(D, cfg, s, True, D.adj_vol2pk, return_splits=True, hp=hp) for s in cfg.seeds]

    har_te = RMR._pred_dict(har["te"], D.y_te, D.tmask_te, D.d_te, D.N)
    harx_te = RMR._pred_dict(harx["te"], D.y_te, D.tmask_te, D.d_te, D.N)
    pt_seed_te = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in pt_outs]
    ptg_seed_te = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in ptg_outs]

    tr_pred = {"HAR": har["tr"], "HAR-X": harx["tr"],
               "PatchTST": RMR._ens_split(pt_outs, "train"),
               "PatchTST_wGAT_vol2pk": RMR._ens_split(ptg_outs, "train")}
    va_pred = {"HAR": har["va"], "HAR-X": harx["va"],
               "PatchTST": RMR._ens_split(pt_outs, "val"),
               "PatchTST_wGAT_vol2pk": RMR._ens_split(ptg_outs, "val")}
    te_pred = {"HAR": har["te"], "HAR-X": harx["te"],
               "PatchTST": RMR._ens_split(pt_outs, "test"),
               "PatchTST_wGAT_vol2pk": RMR._ens_split(ptg_outs, "test")}
    train_m = {m: RMR._split_metrics(tr_pred[m], D.y_tr, D.tmask_tr, fl) for m in _MODELS}
    val_m = {m: RMR._split_metrics(va_pred[m], D.y_va, D.tmask_va, fl) for m in _MODELS}
    test_m = {m: RMR._split_metrics(te_pred[m], D.y_te, D.tmask_te, fl) for m in _MODELS}
    fit = {m: OF.classify_fit(train_m[m], val_m[m], test_m[m]) for m in _MODELS}
    curves = {gm: {"train": [o["train_curve"] for o in outs], "val": [o["val_curve"] for o in outs],
                   "best_epoch": [o["best_epoch"] for o in outs]}
              for gm, outs in (("PatchTST", pt_outs), ("PatchTST_wGAT_vol2pk", ptg_outs))}
    evidence = {"idx": fold.idx, "n_train": int(D.tmask_tr.sum()), "n_val": int(D.tmask_va.sum()),
                "n_forecast": int(D.tmask_te.sum()), "train_metrics": train_m, "val_metrics": val_m,
                "test_metrics": test_m, "fit_diagnostics": fit, "learning_curves": curves}

    mtr, mva = D.tmask_tr.astype(bool), D.tmask_va.astype(bool)
    flats = {"train": (D.y_tr[mtr], {m: tr_pred[m][mtr] for m in _MODELS}),
             "val": (D.y_va[mva], {m: va_pred[m][mva] for m in _MODELS})}
    upd = {"HAR": har_te, "HAR-X": harx_te, "PatchTST_seeds": pt_seed_te, "PatchTST_GAT_seeds": ptg_seed_te}
    return upd, flats, evidence


def run_walkforward(files, wf: PatchTSTWFConfig, cfg, keep_tickers, out_path=None, market="vn100",
                    hp: PatchTSTHParams | None = None):
    """Walk-forward over the OOS region; pool forecasts; compare PatchTST+GAT vs PatchTST (and HAR-X)."""
    hp = hp or PatchTSTHParams()
    t0 = time.time()
    panel = build_enriched_panel(files, wf.lookback, wf.horizon, keep_tickers)
    n = len(panel.anchors)
    test_start = wf.test_start if wf.test_start is not None else int(n * wf.test_frac)
    K = max(1, math.ceil((n - test_start) / wf.folds_target))
    folds = make_folds(n, test_start, K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)

    pooled = {"HAR": {}, "HAR-X": {}}
    pt_pool = [{} for _ in cfg.seeds]
    ptg_pool = [{} for _ in cfg.seeds]
    flat_tr = {k: [] for k in ("y",) + _MODELS}
    flat_va = {k: [] for k in ("y",) + _MODELS}
    lc = {gm: {"train": [], "val": [], "best_epoch": []} for gm in LEARNED}
    per_fold = []
    nfolds = len(folds)
    print(f"[{market} h{wf.horizon}] {nfolds} folds, K={K}, {len(keep_tickers)} nodes, "
          f"{len(cfg.seeds)} seeds -> starting PatchTST walk-forward", flush=True)
    for fi, fold in enumerate(folds):
        ft0 = time.time()
        upd, flats, ev = run_fold(panel, fold, wf, cfg, hp=hp)
        per_fold_s = time.time() - ft0
        eta = per_fold_s * (nfolds - fi - 1)
        print(f"[{market} h{wf.horizon}] fold {fi + 1}/{nfolds} done in {per_fold_s:.0f}s "
              f"(elapsed {(time.time() - t0) / 3600:.2f}h, eta ~{eta / 3600:.2f}h)", flush=True)
        pooled["HAR"].update(upd["HAR"])
        pooled["HAR-X"].update(upd["HAR-X"])
        for si, d in enumerate(upd["PatchTST_seeds"]):
            pt_pool[si].update(d)
        for si, d in enumerate(upd["PatchTST_GAT_seeds"]):
            ptg_pool[si].update(d)
        yt, pt = flats["train"]
        yv, pv = flats["val"]
        flat_tr["y"].append(yt)
        flat_va["y"].append(yv)
        for m in _MODELS:
            flat_tr[m].append(pt[m])
            flat_va[m].append(pv[m])
        for gm in LEARNED:
            for key in ("train", "val", "best_epoch"):
                lc[gm][key].append(ev["learning_curves"][gm][key])
        per_fold.append({k: ev[k] for k in ("idx", "n_train", "n_val", "n_forecast", "fit_diagnostics")})

    fl = cfg.qlike_floor
    pt_ens = RMR._ens(pt_pool)
    ptg_ens = RMR._ens(ptg_pool)
    preds = {"HAR": pooled["HAR"], "HAR-X": pooled["HAR-X"], "PatchTST": pt_ens,
             "PatchTST_wGAT_vol2pk": ptg_ens}
    metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
    ytr = np.concatenate(flat_tr["y"])
    yva = np.concatenate(flat_va["y"])
    train_metrics = {m: _agg_metrics(ytr, np.concatenate(flat_tr[m]), fl) for m in _MODELS}
    val_metrics = {m: _agg_metrics(yva, np.concatenate(flat_va[m]), fl) for m in _MODELS}
    fit_diagnostics = {m: OF.classify_fit(train_metrics[m], val_metrics[m], metrics[m]) for m in _MODELS}
    per_seed = {"PatchTST": RMR.seed_metric_stats(pt_pool, fl),
                "PatchTST_wGAT_vol2pk": RMR.seed_metric_stats(ptg_pool, fl)}
    dm = {"PatchTST_GAT_vs_PatchTST": RMR._dm_all(ptg_ens, pt_ens, wf.horizon, fl),
          "PatchTST_GAT_vs_HARX": RMR._dm_all(ptg_ens, pooled["HAR-X"], wf.horizon, fl),
          "PatchTST_vs_HARX": RMR._dm_all(pt_ens, pooled["HAR-X"], wf.horizon, fl),
          "HARX_vs_HAR": RMR._dm_all(pooled["HAR-X"], pooled["HAR"], wf.horizon, fl)}
    n_dates = len({k[1] for k in pooled["HAR"]})
    res = {
        "dataset": market, "horizon": wf.horizon, "backbone": "patchtst",
        "design": "expanding-window walk-forward PatchTST+GAT (masked enriched panel, periodic retrain)",
        "data_source": f"data/processed_enriched/{market}", "num_nodes": panel.N,
        "n_test_obs": metrics["HAR"]["n"], "n_oos_dates": n_dates, "n_folds": len(folds),
        "retrain_cadence_K": K, "folds_target": wf.folds_target, "val_tail": wf.val,
        "lookback": wf.lookback, "test_start_anchor": test_start, "n_anchors": n, "seeds": list(cfg.seeds),
        "patchtst_hparams": {"patch_len": hp.patch_len, "stride": hp.stride, "d_model": hp.d_model,
                             "n_heads": hp.n_heads, "depth": hp.depth, "ff_dim": hp.ff_dim, "pool": hp.pool},
        "config": {"epochs": cfg.epochs, "patience": cfg.patience, "min_epochs": cfg.min_epochs,
                   "batch_size": cfg.batch_size, "qlike_floor": fl},
        "metrics": metrics, "metrics_per_seed": per_seed, "dm_date_clustered": dm,
        "train_metrics": train_metrics, "val_metrics": val_metrics,
        "fit_diagnostics": fit_diagnostics, "learning_curves": lc,
        "fit_summary": _fit_summary(per_fold), "per_fold": per_fold,
        "seconds": round(time.time() - t0, 1)}
    # self-run the over/under-fit evidence gate with the CORRECT learned tuple (design §8) so evidence
    # is enforced in-code regardless of the shared gate's default LEARNED names.
    ok, probs = OF.check_result_evidence(res, learned=LEARNED)
    res["evidence_self_check"] = {"passed": ok, "problems": probs, "learned": list(LEARNED)}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def _fit_summary(per_fold):
    summary = {}
    for m in _MODELS:
        verdicts = [f["fit_diagnostics"][m]["status"] for f in per_fold]
        summary[m] = {"fold_verdicts": verdicts, "n_ok": verdicts.count("ok"),
                      "n_overfit": verdicts.count("overfit"), "n_underfit": verdicts.count("underfit"),
                      "n_unknown": verdicts.count("unknown")}
    return summary


def _resolve_files(pattern):  # pragma: no cover - trivial glob wrapper exercised via main
    return sorted(f for f in glob.glob(pattern) if "_rejections" not in Path(f).name)


def main():  # pragma: no cover - entry driver (multi-hour GPU run); logic covered via run_walkforward/tests
    ap = argparse.ArgumentParser(description="PatchTST+GAT multi-horizon walk-forward (HAR-X / PatchTST / PatchTST+GAT).")
    ap.add_argument("--market", choices=MARKETS, default="vn100")
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--lookback", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--folds-target", type=int, default=22)
    ap.add_argument("--patch-len", type=int, default=PatchTSTHParams.patch_len)
    ap.add_argument("--stride", type=int, default=PatchTSTHParams.stride)
    ap.add_argument("--d-model", type=int, default=PatchTSTHParams.d_model)
    ap.add_argument("--n-heads", type=int, default=PatchTSTHParams.n_heads)
    ap.add_argument("--depth", type=int, default=PatchTSTHParams.depth)
    ap.add_argument("--ff-dim", type=int, default=PatchTSTHParams.ff_dim)
    ap.add_argument("--pool", choices=["flatten", "mean"], default=PatchTSTHParams.pool)
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 1 seed, 8-ticker slice, 1 fold")
    ap.add_argument("--no-gpu-wait", action="store_true", help="skip nvidia-smi politeness poll")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hp = PatchTSTHParams(patch_len=a.patch_len, stride=a.stride, d_model=a.d_model, n_heads=a.n_heads,
                         depth=a.depth, ff_dim=a.ff_dim, pool=a.pool)
    files = _resolve_files(enriched_glob(a.market))
    if len(files) < 2:
        raise SystemExit(f"no enriched files under {enriched_glob(a.market)}")
    universe = frozen_universe(files, a.lookback, a.horizon)
    if a.smoke:
        wf = PatchTSTWFConfig(lookback=a.lookback, horizon=a.horizon, folds_target=1)
        cfg = training_config(epochs=2, patience=1, seeds=(42,), batch=a.batch)
        universe = universe[:8]
    else:
        wf = PatchTSTWFConfig(lookback=a.lookback, horizon=a.horizon, folds_target=a.folds_target)
        cfg = training_config(epochs=a.epochs, batch=a.batch)
    out = a.out if a.out is not None else str(default_out_path(a.horizon, a.market))
    if not a.no_gpu_wait:
        print("[gpu] polling nvidia-smi for a free GPU (util<15, VRAM<1200MiB) ...", flush=True)
        wait_for_gpu()
        print("[gpu] free -> starting PatchTST walk-forward", flush=True)
    res = run_walkforward(files, wf, cfg, universe, out_path=out, market=a.market, hp=hp)
    m = res["metrics"]
    q = res["dm_date_clustered"]["PatchTST_GAT_vs_PatchTST"].get("qlike", {})
    print(f"[patchtst] nodes={res['num_nodes']} folds={res['n_folds']} oos_dates={res['n_oos_dates']} "
          f"obs={m['HAR']['n']} h{res['horizon']}")
    print(f"[patchtst] pooled QLIKE  HAR-X={m['HAR-X']['qlike']:.4f}  PatchTST={m['PatchTST']['qlike']:.4f}  "
          f"PatchTST+GAT={m['PatchTST_wGAT_vol2pk']['qlike']:.4f}  HAR={m['HAR']['qlike']:.4f}")
    print(f"[patchtst] DM PatchTST+GAT_vs_PatchTST QLIKE p={q.get('p_value')} favors={q.get('favors')} "
          f"-> {out}  ({res['seconds']}s)")


if __name__ == "__main__":  # pragma: no cover
    main()
