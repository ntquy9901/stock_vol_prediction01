"""Expanding-window walk-forward: HAR-X vs no-graph LSTM on the VN100 h1 OOS region.

Reuses the delivered no-graph ``MaskedRichNet`` / ``train_masked_rich`` + HAR/HAR-X OLS + the RMR
metric/DM/evidence helpers UNCHANGED (read-only imports); only the fold/panel construction is new
(see wf_folds / wf_panel). Each fold refits both models on all data before the retrain point, freezes
them, and forecasts the next K days 1-step-ahead. Predictions are pooled over the whole OOS region and
compared to the FIXED-split verdict with a date-clustered Diebold-Mariano.

Real run (GPU, ~1.5-2h):  python run_walkforward.py
Smoke:                     python run_walkforward.py --smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "quality_gate", str(_HERE)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import baselines as B          # noqa: E402  (har_fit / har_predict, read-only)
import run_masked_rich as RMR  # noqa: E402  (MaskedRichNet + train_masked_rich + metric/DM helpers)
import pipeline_config as pc   # noqa: E402  (single source of truth for tunable constants)
from config import Config      # noqa: E402

from wf_folds import assert_no_leakage, make_folds  # noqa: E402
from wf_panel import build_wf_panel, pack_fold        # noqa: E402

DATA_FILES = str(REPO / "submission" / "soict_lstm_gat" / "data" / "vn100" / "*_processed.csv")
PRICE_DIR = str(REPO / "data" / "raw" / "prices" / "vn100_vnstock")
OUT_PATH = REPO / "results" / "walkforward_harx_lstm" / "walkforward_vn100_h1.json"


@dataclass(frozen=True)
class WFConfig:
    lookback: int = pc.LOOKBACK
    horizon: int = pc.WF_HORIZON
    K: int = pc.WF_RETRAIN_K
    val: int = pc.WF_VAL_TAIL
    test_frac: float = pc.WF_TEST_FRAC
    test_start: int | None = None


def training_config(epochs: int = 16, patience: int = 5, batch: int = 32,
                    seeds=(42, 123, 2026, 7, 2024)) -> Config:
    """Delivered hyper-params overridden to the approved walk-forward budget."""
    return replace(Config(), epochs=epochs, patience=patience, min_epochs=min(5, epochs),
                   batch_size=batch, seeds=tuple(seeds))


# ----------------------------- GPU politeness -----------------------------

def _query_gpu():  # pragma: no cover - shells out to nvidia-smi
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()[0]
    util, mem = (float(x) for x in out.split(","))
    return util, mem


def gpu_is_free(util: float, mem_mib: float, util_max: float, mem_max_mib: float) -> bool:
    return util < util_max and mem_mib < mem_max_mib


def wait_for_gpu(query=_query_gpu, util_max: float = 15, mem_max_mib: float = 1200,
                 hold: int = 3, poll: float = 15.0, sleep=time.sleep, max_polls=None) -> bool:
    """Return True once ``hold`` consecutive samples show the GPU free; False if ``max_polls`` exhausted."""
    consecutive = polls = 0
    while max_polls is None or polls < max_polls:
        util, mem = query()
        if gpu_is_free(util, mem, util_max, mem_max_mib):
            consecutive += 1
            if consecutive >= hold:
                return True
        else:
            consecutive = 0
        polls += 1
        sleep(poll)
    return False


# ----------------------------- HAR / HAR-X (OLS, refit per fold) -----------------------------

def _har_ols_preds(D, floor: float, nfloor: np.ndarray):
    """HAR (3-feat) + HAR-X (5-feat) OLS refit on the fold's TRAIN rows -> floored train/val/test preds."""
    mtr = D.tmask_tr.astype(bool)
    coef = B.har_fit(D.har_tr[mtr], D.y_tr[mtr])
    har = {sp: np.maximum(B.har_predict(getattr(D, f"har_{sp}").reshape(-1, 3), coef, floor=floor)
                          .reshape(getattr(D, f"y_{sp}").shape), nfloor) for sp in ("tr", "va", "te")}
    xtr = np.column_stack([np.ones(int(mtr.sum())), D.har5_tr[mtr]])
    cx = np.linalg.lstsq(xtr, D.y_tr[mtr], rcond=None)[0]

    def hx(sp):
        f5 = getattr(D, f"har5_{sp}").reshape(-1, 5)
        return np.maximum((np.column_stack([np.ones(len(f5)), f5]) @ cx).reshape(getattr(D, f"y_{sp}").shape),
                          nfloor)

    harx = {sp: hx(sp) for sp in ("tr", "va", "te")}
    return har, harx


# ----------------------------- one fold -----------------------------

def run_fold(panel, fold, wf: WFConfig, cfg: Config):
    """Train HAR/HAR-X/LSTM on one fold, return pooled-update pred dicts + this fold's fit evidence."""
    D = pack_fold(panel, fold, wf.lookback, wf.horizon)
    fl = cfg.qlike_floor
    nfloor = 1e-2 * D.t_mean + 1e-12
    har, harx = _har_ols_preds(D, fl, nfloor)
    eye = np.eye(D.N, dtype=np.float32)
    lstm_outs = [RMR.train_masked_rich(D, cfg, s, False, eye, return_splits=True) for s in cfg.seeds]

    # forecast-region pred dicts (keyed by (node, date)) for pooling
    har_te = RMR._pred_dict(har["te"], D.y_te, D.tmask_te, D.d_te, D.N)
    harx_te = RMR._pred_dict(harx["te"], D.y_te, D.tmask_te, D.d_te, D.N)
    lstm_seed_te = [RMR._pred_dict(o["test"], D.y_te, D.tmask_te, D.d_te, D.N) for o in lstm_outs]

    # per-fold over/under-fit evidence (seed-ensembled splits + verdict + learning curves)
    lstm_tr = RMR._ens_split(lstm_outs, "train")
    lstm_va = RMR._ens_split(lstm_outs, "val")
    lstm_te = RMR._ens_split(lstm_outs, "test")
    tr_pred = {"HAR": har["tr"], "HAR-X": harx["tr"], "LSTM": lstm_tr}
    va_pred = {"HAR": har["va"], "HAR-X": harx["va"], "LSTM": lstm_va}
    te_pred = {"HAR": har["te"], "HAR-X": harx["te"], "LSTM": lstm_te}
    train_m = {m: RMR._split_metrics(tr_pred[m], D.y_tr, D.tmask_tr, fl) for m in tr_pred}
    val_m = {m: RMR._split_metrics(va_pred[m], D.y_va, D.tmask_va, fl) for m in va_pred}
    test_m = {m: RMR._split_metrics(te_pred[m], D.y_te, D.tmask_te, fl) for m in te_pred}
    fit = {m: RMR.OF.classify_fit(train_m[m], val_m[m], test_m[m]) for m in tr_pred}
    curves = {"train": [o["train_curve"] for o in lstm_outs], "val": [o["val_curve"] for o in lstm_outs],
              "best_epoch": [o["best_epoch"] for o in lstm_outs]}
    evidence = {"idx": fold.idx, "n_train": int(D.tmask_tr.sum()), "n_val": int(D.tmask_va.sum()),
                "n_forecast": int(D.tmask_te.sum()),
                "train_metrics": train_m, "val_metrics": val_m, "test_metrics": test_m,
                "fit_diagnostics": fit, "lstm_learning_curves": curves}
    return {"HAR": har_te, "HAR-X": harx_te, "LSTM_seeds": lstm_seed_te}, evidence


# ----------------------------- pooled run -----------------------------

def run_walkforward(files, price_dir, wf: WFConfig, cfg: Config, out_path=None, keep_tickers=None):
    """Walk-forward over the OOS region; pool forecasts; compare LSTM vs HAR-X (and HAR) with DM."""
    t0 = time.time()
    panel = build_wf_panel(files, price_dir, wf.lookback, wf.horizon, keep_tickers)
    n = len(panel.anchors)
    test_start = wf.test_start if wf.test_start is not None else int(n * wf.test_frac)
    folds = make_folds(n, test_start, wf.K, wf.val, wf.horizon)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)

    pooled = {"HAR": {}, "HAR-X": {}}
    lstm_seed_pool = [{} for _ in cfg.seeds]
    per_fold = []
    for fold in folds:
        upd, evidence = run_fold(panel, fold, wf, cfg)
        pooled["HAR"].update(upd["HAR"])
        pooled["HAR-X"].update(upd["HAR-X"])
        for si, d in enumerate(upd["LSTM_seeds"]):
            lstm_seed_pool[si].update(d)
        per_fold.append(evidence)

    fl = cfg.qlike_floor
    lstm_ens = RMR._ens(lstm_seed_pool)
    preds = {"HAR": pooled["HAR"], "HAR-X": pooled["HAR-X"], "LSTM": lstm_ens}
    metrics = {k: RMR._metrics(v, fl) for k, v in preds.items()}
    per_seed = {"LSTM": RMR.seed_metric_stats(lstm_seed_pool, fl)}
    dm = {"LSTM_vs_HARX": RMR._dm_all(lstm_ens, pooled["HAR-X"], wf.horizon, fl),
          "LSTM_vs_HAR": RMR._dm_all(lstm_ens, pooled["HAR"], wf.horizon, fl),
          "HARX_vs_HAR": RMR._dm_all(pooled["HAR-X"], pooled["HAR"], wf.horizon, fl)}
    fit_summary = _fit_summary(per_fold)
    n_dates = len({k[1] for k in pooled["HAR"]})
    res = {
        "dataset": "vn100", "horizon": wf.horizon, "design": "expanding-window walk-forward (periodic retrain)",
        "num_nodes": panel.N, "n_test_obs": metrics["HAR"]["n"], "n_oos_dates": n_dates,
        "n_folds": len(folds), "retrain_cadence_K": wf.K, "val_tail": wf.val, "lookback": wf.lookback,
        "test_start_anchor": test_start, "n_anchors": n, "seeds": list(cfg.seeds),
        "config": {"epochs": cfg.epochs, "patience": cfg.patience, "min_epochs": cfg.min_epochs,
                   "batch_size": cfg.batch_size, "qlike_floor": fl},
        "fixed_split_reference": {  # the verdict this run tests against (delivered vn100 h1)
            "source": "results/masked_rich_floor1e2/vn100_h1/result.json",
            "LSTM_qlike": 0.5783600705282465, "HARX_qlike": 0.5115274209400663,
            "dm_LSTM_vs_HARX_qlike_p": 0.0011416061059061138, "favors": "HAR-X"},
        "metrics_pooled": metrics, "metrics_per_seed": per_seed, "dm_date_clustered": dm,
        "fit_summary": fit_summary, "per_fold": per_fold,
        "seconds": round(time.time() - t0, 1)}
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def _fit_summary(per_fold):
    """Aggregate the per-fold fit verdicts into per-model counts (over/under-fit evidence roll-up)."""
    models = ("HAR", "HAR-X", "LSTM")
    summary = {}
    for m in models:
        verdicts = [f["fit_diagnostics"][m]["status"] for f in per_fold]
        summary[m] = {"fold_verdicts": verdicts,
                      "n_ok": verdicts.count("ok"), "n_overfit": verdicts.count("overfit"),
                      "n_underfit": verdicts.count("underfit"), "n_unknown": verdicts.count("unknown")}
    return summary


def _resolve_files(pattern):  # pragma: no cover - trivial glob wrapper exercised via main
    return sorted(glob.glob(pattern))


def main():  # pragma: no cover - entry driver (2h GPU run); logic covered via run_walkforward/tests
    ap = argparse.ArgumentParser(description="Walk-forward HAR-X vs no-graph LSTM (VN100 h1).")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 1 seed, K=120 (fast sanity)")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--no-gpu-wait", action="store_true", help="skip nvidia-smi politeness poll")
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args()
    files = _resolve_files(DATA_FILES)
    if len(files) < 2:
        raise SystemExit(f"no vn100 processed files under {DATA_FILES}")
    # freeze the 102-node universe from the delivered fixed split
    import masked_rich as MR
    D = MR.build_masked_rich(files, PRICE_DIR, 10, 1)
    keep = D.tickers
    if a.smoke:
        wf = WFConfig(K=120, val=40)
        cfg = training_config(epochs=2, patience=1, seeds=(42,))
    else:
        wf = WFConfig()
        cfg = training_config(epochs=a.epochs)
    if not a.no_gpu_wait:
        print("[gpu] polling nvidia-smi for a free GPU (util<15, VRAM<1200MiB) ...", flush=True)
        wait_for_gpu()
        print("[gpu] free -> starting walk-forward", flush=True)
    res = run_walkforward(files, PRICE_DIR, wf, cfg, out_path=a.out, keep_tickers=keep)
    m = res["metrics_pooled"]
    q = res["dm_date_clustered"]["LSTM_vs_HARX"].get("qlike", {})
    print(f"[wf] nodes={res['num_nodes']} folds={res['n_folds']} oos_dates={res['n_oos_dates']} "
          f"obs={m['HAR']['n']}")
    print(f"[wf] pooled QLIKE  HAR-X={m['HAR-X']['qlike']:.4f}  LSTM={m['LSTM']['qlike']:.4f}  "
          f"HAR={m['HAR']['qlike']:.4f}")
    print(f"[wf] DM LSTM_vs_HARX QLIKE p={q.get('p_value')} favors={q.get('favors')} "
          f"mean_diff={q.get('mean_diff')}  ({res['seconds']}s)")


if __name__ == "__main__":  # pragma: no cover
    main()
