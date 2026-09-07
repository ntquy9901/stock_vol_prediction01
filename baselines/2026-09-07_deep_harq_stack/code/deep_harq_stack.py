"""Deep + HARQ stack: an equal-weight ensemble of VolGA (deep, nonlinear) and HAR-X-Q (linear, measurement-
error-corrected) beats HAR-X at the horizons where neither alone is significant. Their forecast errors are
decorrelated (deep vs linear-quarticity), so averaging cancels noise.

  stack = w * VolGA + (1-w) * HAR-X-Q
  - PRIMARY: w = 0.5 (equal weight — no fitting, no overfitting, the classic 1/N ensemble).
  - SECONDARY: w fit on the VALIDATION split (pooled, grid) per market×horizon, applied to TEST (no peeking).

HAR-X / HAR-X-Q are recomputed per cell on the exact canonical walk-forward folds via the reviewed
`harq_walkforward` baseline; VolGA is read from the canonical `qlike_anchor` cells (5-seed ensemble). All
share the QLIKE floor; significance = date-clustered Diebold-Mariano vs HAR-X. Writes
`results/deep_harq_stack/stack_{market}_h{h}.json`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
for _p in ("baselines/2026-09-07_harq/code", "baselines/2026-08-31_walkforward_volga/code",
           "baselines/2026-08-30_walkforward_harx_lstm/code", "baselines/2026-08-21_har_anchored_residual/code",
           "submission/soict_lstm_gat"):
    sys.path.insert(0, str(REPO / _p))
import harq_walkforward as H  # noqa: E402
import metrics as M  # noqa: E402
import pipeline_config as pc  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
import stats as ST  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold  # noqa: E402
from wf_folds import make_folds  # noqa: E402

WGRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)          # weight on VolGA for the val-fit variant
MIN_OVERLAP_FRAC = 0.5                                      # test cells must mostly survive the 3-way merge


def _require_overlap(n_aligned, n_volga):
    """Fail loud if the HAR-X/HAR-X-Q/VolGA (ticker,date) intersection dropped most VolGA test cells (a
    stale/mismatched cell dump), instead of silently comparing on a truncated set."""
    if n_aligned == 0 or n_aligned < MIN_OVERLAP_FRAC * n_volga:
        raise ValueError(f"test intersection too small: {n_aligned}/{n_volga} VolGA cells — "
                         f"stale or mismatched VolGA cells vs the recomputed HAR-X panel")
    return n_aligned


def _dict_by_ticker(pred, y, tmask, dates, tickers, N):
    d = RMR._pred_dict(pred, y, tmask, dates, N)
    return {(tickers[j], dt): v for (j, dt), v in d.items()}


def recompute_harx(market, horizon):
    """Per (ticker,date) HAR-X + HAR-X-Q predictions for BOTH val and test splits (canonical folds)."""
    import glob as _g
    files = _g.glob(enriched_glob(market))
    panel = build_enriched_panel(files, pc.LOOKBACK, horizon, frozen_universe(files, pc.LOOKBACK, horizon))
    harq = H.quarticity_panel(panel.feats); tk = list(panel.tickers)
    wf = VolgaWFConfig(lookback=pc.LOOKBACK, horizon=horizon, folds_target=7)
    n = len(panel.anchors); ts = int(n * wf.test_frac); K = max(1, math.ceil((n - ts) / wf.folds_target))
    fl = training_config(epochs=1, seeds=(0,), batch=32).qlike_floor
    out = {"val": {"HAR-X": {}, "HAR-X-Q": {}}, "test": {"HAR-X": {}, "HAR-X-Q": {}}}
    for fold in make_folds(n, ts, K, wf.val, wf.horizon):
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        nfloor = pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS
        mtr = D.tmask_tr.astype(bool)
        h5_tr = D.har5_tr.reshape(-1, 5)[mtr.reshape(-1)]; y_tr = D.y_tr[mtr]
        hq_tr = np.nan_to_num(harq[panel.anchors[fold.train]]).reshape(-1)[mtr.reshape(-1)]
        for split, anc, y_s, tm_s, d_s, h5_s in (
                ("val", fold.val, D.y_va, D.tmask_va, D.d_va, D.har5_va),
                ("test", fold.forecast, D.y_te, D.tmask_te, D.d_te, D.har5_te)):
            hq_s = np.nan_to_num(harq[panel.anchors[anc]]).reshape(-1)
            px = H._ols_predict(h5_tr, y_tr, h5_s.reshape(-1, 5), nfloor, y_s.shape)
            pq = H._ols_predict(np.column_stack([h5_tr, hq_tr]), y_tr,
                                np.column_stack([h5_s.reshape(-1, 5), hq_s]), nfloor, y_s.shape)
            out[split]["HAR-X"].update(_dict_by_ticker(px, y_s, tm_s, d_s, tk, D.N))
            out[split]["HAR-X-Q"].update(_dict_by_ticker(pq, y_s, tm_s, d_s, tk, D.N))
    return out, fl


def load_volga(market, horizon, split):
    vc = pd.read_parquet(REPO / "results" / "qlike_anchor" / "cells" / f"cells_{market}_qlike_none_h{horizon}.parquet")
    vc = vc[(vc["split"] == split) & (vc["model"] == "VolGA")]
    return {(r.ticker, r.date): r.y_pred for r in vc.itertuples()}


def _align(harx, harxq, volga):
    keys = [k for k in harxq if k in volga and k in harx]
    y = np.array([harxq[k][0] for k in keys]); dates = np.array([k[1] for k in keys])
    return (y, dates, np.array([harx[k][1] for k in keys]),
            np.array([harxq[k][1] for k in keys]), np.array([volga[k] for k in keys]))


def _fit_w(harx, harxq, volga, fl):
    """Weight on VolGA minimising pooled VALIDATION QLIKE (grid)."""
    y, _, _, hq, vg = _align(harx, harxq, volga)
    if len(y) == 0:                                         # pragma: no cover - non-empty in practice
        return 0.5
    losses = [M.qlike(y, np.maximum(w * vg + (1 - w) * hq, fl), floor=fl) for w in WGRID]
    return float(WGRID[int(np.argmin(losses))])


def run(market="vn100", horizon=5, out=None):
    hx, fl = recompute_harx(market, horizon)
    vg_va = load_volga(market, horizon, "val"); vg_te = load_volga(market, horizon, "test")
    w_fit = _fit_w(hx["val"]["HAR-X"], hx["val"]["HAR-X-Q"], vg_va, fl)
    y, dates, harx, harxq, volga = _align(hx["test"]["HAR-X"], hx["test"]["HAR-X-Q"], vg_te)
    _require_overlap(len(y), len(vg_te))                    # fail loud on a stale/misaligned VolGA dump
    base = M.per_obs_qlike(y, harx, floor=fl)
    res = {"experiment": "deep_harq_stack", "market": market, "horizon": horizon, "n_test": int(len(y)),
           "w_fit_on_val": w_fit, "qlike": {"HAR-X": M.qlike(y, harx, fl), "HAR-X-Q": M.qlike(y, harxq, fl),
                                            "VolGA": M.qlike(y, volga, fl)}, "stack": {}}
    for tag, w in (("equal_0.5", 0.5), ("val_fit", w_fit)):
        blend = np.maximum(w * volga + (1 - w) * harxq, fl)
        dm = ST.date_clustered_dm(M.per_obs_qlike(y, blend, floor=fl), base, dates, horizon)
        res["stack"][tag] = {"w": w, "qlike": M.qlike(y, blend, fl), "dm_vs_HARX_p": float(dm["p_value"]),
                             "beats_HARX_sig": bool(dm["p_value"] < 0.05 and dm["mean_diff"] < 0)}
    outp = Path(out) if out else REPO / "results" / "deep_harq_stack" / f"stack_{market}_h{horizon}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(outp, "w"), indent=1)
    s = res["stack"]["equal_0.5"]
    print(f"[stack] {market} h{horizon}: HAR-X={res['qlike']['HAR-X']:.4f} -> stack(0.5)={s['qlike']:.4f} "
          f"({(res['qlike']['HAR-X'] - s['qlike']) / res['qlike']['HAR-X'] * 100:+.2f}%) p={s['dm_vs_HARX_p']:.4f}"
          f"{'*' if s['beats_HARX_sig'] else ''}  | val-fit w={w_fit} p={res['stack']['val_fit']['dm_vs_HARX_p']:.4f}", flush=True)
    return res


def main():  # pragma: no cover - CLI driver
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="vn100")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 10, 22])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.market, a.horizon, a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
