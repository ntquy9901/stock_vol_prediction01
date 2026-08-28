"""Forecasting ablation: does replacing the Parkinson target with another daily variance estimator change
the HAR-X out-of-sample QLIKE/MSE? QLIKE is scale-invariant, so it is comparable across estimator targets.

For each panel we regenerate processed CSVs whose target column is a chosen estimator (computed from raw
OHLCV), rebuild the masked panel, fit the deterministic HAR-X baseline on TRAIN and score the TEST fold, and
report QLIKE/MSE + the fraction of test targets at/under the per-node floor (the QLIKE driver). HAR-X only
(deterministic, cheap); the deep models are not retrained here.

Usage: python scripts/eda/estimator_forecast_ablation.py --panels vn30 hnx --estimators parkinson rs_overnight garman_klass
"""
from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "garch_masked",
           REPO / "scripts" / "eda"):
    sys.path.insert(0, str(_p))

import masked_rich as MR          # noqa: E402
import compute_garch_masked as CG  # noqa: E402
import metrics as M               # noqa: E402
from config import Config         # noqa: E402
import volatility_estimators as VE  # noqa: E402
import floor_sensitivity as FS    # noqa: E402

PROC = {
    "vn30": REPO / "submission" / "soict_lstm_gat" / "data" / "vn30",
    "vn100": REPO / "submission" / "soict_lstm_gat" / "data" / "vn100",
    "hose": REPO / "data" / "processed" / "hose",
    "hnx": REPO / "data" / "processed" / "hnx",
    "sp500": REPO / "data" / "processed" / "sp500",
}
SCREEN = {"hose", "hnx", "sp500"}


def screened_tickers(panel):
    """Canonical liquidity-screened ticker universe for a panel, from the DELIVERED processed Parkinson files
    (the exact set used by the shipped result.json). None means 'keep all' (unscreened panels). Applying the
    SAME ticker set to every estimator makes the ablation comparable to the delivered numbers AND fair across
    estimators (the screen is defined on Parkinson H==L days; regenerating a different estimator must not
    change the universe)."""
    if panel not in SCREEN:
        return None
    kept = FS.screen_files(sorted(glob.glob(str(PROC[panel] / "*_processed.csv"))))
    return {Path(f).name.replace("_processed.csv", "") for f in kept}


def _write_estimator_processed(panel, estimator, out_dir, keep_tickers=None):
    """For every ticker with both a processed CSV and a raw OHLCV file, write a processed CSV whose
    'parkinson_volatility' column is the chosen estimator computed from raw OHLCV (aligned on date).
    ``keep_tickers`` (if given) restricts to the canonical screened universe."""
    price_dir = VE.PRICE[panel]
    written = []
    for pf in sorted(glob.glob(str(PROC[panel] / "*_processed.csv"))):
        tk = Path(pf).name.replace("_processed.csv", "")
        if keep_tickers is not None and tk not in keep_tickers:
            continue
        rf = price_dir / f"{tk}_ohlcv.csv"
        if not rf.exists():
            continue
        raw = pd.read_csv(rf)
        # External review H-04: the estimators are order-dependent (rolling/ewm/diff), so sort by date
        # and drop duplicate dates BEFORE computing them -- otherwise an unsorted/duplicated raw file
        # would both corrupt the rolling windows and mis-pair (date, value). No-op on the clean ETL data.
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        est = VE.estimators_from_ohlcv(raw)
        v = est[estimator].to_numpy(dtype=float)
        # FAIR comparison: keep the SAME date grid for every estimator (do NOT drop rows, which would
        # fragment the panel differently per estimator and confound the QLIKE). Floor non-positive/zero
        # values at a tiny epsilon so the only difference across estimators is the value on each day.
        v = np.where(np.isfinite(v), np.maximum(v, 1e-10), np.nan)
        df = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "parkinson_volatility": v})
        df = df.dropna(subset=["parkinson_volatility"])   # drops only invalid-price / first-overnight rows
        if len(df) < 260:
            continue
        df.to_csv(Path(out_dir) / f"{tk}_processed.csv", index=False)
        written.append(str(Path(out_dir) / f"{tk}_processed.csv"))
    return sorted(written)


def _harx_scores(files, price_dir, cfg, horizon):
    D = MR.build_masked_rich(files, str(price_dir), cfg.lookback, horizon)
    pred = CG._harx_pred(D, cfg)
    ks = sorted(pred)
    y = np.array([pred[k][0] for k in ks]); p = np.array([pred[k][1] for k in ks])
    # floored fraction: test targets at/under the shared per-node floor used in QLIKE scoring basis
    node_floor = 1e-2 * D.t_mean + 1e-12
    floor_by_key = np.array([node_floor[k[0]] for k in ks])
    return {
        "n_nodes": D.N, "n_obs": len(ks),
        "qlike": M.qlike(y, p, cfg.qlike_floor), "mse": M.mse(y, p), "mae": M.mae(y, p), "r2": M.r2(y, p),
        "true_floored_frac": float(np.mean(y <= floor_by_key)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=["vn30", "hnx"])
    ap.add_argument("--estimators", nargs="+", default=["parkinson", "rs_overnight", "garman_klass"])
    ap.add_argument("--horizon", type=int, default=1)
    a = ap.parse_args()
    cfg = Config()
    print(f"HAR-X forecasting ablation (h{a.horizon}); QLIKE is scale-invariant -> comparable across targets\n")
    print(f"{'panel':6} {'estimator':16} {'nodes':>5} {'obs':>7} {'QLIKE':>8} {'MSE':>10} {'R2':>7} {'floored%':>9}")
    for panel in a.panels:
        price_dir = VE.PRICE[panel]
        keep = screened_tickers(panel)           # fixed canonical universe (matches the delivered result.json)
        for est in a.estimators:
            with tempfile.TemporaryDirectory() as td:
                files = _write_estimator_processed(panel, est, td, keep_tickers=keep)
                if len(files) < 2:
                    print(f"{panel:6} {est:16} -- too few tickers"); continue
                s = _harx_scores(files, price_dir, cfg, a.horizon)
                print(f"{panel:6} {est:16} {s['n_nodes']:>5} {s['n_obs']:>7} {s['qlike']:>8.4f} "
                      f"{s['mse']:>10.3e} {s['r2']:>7.3f} {s['true_floored_frac']*100:>8.2f}%", flush=True)


if __name__ == "__main__":
    main()
