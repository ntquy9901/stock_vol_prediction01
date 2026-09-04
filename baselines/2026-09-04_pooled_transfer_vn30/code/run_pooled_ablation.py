"""Pooled/transfer ablation driver: both arms on the shared VN100 folds + paired DM + diff-in-diff.

One VN100 panel + one fold set. Arm 1 (pooled) trains all 102 nodes; Arm 0 (baseline) trains the 31
VN30 nodes. Both score the identical VN30 OOS grid. Headline = paired DM Arm1-vs-Arm0 (LSTM/VolGA,
3 loss bases). Secondary = diff-in-diff of gap(deep-HAR). Honest a-priori decision rule.

Run:  .venv_gpu_encode/Scripts/python.exe baselines/2026-09-04_pooled_transfer_vn30/code/run_pooled_ablation.py --horizon 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import pooled_panel as pp  # noqa: E402
import run_pooled_arm as ra  # noqa: E402
import pipeline_config as pc  # noqa: E402  (record the volume z-score window in the result JSON)
import run_masked_rich as RMR  # noqa: E402
from run_volga_walkforward import VolgaWFConfig, enriched_glob  # noqa: E402
from run_walkforward import training_config  # noqa: E402
from wf_enriched_panel import build_enriched_panel  # noqa: E402
from wf_folds import assert_no_leakage, make_folds  # noqa: E402

_DEEP = ("LSTM", "LSTM_wGAT_vol2pk")


def _build(horizon: int, folds_target: int, epochs: int = 16, lookback: int = 22):
    """Screen VN30, build the shared VN100 panel + folds, map VN30 indices, make the training config.

    ``lookback`` default 22 = the approved experiment value (a variation over canonical
    ``pc.LOOKBACK`` = 10), matching the delivered 2026-08-31 VolGA walk-forward (run with ``--lookback 22``).
    """
    vn30_tk = pp.screened_universe("vn30", lookback, horizon)
    vn100_files = glob.glob(enriched_glob("vn100"))
    vn100_keep = pp.screened_universe("vn100", lookback, horizon)
    panel = build_enriched_panel(vn100_files, lookback, horizon, vn100_keep)
    present = [t for t in vn30_tk if t in set(panel.tickers)]
    dropped = [t for t in vn30_tk if t not in set(panel.tickers)]
    if dropped:  # pragma: no cover  (defensive: VN30 ⊂ VN100, so never triggers on real data)
        print(f"[pooled] {len(dropped)} VN30 ticker(s) not in the screened VN100 panel, dropped from "
              f"score set: {', '.join(dropped)}", flush=True)
    score_idx = pp.vn30_index(panel, present)
    wf = VolgaWFConfig(lookback=lookback, horizon=horizon, folds_target=folds_target)
    n = len(panel.anchors)
    test_start = int(n * wf.test_frac)
    K = max(1, math.ceil((n - test_start) / wf.folds_target))
    folds = make_folds(n, test_start, K, wf.val, wf.horizon)
    cfg = training_config(epochs=epochs)
    return panel, folds, wf, cfg, np.arange(panel.N), score_idx


def _diff_in_diff(arm0, arm1):
    """gap(deep - HAR) per arm and its change; negative gap = deep better than HAR."""
    out = {}
    for m in _DEEP:
        g0 = arm0["metrics"][m]["qlike"] - arm0["metrics"]["HAR"]["qlike"]
        g1 = arm1["metrics"][m]["qlike"] - arm1["metrics"]["HAR"]["qlike"]
        out[m] = {"gap_arm0": g0, "gap_arm1": g1, "delta_gap": g1 - g0}
    return out


def run_ablation(horizon: int, folds_target: int = 22, epochs: int = 16, lookback: int = 22, out=None):
    t0 = time.time()
    panel, folds, wf, cfg, all_idx, score_idx = _build(horizon, folds_target, epochs, lookback)
    assert_no_leakage(folds, panel.target_dates, wf.horizon)
    print(f"[pooled] h{horizon}: {panel.N} panel nodes, {len(score_idx)} VN30 scored, "
          f"{len(folds)} folds, {len(cfg.seeds)} seeds -> Arm1 (pooled)", flush=True)
    arm1 = ra.run_arm(panel, folds, wf, cfg, all_idx, score_idx)
    print(f"[pooled] h{horizon}: Arm0 (VN30-only) after {(time.time() - t0) / 3600:.2f}h", flush=True)
    arm0 = ra.run_arm(panel, folds, wf, cfg, score_idx, score_idx)
    fl = cfg.qlike_floor
    paired_dm = {m: RMR._dm_all(arm1["preds"][m], arm0["preds"][m], horizon, fl) for m in _DEEP}
    result = {
        "horizon": horizon,
        "meta": {"panel_nodes": int(panel.N), "vn30_scored": int(len(score_idx)),
                 "n_folds": len(folds), "seeds": list(cfg.seeds), "lookback": wf.lookback,
                 "folds_target": folds_target, "epochs": epochs, "seconds": time.time() - t0,
                 "volume_zscore_window": pc.VOLUME_ZSCORE_WINDOW,
                 "design": "single VN100 panel; Arm0 train=VN30, Arm1 train=VN100; score=VN30"},
        "volume_zscore_window": pc.VOLUME_ZSCORE_WINDOW,
        "arm0": {"metrics": arm0["metrics"], "seed_stats": arm0["seed_stats"], "per_fold": arm0["per_fold"]},
        "arm1": {"metrics": arm1["metrics"], "seed_stats": arm1["seed_stats"], "per_fold": arm1["per_fold"]},
        "paired_dm": paired_dm,          # favors "A" => Arm1 (pooled) better
        "diff_in_diff": _diff_in_diff(arm0, arm1),
    }
    if out is None:  # pragma: no cover  (default path; tests pass an explicit out)
        out = REPO / "results" / "pooled_transfer_vn30" / f"pooled_vn30_h{horizon}.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[pooled] h{horizon}: wrote {out} ({result['meta']['seconds'] / 3600:.2f}h)", flush=True)
    result["arm0"]["preds"] = arm0["preds"]  # returned (not serialised) for callers/tests
    result["arm1"]["preds"] = arm1["preds"]
    return result


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10, 22])
    ap.add_argument("--folds-target", type=int, default=22)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lookback", type=int, default=22)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_ablation(a.horizon, a.folds_target, a.epochs, a.lookback, a.out)


if __name__ == "__main__":  # pragma: no cover
    main()
