"""One arm's walk-forward over the shared VN100 folds (restrict training universe, score VN30)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pooled_panel as pp  # noqa: E402
from wf_enriched_panel import pack_fold  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
from run_walkforward import _har_ols_preds  # noqa: E402
import pipeline_config as pc  # noqa: E402

_MODELS = ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk")


def run_arm(panel, folds, wf, cfg, train_idx, score_idx):
    """Walk-forward one arm: train on ``train_idx`` nodes, score ``score_idx`` nodes.

    Arm 0 (baseline): ``train_idx = score_idx`` (VN30). Arm 1 (pooled): ``train_idx = arange(N)``.
    Returns pooled VN30 predictions (per model), pooled metrics, per-seed stats, and per-fold sizes.
    """
    fl = cfg.qlike_floor
    pooled = {"HAR": {}, "HAR-X": {}}
    lstm_pool = [{} for _ in cfg.seeds]
    volga_pool = [{} for _ in cfg.seeds]
    per_fold = []
    for fold in folds:
        D = pack_fold(panel, fold, wf.lookback, wf.horizon)
        Dr = pp.restrict_fold(D, train_idx)
        smask = pp.score_mask(D.tmask_te, score_idx)
        nfloor = pc.POS_FLOOR_FRAC * Dr.t_mean + pc.POS_FLOOR_EPS
        har, harx = _har_ols_preds(Dr, fl, nfloor)
        eye = np.eye(Dr.N, dtype=np.float32)
        lstm = [RMR.train_masked_rich(Dr, cfg, s, False, eye, return_splits=True) for s in cfg.seeds]
        volga = [RMR.train_masked_rich(Dr, cfg, s, True, Dr.adj_vol2pk, return_splits=True) for s in cfg.seeds]
        pooled["HAR"].update(RMR._pred_dict(har["te"], D.y_te, smask, D.d_te, D.N))
        pooled["HAR-X"].update(RMR._pred_dict(harx["te"], D.y_te, smask, D.d_te, D.N))
        for si, o in enumerate(lstm):
            lstm_pool[si].update(RMR._pred_dict(o["test"], D.y_te, smask, D.d_te, D.N))
        for si, o in enumerate(volga):
            volga_pool[si].update(RMR._pred_dict(o["test"], D.y_te, smask, D.d_te, D.N))
        per_fold.append({"idx": fold.idx, "n_train": int(Dr.tmask_tr.sum()),
                         "n_forecast": int(smask.sum())})
    lstm_ens = RMR._ens(lstm_pool)
    volga_ens = RMR._ens(volga_pool)
    preds = {"HAR": pooled["HAR"], "HAR-X": pooled["HAR-X"], "LSTM": lstm_ens,
             "LSTM_wGAT_vol2pk": volga_ens}
    metrics = {m: RMR._metrics(preds[m], fl) for m in _MODELS}
    seed_stats = {"LSTM": RMR.seed_metric_stats(lstm_pool, fl),
                  "LSTM_wGAT_vol2pk": RMR.seed_metric_stats(volga_pool, fl)}
    return {"metrics": metrics, "seed_stats": seed_stats, "preds": preds, "per_fold": per_fold}
