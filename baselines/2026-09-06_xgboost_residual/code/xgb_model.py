"""XGBoost fitting + validation-only hyperparameter / guardrail selection (CPU, deterministic).

Every choice -- tree hyperparameters, early-stopping iteration count, and the residual guardrail
(alpha shrinkage + z-clip) -- is made on the VALIDATION split only; the test split never enters
selection. Trees use ``tree_method='hist'`` with a bounded ``n_jobs`` and a fixed seed, so a run is
reproducible (a determinism test pins identical predictions across two fits).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

import xgb_config as C

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import metrics as M  # noqa: E402  (shared QLIKE with the project positivity floor, read-only)

from xgb_oof import reconstruct  # noqa: E402


def _fit_xgb(x_tr, y_tr, x_va, y_va, params, seed=C.XGB_SEED):
    """Fit one XGBoost regressor with early stopping on the validation target (RMSE)."""
    model = xgb.XGBRegressor(
        tree_method="hist", n_jobs=C.XGB_N_JOBS, random_state=seed,
        n_estimators=C.XGB_MAX_ESTIMATORS, early_stopping_rounds=C.XGB_EARLY_STOPPING_ROUNDS,
        eval_metric="rmse", **params)
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    return model


def _floored(pred, floor, nfloor):
    """Apply the shared positivity floors (scalar ``floor`` then per-row ``nfloor``)."""
    return np.maximum(np.maximum(pred, floor), nfloor)


def tune_direct(x_tr, y_tr_true, x_va, y_va_true, nfloor_tr, nfloor_va, floor, seed=C.XGB_SEED):
    """Direct XGBoost on the LOG target; select tree params by validation QLIKE.

    Trains on ``log(max(y,floor))`` and early-stops on the val log target; the forecast is ``exp(pred)``
    floored. Returns ``dict`` with the fitted ``model``, chosen ``params``, ``val_qlike`` and
    ``best_iteration``.
    """
    ytr_log = np.log(np.maximum(y_tr_true, floor))
    yva_log = np.log(np.maximum(y_va_true, floor))
    best = None
    for params in C.XGB_GRID:
        model = _fit_xgb(x_tr, ytr_log, x_va, yva_log, params, seed)
        pv = _floored(np.exp(model.predict(x_va)), floor, nfloor_va)
        q = M.qlike(y_va_true, pv, floor)
        if best is None or q < best["val_qlike"]:
            best = {"model": model, "params": params, "val_qlike": float(q),
                    "best_iteration": int(getattr(model, "best_iteration", 0) or 0)}
    return best


def tune_residual(x_tr, z_tr, x_va, z_va, harx_va, y_va_true, nfloor_va, floor, seed=C.XGB_SEED):
    """HAR-X residual-ratio XGBoost; jointly select tree params + (alpha, clip) by validation QLIKE.

    ``z_tr`` is the OOF residual target on train cells; ``z_va`` the val residual (against the fold-train
    HAR-X ``harx_va``) used only for early stopping. For each grid candidate the val ``zhat`` is
    reconstructed as ``harx_va * exp(alpha*clip(zhat))`` over the guardrail grid and scored by val QLIKE;
    the global best (candidate, alpha, clip) is returned. ``alpha=0`` selected => the correction adds no
    value. Returns ``dict`` with ``model``, ``params``, ``alpha``, ``clip``, ``val_qlike``,
    ``best_iteration``.
    """
    best = None
    for params in C.XGB_GRID:
        model = _fit_xgb(x_tr, z_tr, x_va, z_va, params, seed)
        zhat_va = model.predict(x_va)
        for alpha in C.ALPHA_GRID:
            for clip in C.CLIP_GRID:
                pv = _floored(reconstruct(harx_va, zhat_va, alpha, clip), floor, nfloor_va)
                q = M.qlike(y_va_true, pv, floor)
                if best is None or q < best["val_qlike"]:
                    best = {"model": model, "params": params, "alpha": float(alpha), "clip": float(clip),
                            "val_qlike": float(q),
                            "best_iteration": int(getattr(model, "best_iteration", 0) or 0)}
    return best
