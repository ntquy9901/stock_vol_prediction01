"""Stage-1 Gamma GLM (log-link) + Stage-2 XGBoost gamma residual with a base margin (Hướng 3).

The GLM is loss-consistent with the target (gamma deviance = QLIKE up to constants), so its linear predictor
``eta = log(mu)`` is an unbiased-under-QLIKE anchor that, being linear, extrapolates beyond the tree training
range. XGBoost boosts the residual over ``eta`` via ``set_base_margin`` so the final forecast is
``exp(eta_GLM + sum trees)``. Reuses the champion floor from ``full_matrix`` (single source).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import StandardScaler

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402
import glm_anchor_config as C  # noqa: E402

FL = FM.FL


def fit_glm_eta(trf, combo, cols, floor=FL):
    """Stage 1: fit a gamma GLM (log-link, IRLS on gamma deviance) on ``trf`` z-scored features, return its
    linear predictor ``eta = log(mu)`` for the train rows and for the ``combo`` rows. Loss-consistent anchor;
    the scaler + GLM see train rows only (causal)."""
    sc = StandardScaler().fit(trf[cols].to_numpy(float))
    x_tr = sc.transform(trf[cols].to_numpy(float))
    x_co = sc.transform(combo[cols].to_numpy(float))
    y_tr = np.maximum(trf["y"].to_numpy(float), floor)
    glm = GammaRegressor(alpha=C.GLM_ALPHA, max_iter=C.GLM_MAX_ITER).fit(x_tr, y_tr)
    # eta = linear predictor = log(mu) (log-link); compute it directly and clip to [log(floor), log(PRED_CAP)]
    # so an extreme z-scored row cannot push the exp-link base margin to +inf downstream.
    lo, hi = np.log(floor), np.log(C.PRED_CAP)
    eta_tr = np.clip(x_tr @ glm.coef_ + glm.intercept_, lo, hi)
    eta_co = np.clip(x_co @ glm.coef_ + glm.intercept_, lo, hi)
    return eta_tr.astype(float), eta_co.astype(float)


def xgb_gamma(trf, combo, cols, seed, base_margin_tr=None, base_margin_co=None, floor=FL):
    """Stage 2: XGBoost ``reg:gamma`` fit on ``trf``, predict ``combo`` (floored). If a base margin is given
    (train + combo, in log space), the trees boost the residual over it: ``pred = exp(margin + sum trees)``.
    Capacity mirrors the champion HGBR gamma booster."""
    y_tr = np.maximum(trf["y"].to_numpy(float), floor)
    d_tr = xgb.DMatrix(trf[cols].to_numpy(float), label=y_tr)
    d_co = xgb.DMatrix(combo[cols].to_numpy(float))
    if base_margin_tr is not None:
        d_tr.set_base_margin(np.asarray(base_margin_tr, float))
        d_co.set_base_margin(np.asarray(base_margin_co, float))
    params = {"objective": "reg:gamma", "eta": C.XGB_LR, "max_leaves": C.XGB_MAX_LEAVES,
              "max_depth": C.XGB_MAX_DEPTH, "grow_policy": "lossguide", "lambda": C.XGB_L2,
              "min_child_weight": C.XGB_MIN_CHILD_WEIGHT, "seed": int(seed), "tree_method": "hist",
              "verbosity": 0}
    bst = xgb.train(params, d_tr, num_boost_round=C.XGB_N_ESTIMATORS)
    return np.clip(bst.predict(d_co), floor, C.PRED_CAP)


def predict_xgb(trf, combo, cols, seeds, floor=FL):
    """Seed-ensembled plain XGBoost gamma (no base margin): mean floored prediction over ``seeds``."""
    return np.mean([xgb_gamma(trf, combo, cols, s, floor=floor) for s in seeds], 0)


def predict_glm_xgb(trf, combo, cols, seeds, floor=FL):
    """Seed-ensembled GLM-anchored XGBoost: fit the GLM once (deterministic), then boost the gamma residual
    over its base margin, averaged over the XGB ``seeds``."""
    eta_tr, eta_co = fit_glm_eta(trf, combo, cols, floor)
    return np.mean([xgb_gamma(trf, combo, cols, s, base_margin_tr=eta_tr, base_margin_co=eta_co, floor=floor)
                    for s in seeds], 0)
