"""Level-1 base models + leakage-safe OOF cross-fitting + level-2 simplex-QLIKE meta-optimiser (Hướng 4).

Four deliberately-diverse own-history base learners:
  * ``GBME`` — champion HistGradientBoosting gamma (reuses ``full_matrix.gbm``), OWN-8 + EARN.
  * ``XGB``  — XGBoost ``reg:gamma`` at champion-matched capacity, OWN-8 + EARN.
  * ``GLM``  — sklearn ``GammaRegressor`` (log-link, alpha=1.0 default) on z-scored OWN-8 + EARN.
  * ``HAR``  — OLS on har_daily/weekly/monthly (a deliberately MSE-trained, different member).

``oof_predict`` produces out-of-fold train predictions via an inner temporal K-fold (mirrors the OOF
pattern in ``baselines/2026-09-14_gbm_gnn_embed/code/embed.py::oof_train_z``): every train row is
predicted by a base model that did NOT train on it. ``fit_simplex_qlike`` picks non-negative simplex
weights minimising pooled QLIKE on those OOF predictions (the level-2 meta-learner).

Reuses ``full_matrix`` (champion GBM, floor, feature lists) and ``metrics.per_obs_qlike`` (champion
QLIKE) as single sources. No shared module is edited.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import GammaRegressor
from sklearn.preprocessing import StandardScaler

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402
import metrics as M  # noqa: E402  (resolves via FM's path chain -> submission/soict_lstm_gat/metrics.py)
import stacking_config as C  # noqa: E402

FL = FM.FL


# --------------------------------------------------------------------------- base predictors
# Uniform signature ``(tr, te, cols, seeds) -> np.ndarray`` so they are interchangeable in the OOF loop.
def predict_gbme(tr, te, cols, seeds):
    """Champion HGBR gamma (single source ``FM.gbm``), seed-ensembled, floored at ``FL``."""
    return np.mean([FM.gbm(tr, te, cols, s) for s in seeds], 0)


def _xgb_one(tr, te, cols, seed):
    """One XGBoost ``reg:gamma`` fit (champion-matched capacity), floored/capped to ``[FL, PRED_CAP]``."""
    y = np.maximum(tr["y"].to_numpy(float), FL)
    d_tr = xgb.DMatrix(tr[cols].to_numpy(float), label=y)
    d_te = xgb.DMatrix(te[cols].to_numpy(float))
    params = {"objective": "reg:gamma", "eta": C.XGB_LR, "max_leaves": C.XGB_MAX_LEAVES,
              "max_depth": C.XGB_MAX_DEPTH, "grow_policy": "lossguide", "lambda": C.XGB_L2,
              "min_child_weight": C.XGB_MIN_CHILD_WEIGHT, "seed": int(seed), "tree_method": "hist",
              "verbosity": 0}
    bst = xgb.train(params, d_tr, num_boost_round=C.XGB_N_ESTIMATORS)
    return np.clip(bst.predict(d_te), FL, C.PRED_CAP)


def predict_xgb(tr, te, cols, seeds):
    """Seed-ensembled XGBoost gamma."""
    return np.mean([_xgb_one(tr, te, cols, s) for s in seeds], 0)


def predict_glm(tr, te, cols, seeds):
    """Gamma GLM (log-link) on z-scored features; predictions clipped to ``[FL, PRED_CAP]``.

    ``seeds`` is accepted for a uniform predictor signature but unused (the GLM is deterministic)."""
    x_tr = tr[cols].to_numpy(float)
    sc = StandardScaler().fit(x_tr)
    y = np.maximum(tr["y"].to_numpy(float), FL)
    glm = GammaRegressor(alpha=C.GLM_ALPHA, max_iter=C.GLM_MAX_ITER).fit(sc.transform(x_tr), y)
    return np.clip(glm.predict(sc.transform(te[cols].to_numpy(float))), FL, C.PRED_CAP)


def predict_har(tr, te, cols, seeds):
    """OLS on har_daily/weekly/monthly (MSE-trained), floored at ``HAR_FLOOR_FRAC * mean(train y)``.

    ``cols``/``seeds`` are accepted for a uniform predictor signature but unused (HAR uses HAR-3 only)."""
    dm = lambda df: np.column_stack([np.ones(len(df)), df[FM.HAR].to_numpy(float)])  # noqa: E731
    y = np.maximum(tr["y"].to_numpy(float), FL)
    coef = np.linalg.lstsq(dm(tr), y, rcond=None)[0]
    floor = C.HAR_FLOOR_FRAC * y.mean()
    return np.clip(dm(te) @ coef, floor, C.PRED_CAP)


# Insertion order defines the column order of the OOF/base prediction matrix and the reported weights.
BASE = {"GBME": predict_gbme, "XGB": predict_xgb, "GLM": predict_glm, "HAR": predict_har}


# --------------------------------------------------------------------------- OOF cross-fitting
def inner_blocks(dates, k):
    """Split sorted unique ``dates`` into up to ``k`` contiguous temporal blocks (drop empties)."""
    return [b for b in np.array_split(np.sort(np.unique(dates)), k) if len(b)]


def oof_predict(trf, cols, seeds, predictors, k):
    """Out-of-fold base predictions on ``trf``: an inner temporal K-fold — every train row is predicted
    by a base model whose training dates EXCLUDE that row's date. Returns ``{name: array aligned to trf}``.

    Raises ``RuntimeError`` if any train row is left unfilled (fail-loud coverage guarantee)."""
    blocks = inner_blocks(trf["date"].to_numpy(), k)
    pos = {ix: i for i, ix in enumerate(trf.index.to_numpy())}
    oof = {m: np.full(len(trf), np.nan) for m in predictors}
    for held in blocks:
        is_held = trf["date"].isin(held)
        inner, held_df = trf[~is_held], trf[is_held]
        idxs = held_df.index.to_numpy()
        for m, fn in predictors.items():
            p = fn(inner, held_df, cols, seeds)
            for pi, ix in zip(p, idxs):
                oof[m][pos[ix]] = pi
    for m in predictors:
        if np.isnan(oof[m]).any():
            raise RuntimeError(f"OOF coverage gap: base {m!r} left some train rows unfilled")
    return oof


# --------------------------------------------------------------------------- level-2 meta-optimiser
def stack_qlike(w, P, y, floor, cap):
    """Pooled QLIKE of the weighted base blend ``clip(P @ w, floor, cap)`` — the meta objective."""
    p = np.clip(P @ w, floor, cap)
    return float(np.mean(M.per_obs_qlike(y, p, floor=floor)))


def fit_simplex_qlike(P, y, floor=FL, cap=None, maxiter=None):
    """Level-2 meta-learner: non-negative simplex weights (``w>=0``, ``sum w = 1``) minimising pooled
    QLIKE of ``P @ w`` (SLSQP, uniform start). Returns the normalised weight vector.

    A degenerate solver return (all-zero) falls back to uniform weights (fail-safe, not silent-zero)."""
    cap = C.PRED_CAP if cap is None else cap
    maxiter = C.SLSQP_MAXITER if maxiter is None else maxiter
    m = P.shape[1]
    w0 = np.full(m, 1.0 / m)
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * m
    res = minimize(stack_qlike, w0, args=(P, y, floor, cap), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": maxiter, "ftol": C.SLSQP_FTOL})
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(m, 1.0 / m)
